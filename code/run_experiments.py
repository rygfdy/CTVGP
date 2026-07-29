"""Run the isolated CTVGP rebuttal-v2 experiment matrices.

Examples
--------
python code/run_experiments.py --config configs/smoke.json --experiment all
python code/run_experiments.py --config configs/full.json --experiment main --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Each process is a task-level worker.  Limiting BLAS to one thread prevents
# four workers from silently expanding into 32 competing numerical threads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy

from core import (
    GaussianTarget,
    TransportFit,
    basis_derivative_point,
    ep_probit_moments,
    estimate_log_orthant,
    evaluate_basis_method,
    evaluate_gaussian_method,
    evaluate_transport_method,
    expand_warm_start,
    family_rank,
    hard_reference_samples,
    make_derivative_target,
    optimize_transport,
    sample_transport,
    transport_covariance,
    warm_start_full,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
METRICS = ROOT / "metrics"
LOGS = ROOT / "logs"


def deterministic_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(joined) & 0x7FFFFFFF)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isnan(numeric):
            return None
        if np.isposinf(numeric):
            return "Infinity"
        if np.isneginf(numeric):
            return "-Infinity"
        return numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fit_summary(fit: TransportFit, selected: bool = False) -> dict[str, Any]:
    return {
        "family": fit.family,
        "rank": fit.rank,
        "map": fit.map_name,
        "map_scale": fit.map_scale,
        "validation_elbo": fit.validation_elbo,
        "normalized_gradient": fit.normalized_gradient,
        "adam_steps": fit.adam_steps,
        "lbfgs_iterations": fit.lbfgs_iterations,
        "lbfgs_function_evaluations": fit.lbfgs_function_evaluations,
        "wall_time_seconds": fit.wall_time_seconds,
        "plateau_passed": fit.plateau_passed,
        "gradient_passed": fit.gradient_passed,
        "independent_elbo_1": fit.independent_elbo_1,
        "independent_elbo_1_se": fit.independent_elbo_1_se,
        "independent_elbo_2": fit.independent_elbo_2,
        "independent_elbo_2_se": fit.independent_elbo_2_se,
        "independent_agreement_passed": fit.independent_agreement_passed,
        "converged": fit.converged,
        "failure_reason": fit.failure_reason,
        "initialization": fit.initialization,
        "selected": selected,
    }


def select_fit(fits: list[TransportFit]) -> TransportFit | None:
    eligible = [fit for fit in fits if fit.converged]
    if not eligible:
        return None
    return max(eligible, key=lambda fit: fit.validation_elbo)


def optimizer_arguments(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_steps": int(config["max_steps"]),
        "batch_size": int(config["batch_size"]),
        "check_every": int(config["check_every"]),
        "validation_samples": int(config["validation_samples"]),
        "polish_samples": int(config["polish_samples"]),
        "polish_maxiter": int(config["polish_maxiter"]),
        "polish_maxfun": int(config.get("polish_maxfun", 650)),
    }


def prepare_target(
    case: str,
    data_seed: int,
    m_constraints: int,
    evaluation: dict[str, Any],
    namespace: str,
) -> tuple[
    GaussianTarget,
    dict[str, Any],
    dict[str, Any],
    np.ndarray | None,
    dict[str, Any],
]:
    target, metadata = make_derivative_target(
        case, data_seed, m_constraints
    )
    stream_seed = deterministic_seed(
        namespace, case, data_seed, m_constraints, "normalizer"
    )
    normalizer = estimate_log_orthant(
        target,
        samples_per_seed=int(evaluation["ghk_samples"]),
        replicates=int(evaluation["ghk_replicates"]),
        seed=stream_seed,
    )
    reference: np.ndarray | None = None
    reference_diagnostics: dict[str, Any] = {
        "method": "not_run",
        "reliable": False,
    }
    if normalizer["reliable"]:
        logz = float(normalizer["primary"]["logz"])
        try:
            reference, reference_diagnostics = hard_reference_samples(
                target,
                int(evaluation["reference_samples"]),
                deterministic_seed(
                    namespace, case, data_seed, m_constraints, "reference"
                ),
                logz=logz,
            )
        except Exception as error:  # preserve target and explicit failure
            reference_diagnostics = {
                "method": "failed",
                "reliable": False,
                "failure_reason": repr(error),
            }
    return (
        target,
        metadata,
        normalizer,
        reference,
        reference_diagnostics,
    )


def fit_independent_restarts(
    target: GaussianTarget,
    family: str,
    map_name: str,
    map_scale: float,
    restarts: list[int],
    optimizer: dict[str, Any],
    namespace: str,
    case: str,
    data_seed: int,
    m_constraints: int,
) -> list[TransportFit]:
    fits: list[TransportFit] = []
    for restart in restarts:
        fit = optimize_transport(
            target,
            family,
            deterministic_seed(
                namespace,
                case,
                data_seed,
                m_constraints,
                family,
                map_name,
                restart,
            ),
            map_name=map_name,
            map_scale=map_scale,
            initialization="independent",
            **optimizer_arguments(optimizer),
        )
        fits.append(fit)
    return fits


def unavailable_row(method: str, reason: str) -> dict[str, Any]:
    return {
        "method": method,
        "metric_available": False,
        "unavailable_reason": reason,
        "reverse_kl": None,
        "reverse_kl_se": None,
        "forward_kl": None,
        "forward_kl_se": None,
        "tv": None,
        "tv_se": None,
        "average_marginal_w1": None,
        "sliced_w1": None,
        "mean_relative_error": None,
        "covariance_relative_error": None,
    }


def run_main_target(task: dict[str, Any]) -> dict[str, Any]:
    case = task["case"]
    data_seed = int(task["data_seed"])
    settings = task["settings"]
    evaluation = task["evaluation"]
    optimizer = task["optimizer"]
    m_constraints = int(settings["m_constraints"])
    target, metadata, normalizer, reference, reference_diagnostics = (
        prepare_target(
            case,
            data_seed,
            m_constraints,
            evaluation,
            "main",
        )
    )
    fits = fit_independent_restarts(
        target,
        settings["family"],
        settings["map"],
        1.0,
        list(settings["restarts"]),
        optimizer,
        "main",
        case,
        data_seed,
        m_constraints,
    )
    selected = select_fit(fits)
    restart_rows: list[dict[str, Any]] = []
    for restart, fit in zip(settings["restarts"], fits):
        row = fit_summary(fit, selected is fit)
        row.update(
            {
                "experiment": "main",
                "case": case,
                "data_seed": data_seed,
                "m_constraints": m_constraints,
                "restart": int(restart),
            }
        )
        restart_rows.append(row)
    common = {
        "experiment": "main",
        "case": case,
        "data_seed": data_seed,
        "m_constraints": m_constraints,
        "normalizer_reliable": bool(normalizer["reliable"]),
        "reference_reliable": bool(
            reference_diagnostics.get("reliable", False)
        ),
        "logz": (
            float(normalizer["primary"]["logz"])
            if normalizer["reliable"]
            else None
        ),
    }
    distance_rows: list[dict[str, Any]] = []
    reliable = bool(
        normalizer["reliable"]
        and reference is not None
        and reference_diagnostics.get("reliable", False)
    )
    if reliable:
        assert reference is not None
        logz = float(normalizer["primary"]["logz"])
        standard = evaluate_gaussian_method(
            "Standard GP",
            target,
            target,
            logz,
            reference,
            int(evaluation["candidate_samples"]),
            int(evaluation["tv_samples_per_component"]),
            int(evaluation["sliced_projections"]),
            deterministic_seed("main", case, data_seed, "standard_metrics"),
        )
        projection = dict(standard)
        projection["method"] = "Projection GP"
        basis_point, basis_diagnostics = basis_derivative_point(
            case, data_seed, np.asarray(metadata["constraints"])
        )
        basis = evaluate_basis_method(
            basis_point,
            reference,
            int(evaluation["sliced_projections"]),
            deterministic_seed("main", case, data_seed, "basis_metrics"),
        )
        try:
            ep_target, ep_diagnostics = ep_probit_moments(
                target, softness=0.35
            )
            ep = evaluate_gaussian_method(
                "EP-GP",
                ep_target,
                target,
                logz,
                reference,
                int(evaluation["candidate_samples"]),
                int(evaluation["tv_samples_per_component"]),
                int(evaluation["sliced_projections"]),
                deterministic_seed("main", case, data_seed, "ep_metrics"),
            )
        except Exception as error:
            ep_diagnostics = {
                "converged": False,
                "failure_reason": repr(error),
            }
            ep = unavailable_row("EP-GP", "EP failure")
        if selected is not None:
            ctvgp = evaluate_transport_method(
                selected,
                target,
                logz,
                reference,
                int(evaluation["elbo_samples"]),
                int(evaluation["candidate_samples"]),
                int(evaluation["tv_samples_per_component"]),
                int(evaluation["sliced_projections"]),
                deterministic_seed("main", case, data_seed, "ctvgp_metrics"),
            )
            ctvgp["metric_available"] = True
            ctvgp["unavailable_reason"] = ""
        else:
            ctvgp = unavailable_row(
                "CTVGP", "all three VI restarts failed convergence"
            )
        for row in [standard, projection, basis, ep, ctvgp]:
            row.setdefault("metric_available", True)
            row.setdefault("unavailable_reason", "")
            row.update(common)
            distance_rows.append(row)
    else:
        reason = (
            "normalizer unreliable"
            if not normalizer["reliable"]
            else "reference sampler unreliable"
        )
        for method in [
            "Standard GP",
            "Projection GP",
            "Basis GP",
            "EP-GP",
            "CTVGP",
        ]:
            row = unavailable_row(method, reason)
            row.update(common)
            distance_rows.append(row)
        basis_diagnostics = {"not_run": reason}
        ep_diagnostics = {"not_run": reason}
    return {
        "key": f"{case}_data{data_seed}_md{m_constraints}",
        "target_mean": target.mean,
        "target_covariance": target.covariance,
        "metadata": metadata,
        "normalizer": normalizer,
        "reference_diagnostics": reference_diagnostics,
        "basis_diagnostics": basis_diagnostics,
        "ep_diagnostics": ep_diagnostics,
        "restart_rows": restart_rows,
        "distance_rows": distance_rows,
        "fits": [fit.jsonable() for fit in fits],
    }


def run_capacity_target(task: dict[str, Any]) -> dict[str, Any]:
    case = task["case"]
    data_seed = int(task["data_seed"])
    m_constraints = int(task["m_constraints"])
    settings = task["settings"]
    evaluation = task["evaluation"]
    optimizer = task["optimizer"]
    target, metadata, normalizer, reference, reference_diagnostics = (
        prepare_target(
            case,
            data_seed,
            m_constraints,
            evaluation,
            "capacity",
        )
    )
    all_fits: dict[str, list[TransportFit]] = {}
    selected_by_family: dict[str, TransportFit | None] = {}
    previous_selected: TransportFit | None = None
    for family in settings["families"]:
        fits: list[TransportFit] = []
        for restart in settings["restarts"]:
            initial = None
            initialization = "independent"
            if int(restart) == 0 and previous_selected is not None:
                if family == "full":
                    initial = warm_start_full(
                        previous_selected, target.dimension
                    )
                else:
                    initial = expand_warm_start(
                        previous_selected,
                        family,
                        target.dimension,
                        deterministic_seed(
                            "capacity", case, data_seed, m_constraints, family
                        ),
                    )
                initialization = "nested_warm_start"
            fit = optimize_transport(
                target,
                family,
                deterministic_seed(
                    "capacity",
                    case,
                    data_seed,
                    m_constraints,
                    family,
                    restart,
                ),
                map_name=settings["map"],
                map_scale=1.0,
                initial_parameters=initial,
                initialization=initialization,
                **optimizer_arguments(optimizer),
            )
            fits.append(fit)
        selected = select_fit(fits)
        all_fits[family] = fits
        selected_by_family[family] = selected
        if selected is not None:
            previous_selected = selected
    restart_rows: list[dict[str, Any]] = []
    for family, fits in all_fits.items():
        for restart, fit in zip(settings["restarts"], fits):
            row = fit_summary(
                fit, selected_by_family[family] is fit
            )
            row.update(
                {
                    "experiment": "capacity",
                    "case": case,
                    "data_seed": data_seed,
                    "m_constraints": m_constraints,
                    "family": family,
                    "restart": int(restart),
                }
            )
            restart_rows.append(row)
    reliable = bool(
        normalizer["reliable"]
        and reference is not None
        and reference_diagnostics.get("reliable", False)
    )
    result_rows: list[dict[str, Any]] = []
    common = {
        "experiment": "capacity",
        "case": case,
        "data_seed": data_seed,
        "m_constraints": m_constraints,
        "normalizer_reliable": bool(normalizer["reliable"]),
        "reference_reliable": bool(
            reference_diagnostics.get("reliable", False)
        ),
    }
    for family in settings["families"]:
        selected = selected_by_family[family]
        if reliable and selected is not None:
            assert reference is not None
            metrics = evaluate_transport_method(
                selected,
                target,
                float(normalizer["primary"]["logz"]),
                reference,
                int(evaluation["elbo_samples"]),
                int(evaluation["candidate_samples"]),
                int(evaluation["tv_samples_per_component"]),
                int(evaluation["sliced_projections"]),
                deterministic_seed(
                    "capacity",
                    case,
                    data_seed,
                    m_constraints,
                    family,
                    "metrics",
                ),
            )
            metrics["metric_available"] = True
            metrics["unavailable_reason"] = ""
            metrics["normalized_reverse_kl"] = (
                metrics["reverse_kl"] / m_constraints
            )
        else:
            reason = (
                "no converged restart"
                if selected is None
                else "normalizer/reference unreliable"
            )
            metrics = unavailable_row("CTVGP", reason)
            metrics["normalized_reverse_kl"] = None
        metrics.update(common)
        metrics.update(
            {
                "family": family,
                "rank": family_rank(family),
                "selected_validation_elbo": (
                    selected.validation_elbo
                    if selected is not None
                    else None
                ),
                "selected_normalized_gradient": (
                    selected.normalized_gradient
                    if selected is not None
                    else None
                ),
                "selected_wall_time_seconds": (
                    selected.wall_time_seconds
                    if selected is not None
                    else None
                ),
                "converged_restarts": int(
                    sum(fit.converged for fit in all_fits[family])
                ),
                "restart_count": len(all_fits[family]),
            }
        )
        result_rows.append(metrics)
    nesting_rows: list[dict[str, Any]] = []
    families = list(settings["families"])
    for smaller, larger in zip(families[:-1], families[1:]):
        first = selected_by_family[smaller]
        second = selected_by_family[larger]
        if first is None or second is None:
            passed = False
            difference = None
            tolerance = None
        else:
            difference = second.independent_elbo_1 - first.independent_elbo_1
            tolerance = 3.0 * np.sqrt(
                second.independent_elbo_1_se**2
                + first.independent_elbo_1_se**2
            )
            passed = bool(difference >= -tolerance)
        nesting_rows.append(
            {
                **common,
                "smaller_family": smaller,
                "larger_family": larger,
                "elbo_difference": difference,
                "combined_3se": tolerance,
                "nesting_check_passed": passed,
                "interpretation": (
                    "consistent_with_rank_nesting"
                    if passed
                    else "optimization_failure_or_missing_fit"
                ),
            }
        )
    return {
        "key": f"{case}_data{data_seed}_md{m_constraints}",
        "target_mean": target.mean,
        "target_covariance": target.covariance,
        "metadata": metadata,
        "normalizer": normalizer,
        "reference_diagnostics": reference_diagnostics,
        "restart_rows": restart_rows,
        "result_rows": result_rows,
        "nesting_rows": nesting_rows,
        "fits": {
            family: [fit.jsonable() for fit in fits]
            for family, fits in all_fits.items()
        },
    }


def run_map_target(task: dict[str, Any]) -> dict[str, Any]:
    case = task["case"]
    data_seed = int(task["data_seed"])
    m_constraints = int(task["m_constraints"])
    settings = task["settings"]
    evaluation = task["evaluation"]
    optimizer = task["optimizer"]
    target, metadata, normalizer, reference, reference_diagnostics = (
        prepare_target(
            case, data_seed, m_constraints, evaluation, "map"
        )
    )
    derivative_scale = float(
        np.median(np.sqrt(np.diag(target.covariance)))
    )
    all_fits: dict[str, list[TransportFit]] = {}
    selected_by_map: dict[str, TransportFit | None] = {}
    for map_name in settings["maps"]:
        map_scale = (
            0.25 * derivative_scale
            if map_name == "squareplus"
            else 1.0
        )
        fits = fit_independent_restarts(
            target,
            settings["family"],
            map_name,
            map_scale,
            list(settings["restarts"]),
            optimizer,
            "map",
            case,
            data_seed,
            m_constraints,
        )
        all_fits[map_name] = fits
        selected_by_map[map_name] = select_fit(fits)
    restart_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    reliable = bool(
        normalizer["reliable"]
        and reference is not None
        and reference_diagnostics.get("reliable", False)
    )
    for map_name, fits in all_fits.items():
        selected = selected_by_map[map_name]
        for restart, fit in zip(settings["restarts"], fits):
            row = fit_summary(fit, selected is fit)
            row.update(
                {
                    "experiment": "map",
                    "case": case,
                    "data_seed": data_seed,
                    "m_constraints": m_constraints,
                    "family": settings["family"],
                    "map": map_name,
                    "derivative_scale": derivative_scale,
                    "restart": int(restart),
                }
            )
            restart_rows.append(row)
        if reliable and selected is not None:
            assert reference is not None
            metrics = evaluate_transport_method(
                selected,
                target,
                float(normalizer["primary"]["logz"]),
                reference,
                int(evaluation["elbo_samples"]),
                int(evaluation["candidate_samples"]),
                int(evaluation["tv_samples_per_component"]),
                int(evaluation["sliced_projections"]),
                deterministic_seed(
                    "map",
                    case,
                    data_seed,
                    m_constraints,
                    map_name,
                    "metrics",
                ),
            )
            metrics["metric_available"] = True
            metrics["unavailable_reason"] = ""
        else:
            reason = (
                "no converged restart"
                if selected is None
                else "normalizer/reference unreliable"
            )
            metrics = unavailable_row("CTVGP", reason)
        metrics.update(
            {
                "experiment": "map",
                "case": case,
                "data_seed": data_seed,
                "m_constraints": m_constraints,
                "family": settings["family"],
                "map": map_name,
                "derivative_scale": derivative_scale,
                "squareplus_alpha_over_sd": (
                    0.25 if map_name == "squareplus" else None
                ),
                "selected_validation_elbo": (
                    selected.validation_elbo
                    if selected is not None
                    else None
                ),
                "selected_normalized_gradient": (
                    selected.normalized_gradient
                    if selected is not None
                    else None
                ),
                "selected_wall_time_seconds": (
                    selected.wall_time_seconds
                    if selected is not None
                    else None
                ),
                "converged_restarts": int(
                    sum(fit.converged for fit in fits)
                ),
                "restart_count": len(fits),
                "normalizer_reliable": bool(normalizer["reliable"]),
                "reference_reliable": bool(
                    reference_diagnostics.get("reliable", False)
                ),
            }
        )
        result_rows.append(metrics)
    return {
        "key": f"{case}_data{data_seed}_md{m_constraints}",
        "target_mean": target.mean,
        "target_covariance": target.covariance,
        "metadata": metadata,
        "normalizer": normalizer,
        "reference_diagnostics": reference_diagnostics,
        "restart_rows": restart_rows,
        "result_rows": result_rows,
        "fits": {
            map_name: [fit.jsonable() for fit in fits]
            for map_name, fits in all_fits.items()
        },
    }


def execute_tasks(
    experiment: str,
    tasks: list[dict[str, Any]],
    worker,
    workers: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    started = time.perf_counter()
    if workers <= 1:
        for index, task in enumerate(tasks, 1):
            result = worker(task)
            output.append(result)
            print(
                f"[{experiment}] {index}/{len(tasks)} {result['key']} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        return output
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            output.append(result)
            completed += 1
            print(
                f"[{experiment}] {completed}/{len(tasks)} {result['key']} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return output


def persist_results(
    experiment: str, results: list[dict[str, Any]]
) -> None:
    restart_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    nesting_rows: list[dict[str, Any]] = []
    raw_root = RAW / experiment
    raw_root.mkdir(parents=True, exist_ok=True)
    for result in sorted(results, key=lambda item: item["key"]):
        key = result["key"]
        np.savez_compressed(
            raw_root / f"{key}_target.npz",
            mean=result["target_mean"],
            covariance=result["target_covariance"],
        )
        raw_payload = {
            key_name: value
            for key_name, value in result.items()
            if key_name
            not in {
                "target_mean",
                "target_covariance",
                "restart_rows",
                "distance_rows",
                "result_rows",
                "nesting_rows",
            }
        }
        write_json(raw_root / f"{key}.json", raw_payload)
        restart_rows.extend(result["restart_rows"])
        if experiment == "main":
            primary_rows.extend(result["distance_rows"])
        else:
            primary_rows.extend(result["result_rows"])
        nesting_rows.extend(result.get("nesting_rows", []))
    write_csv(
        METRICS / f"{experiment}_restart_metrics.csv",
        restart_rows,
    )
    write_csv(
        METRICS
        / (
            "main_distance_raw.csv"
            if experiment == "main"
            else f"{experiment}_raw.csv"
        ),
        primary_rows,
    )
    if nesting_rows:
        write_csv(
            METRICS / "capacity_nesting_checks.csv",
            nesting_rows,
        )


def build_tasks(
    experiment: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    settings = config[experiment]
    tasks: list[dict[str, Any]] = []
    if experiment == "main":
        for case in settings["cases"]:
            for data_seed in settings["data_seeds"]:
                tasks.append(
                    {
                        "case": case,
                        "data_seed": data_seed,
                        "settings": settings,
                        "evaluation": config["evaluation"],
                        "optimizer": config["optimizer"],
                    }
                )
    else:
        for case in settings["cases"]:
            for data_seed in settings["data_seeds"]:
                for m_constraints in settings["m_constraints"]:
                    tasks.append(
                        {
                            "case": case,
                            "data_seed": data_seed,
                            "m_constraints": m_constraints,
                            "settings": settings,
                            "evaluation": config["evaluation"],
                            "optimizer": config["optimizer"],
                        }
                    )
    return tasks


def capture_environment(config_path: Path, workers: int) -> None:
    payload = {
        "timestamp_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "workers": workers,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "config": str(config_path.resolve()),
        "cwd": str(Path.cwd().resolve()),
        "scope_root": str(ROOT.resolve()),
    }
    write_json(ROOT / "environment.json", payload)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "smoke.json",
    )
    parser.add_argument(
        "--experiment",
        choices=["main", "capacity", "map", "all"],
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.log_file is not None:
        log_path = arguments.log_file.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle
    config_path = arguments.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ROOT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    capture_environment(config_path, arguments.workers)
    experiments = (
        ["main", "capacity", "map"]
        if arguments.experiment == "all"
        else [arguments.experiment]
    )
    workers_by_experiment = {
        "main": run_main_target,
        "capacity": run_capacity_target,
        "map": run_map_target,
    }
    for experiment in experiments:
        tasks = build_tasks(experiment, config)
        print(
            f"Starting {experiment}: {len(tasks)} targets, "
            f"{arguments.workers} workers",
            flush=True,
        )
        results = execute_tasks(
            experiment,
            tasks,
            workers_by_experiment[experiment],
            arguments.workers,
        )
        persist_results(experiment, results)
        print(f"Persisted {experiment} results", flush=True)


if __name__ == "__main__":
    main()
