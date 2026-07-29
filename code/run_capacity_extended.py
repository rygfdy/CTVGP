"""Evaluate every capacity cell, retaining explicit reliability labels.

This script is deliberately separate from the strict preregistered pipeline.
It reads the completed capacity fits, reruns the hard-reference sampling after
the acceptance-based sampler fix, and reports exploratory metrics even when
the reference or selected VI fit fails a strict diagnostic.

Outputs
-------
metrics/capacity_extended_raw.csv
metrics/capacity_extended_summary.csv
tables/table_capacity_extended.md
logs/capacity_extended_diagnostics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

from core import (
    GaussianTarget,
    TransportFit,
    evaluate_transport_method,
    hard_reference_samples,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "capacity"
METRICS = ROOT / "metrics"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"

METRIC_NAMES = (
    "normalized_reverse_kl",
    "tv",
    "average_marginal_w1",
    "sliced_w1",
)


def deterministic_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(joined) & 0x7FFFFFFF)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_fit(payload: dict[str, Any]) -> TransportFit:
    values = dict(payload)
    values["parameters"] = np.asarray(values["parameters"], dtype=np.float64)
    values["history"] = list(values.get("history", []))
    return TransportFit(**values)


def choose_fit(payloads: list[dict[str, Any]]) -> tuple[TransportFit, str]:
    fits = [load_fit(payload) for payload in payloads]
    converged = [fit for fit in fits if fit.converged]
    if converged:
        return (
            max(converged, key=lambda fit: fit.validation_elbo),
            "best_converged_validation_elbo",
        )
    finite = [fit for fit in fits if np.isfinite(fit.validation_elbo)]
    if not finite:
        raise RuntimeError("no finite VI fit is available")
    return (
        max(finite, key=lambda fit: fit.validation_elbo),
        "best_unconverged_validation_elbo",
    )


def process_target(
    json_path: str,
    evaluation: dict[str, int],
) -> dict[str, Any]:
    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_payload = np.load(path.with_name(path.stem + "_target.npz"))
    target = GaussianTarget.from_moments(
        target_payload["mean"], target_payload["covariance"]
    )
    metadata = payload["metadata"]
    case = str(metadata["benchmark"])
    data_seed = int(metadata["data_seed"])
    m_constraints = int(metadata["m_constraints"])
    normalizer = payload["normalizer"]
    logz = float(normalizer["primary"]["logz"])
    normalizer_reliable = as_bool(normalizer.get("reliable", False))

    reference, reference_diagnostics = hard_reference_samples(
        target,
        int(evaluation["reference_samples"]),
        deterministic_seed(
            "capacity_extended",
            case,
            data_seed,
            m_constraints,
            "reference",
        ),
        logz=logz,
    )
    reference_reliable = as_bool(
        reference_diagnostics.get("reliable", False)
    )

    rows: list[dict[str, Any]] = []
    for family in ("diag", "rank2", "rank4", "full"):
        fit, selection = choose_fit(payload["fits"][family])
        metrics = evaluate_transport_method(
            fit,
            target,
            logz,
            reference,
            int(evaluation["elbo_samples"]),
            int(evaluation["candidate_samples"]),
            int(evaluation["tv_samples_per_component"]),
            int(evaluation["sliced_projections"]),
            deterministic_seed(
                "capacity_extended",
                case,
                data_seed,
                m_constraints,
                family,
                "metrics",
            ),
        )
        metrics["normalized_reverse_kl"] = (
            float(metrics["reverse_kl"]) / m_constraints
        )
        reliability_flags: list[str] = []
        if not normalizer_reliable:
            reliability_flags.append("U_NORM")
        if not reference_reliable:
            reliability_flags.append("U_REF")
        if not fit.converged:
            reliability_flags.append("U_VI")
        row = {
            "experiment": "capacity_extended",
            "case": case,
            "data_seed": data_seed,
            "m_constraints": m_constraints,
            "family": family,
            "selection": selection,
            "strictly_reliable": not reliability_flags,
            "reliability_code": (
                "OK" if not reliability_flags else "+".join(reliability_flags)
            ),
            "normalizer_reliable": normalizer_reliable,
            "reference_reliable": reference_reliable,
            "reference_method": reference_diagnostics.get("method", ""),
            "reference_rhat_max": reference_diagnostics.get(
                "split_rhat_max", ""
            ),
            "reference_ess_min": reference_diagnostics.get("ess_min", ""),
            "vi_converged": fit.converged,
            "vi_failure_reason": fit.failure_reason,
            "selected_validation_elbo": fit.validation_elbo,
            "selected_normalized_gradient": fit.normalized_gradient,
            **metrics,
        }
        rows.append(row)
    return {
        "key": f"{case}_data{data_seed}_md{m_constraints}",
        "rows": rows,
        "diagnostics": {
            "case": case,
            "data_seed": data_seed,
            "m_constraints": m_constraints,
            "normalizer": normalizer,
            "reference": reference_diagnostics,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(
    values: np.ndarray, seed: int, replicates: int
) -> tuple[float, float]:
    random = np.random.default_rng(seed)
    count = len(values)
    indices = random.integers(0, count, size=(replicates, count))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    rows: list[dict[str, Any]], bootstrap_replicates: int
) -> list[dict[str, Any]]:
    family_order = {"diag": 0, "rank2": 1, "rank4": 2, "full": 3}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["case"]),
            int(row["m_constraints"]),
            str(row["family"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            family_order[item[0][2]],
        ),
    )
    for (case, dimension, family), frame in ordered_groups:
        result: dict[str, Any] = {
            "case": case,
            "m_constraints": dimension,
            "family": family,
            "n_seeds": len(frame),
            "strictly_reliable_seeds": sum(
                as_bool(row["strictly_reliable"]) for row in frame
            ),
            "normalizer_reliable_seeds": sum(
                as_bool(row["normalizer_reliable"]) for row in frame
            ),
            "reference_reliable_seeds": sum(
                as_bool(row["reference_reliable"]) for row in frame
            ),
            "vi_converged_seeds": sum(
                as_bool(row["vi_converged"]) for row in frame
            ),
            "reliability_codes": ";".join(
                sorted({str(row["reliability_code"]) for row in frame})
            ),
        }
        for metric in METRIC_NAMES:
            values = np.asarray(
                [float(row[metric]) for row in frame], dtype=np.float64
            )
            low, high = bootstrap_interval(
                values,
                deterministic_seed(
                    "capacity_extended_summary",
                    case,
                    dimension,
                    family,
                    metric,
                ),
                bootstrap_replicates,
            )
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
        output.append(result)
    return output


def format_interval(row: dict[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    low = float(row[f"{metric}_ci95_low"])
    high = float(row[f"{metric}_ci95_high"])
    if metric in {"average_marginal_w1", "sliced_w1"}:
        return f"{mean:.4f} [{low:.4f}, {high:.4f}]"
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def write_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Approximation capacity versus constraint dimension and auxiliary family",
        "",
        "All cells are computed. Intervals are 10,000-replicate bootstrap 95% "
        "CIs over the five data seeds. Values from any seed failing a strict "
        "diagnostic are retained as exploratory estimates and marked in the "
        "Reliability column.",
        "",
        "| Case | $m_d$ | Family | Reverse KL/$m_d$ | TV | Avg. marginal W1 | "
        "Sliced W1 | Reliability |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        total = int(row["n_seeds"])
        strict = int(row["strictly_reliable_seeds"])
        reference = int(row["reference_reliable_seeds"])
        vi = int(row["vi_converged_seeds"])
        reliability = (
            f"OK ({strict}/{total})"
            if strict == total
            else (
                f"UNRELIABLE: strict {strict}/{total}; "
                f"reference {reference}/{total}; VI {vi}/{total}"
            )
        )
        lines.append(
            "| {case} | {dimension} | {family} | {reverse} | {tv} | "
            "{marginal} | {sliced} | {reliability} |".format(
                case=row["case"],
                dimension=row["m_constraints"],
                family=row["family"],
                reverse=format_interval(row, "normalized_reverse_kl"),
                tv=format_interval(row, "tv"),
                marginal=format_interval(row, "average_marginal_w1"),
                sliced=format_interval(row, "sliced_w1"),
                reliability=reliability,
            )
        )
    lines.extend(
        [
            "",
            "Reliability codes in the raw CSV: `U_NORM` = orthant normalizer "
            "cross-check failed; `U_REF` = truncated-Gaussian reference failed "
            "$\\widehat R<1.05$ and/or ESS $\\ge400$; `U_VI` = no restart "
            "passed every VI convergence criterion. `OK` means all three "
            "checks passed. An unreliable number is descriptive only and "
            "must not support a confirmatory claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "full.json"
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    evaluation = {
        name: int(value) for name, value in config["evaluation"].items()
    }
    paths = sorted(str(path) for path in RAW.glob("*.json"))
    if not paths:
        raise RuntimeError("no completed capacity raw files found")

    results: list[dict[str, Any]] = []
    if arguments.workers <= 1:
        for index, path in enumerate(paths, 1):
            result = process_target(path, evaluation)
            results.append(result)
            print(f"[{index}/{len(paths)}] {result['key']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            futures = {
                executor.submit(process_target, path, evaluation): path
                for path in paths
            }
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(paths)}] {result['key']}", flush=True)

    rows = [
        row
        for result in sorted(results, key=lambda item: item["key"])
        for row in result["rows"]
    ]
    summary = summarize(rows, int(arguments.bootstrap_replicates))
    METRICS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS / "capacity_extended_raw.csv", rows)
    write_csv(METRICS / "capacity_extended_summary.csv", summary)
    write_markdown(TABLES / "table_capacity_extended.md", summary)
    diagnostics = [
        result["diagnostics"]
        for result in sorted(results, key=lambda item: item["key"])
    ]
    (LOGS / "capacity_extended_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows)} rows and {len(summary)} summary cells",
        flush=True,
    )


if __name__ == "__main__":
    main()
