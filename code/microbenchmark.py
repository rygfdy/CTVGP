

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import psutil

from core import (
    GaussianTarget,
    TransportFit,
    draw_auxiliary,
    elbo_and_gradient,
    family_rank,
    initial_transport_parameters,
    make_derivative_target,
    positive_map_forward,
    project_transport_parameters,
    sample_transport,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
RAW = ROOT / "raw" / "microbenchmark"
MD_VALUES = [5, 10, 20, 40, 80, 160]
FAMILIES = ["diag", "rank2", "rank4", "rank8", "full"]


def timed(operation: Callable[[], object]) -> float:
    start = time.perf_counter()
    operation()
    return time.perf_counter() - start


def monitored_peak_rss(operation: Callable[[], object]) -> float:
    process = psutil.Process()
    stop = threading.Event()
    peak = [process.memory_info().rss]

    def monitor() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.001)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        operation()
    finally:
        stop.set()
        thread.join()
        peak[0] = max(peak[0], process.memory_info().rss)
    return peak[0] / (1024.0 * 1024.0)


def make_fit(
    target: GaussianTarget, family: str, seed: int
) -> TransportFit:
    parameters = initial_transport_parameters(
        target, family, seed, "softplus", 1.0
    )
    return TransportFit(
        family=family,
        rank=family_rank(family),
        map_name="softplus",
        map_scale=1.0,
        parameters=parameters,
        validation_elbo=0.0,
        normalized_gradient=0.0,
        adam_steps=0,
        lbfgs_iterations=0,
        wall_time_seconds=0.0,
        plateau_passed=True,
        gradient_passed=True,
        independent_elbo_1=0.0,
        independent_elbo_1_se=0.0,
        independent_elbo_2=0.0,
        independent_elbo_2_se=0.0,
        independent_agreement_passed=True,
        converged=True,
        failure_reason="",
        history=[],
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(repeats: int = 7) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dimension in MD_VALUES:
        # Warmup and seven independent repeats of exact-GP posterior creation.
        make_derivative_target("log", 0, dimension)
        for repeat in range(repeats):
            elapsed = timed(
                lambda: make_derivative_target("log", 0, dimension)
            )
            rows.append(
                {
                    "m_constraints": dimension,
                    "family": "shared",
                    "component": "posterior_construction",
                    "repeat": repeat,
                    "value": elapsed,
                    "unit": "seconds",
                    "batch_size": 0,
                    "iterations": 1,
                }
            )
        target, _ = make_derivative_target("log", 0, dimension)
        GaussianTarget.from_moments(target.mean, target.covariance)
        for repeat in range(repeats):
            elapsed = timed(
                lambda: GaussianTarget.from_moments(
                    target.mean, target.covariance
                )
            )
            rows.append(
                {
                    "m_constraints": dimension,
                    "family": "shared",
                    "component": "target_factorization",
                    "repeat": repeat,
                    "value": elapsed,
                    "unit": "seconds",
                    "batch_size": 0,
                    "iterations": 1,
                }
            )
        for family in FAMILIES:
            if family == "full" and dimension > 80:
                continue
            rank = family_rank(family)
            fit = make_fit(target, family, 71 + dimension)
            rng = np.random.default_rng(919 + dimension + max(rank, 0))
            diagonal_512 = rng.standard_normal((512, dimension))
            rank_512 = (
                rng.standard_normal((512, rank))
                if rank > 0
                else None
            )
            diagonal_8 = rng.standard_normal((8, dimension))
            rank_8 = (
                rng.standard_normal((8, rank)) if rank > 0 else None
            )

            def objective_512() -> object:
                return elbo_and_gradient(
                    fit.parameters,
                    target,
                    family,
                    diagonal_512,
                    rank_512,
                    "softplus",
                    1.0,
                )

            objective_512()
            for repeat in range(repeats):
                elapsed = timed(objective_512)
                rows.append(
                    {
                        "m_constraints": dimension,
                        "family": family,
                        "component": "objective_gradient",
                        "repeat": repeat,
                        "value": elapsed,
                        "unit": "seconds",
                        "batch_size": 512,
                        "iterations": 1,
                    }
                )

            def thousand_iterations() -> None:
                parameters = fit.parameters.copy()
                for _ in range(1000):
                    _, gradient = elbo_and_gradient(
                        parameters,
                        target,
                        family,
                        diagonal_8,
                        rank_8,
                        "softplus",
                        1.0,
                    )
                    # A tiny update includes parameter projection while keeping
                    # the benchmark in a stable local numerical regime.
                    parameters = project_transport_parameters(
                        parameters + 1e-8 * gradient,
                        dimension,
                        family,
                    )

            thousand_iterations()
            for repeat in range(repeats):
                elapsed = timed(thousand_iterations)
                rows.append(
                    {
                        "m_constraints": dimension,
                        "family": family,
                        "component": "1000_iterations",
                        "repeat": repeat,
                        "value": elapsed,
                        "unit": "seconds",
                        "batch_size": 8,
                        "iterations": 1000,
                    }
                )

            def ten_thousand_samples() -> object:
                return sample_transport(
                    fit,
                    10_000,
                    seed=dimension * 10_000 + max(rank, 0) + 3,
                    chunk_size=2000,
                )

            ten_thousand_samples()
            for repeat in range(repeats):
                elapsed = timed(ten_thousand_samples)
                rows.append(
                    {
                        "m_constraints": dimension,
                        "family": family,
                        "component": "10000_samples",
                        "repeat": repeat,
                        "value": elapsed,
                        "unit": "seconds",
                        "batch_size": 10_000,
                        "iterations": 1,
                    }
                )
            gc.collect()
            peak_rss = monitored_peak_rss(
                lambda: [objective_512() for _ in range(20)]
            )
            for repeat in range(repeats):
                # RSS is a process-level peak. Repeating the recorded,
                # stabilized measurement preserves the seven-row schema while
                # avoiding a misleading claim of independent memory samples.
                rows.append(
                    {
                        "m_constraints": dimension,
                        "family": family,
                        "component": "peak_memory",
                        "repeat": repeat,
                        "value": peak_rss,
                        "unit": "MiB_peak_RSS",
                        "batch_size": 512,
                        "iterations": 20,
                    }
                )
            print(
                f"benchmark md={dimension} family={family}",
                flush=True,
            )
    write_csv(METRICS / "microbenchmark_raw.csv", rows)
    summary_rows: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                int(row["m_constraints"]),
                str(row["family"]),
                str(row["component"]),
                str(row["unit"]),
                int(row["batch_size"]),
                int(row["iterations"]),
            )
            for row in rows
        }
    )
    for (
        dimension,
        family,
        component,
        unit,
        batch_size,
        iterations,
    ) in keys:
        values = [
            float(row["value"])
            for row in rows
            if int(row["m_constraints"]) == dimension
            and row["family"] == family
            and row["component"] == component
        ]
        summary_rows.append(
            {
                "m_constraints": dimension,
                "family": family,
                "component": component,
                "median": statistics.median(values),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "unit": unit,
                "batch_size": batch_size,
                "iterations": iterations,
                "repeats": len(values),
            }
        )
    write_csv(
        METRICS / "microbenchmark_summary.csv", summary_rows
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.log_file is not None:
        path = arguments.log_file.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = handle
        sys.stderr = handle
    benchmark(arguments.repeats)


if __name__ == "__main__":
    main()
