"""Aggregate full experiments with paired seed bootstrap CIs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
TABLES = ROOT / "tables"
CASE_LABELS = {"linear": "(a)", "log": "(b)", "sigmoid": "(c)"}
METHOD_ORDER = [
    "Standard GP",
    "Projection GP",
    "Basis GP",
    "EP-GP",
    "CTVGP",
]
FAMILY_ORDER = ["diag", "rank2", "rank4", "full"]
MAP_ORDER = ["softplus", "exp", "squareplus"]
DISTANCE_METRICS = [
    "reverse_kl",
    "forward_kl",
    "tv",
    "average_marginal_w1",
    "sliced_w1",
]


def bootstrap_interval(
    values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    means = np.mean(values[indices], axis=1)
    return (
        float(np.mean(values)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def summarize_values(
    values: np.ndarray,
    indices: np.ndarray | None,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.any(np.isnan(values)):
        return {
            "mean": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "status": "unavailable",
        }
    if np.all(np.isposinf(values)):
        return {
            "mean": np.inf,
            "ci95_low": np.inf,
            "ci95_high": np.inf,
            "status": "infinite",
        }
    if not np.all(np.isfinite(values)):
        return {
            "mean": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "status": "mixed_nonfinite",
        }
    if indices is None:
        random = np.random.default_rng(20260728)
        indices = random.integers(
            0, len(values), size=(10_000, len(values))
        )
    mean, low, high = bootstrap_interval(values, indices)
    return {
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "status": "finite",
    }


def aggregate_main() -> pd.DataFrame:
    raw = pd.read_csv(METRICS / "main_distance_raw.csv")
    raw["metric_available_bool"] = (
        raw["metric_available"].astype(str).str.lower() == "true"
    )
    rows: list[dict[str, object]] = []
    for case, case_frame in raw.groupby("case", sort=False):
        complete_seeds: list[int] = []
        for seed, seed_frame in case_frame.groupby("data_seed"):
            if (
                len(seed_frame) == len(METHOD_ORDER)
                and bool(seed_frame["metric_available_bool"].all())
            ):
                complete_seeds.append(int(seed))
        complete_seeds = sorted(complete_seeds)
        claim_eligible = len(complete_seeds) >= 18
        random = np.random.default_rng(
            721_001 + list(CASE_LABELS).index(case)
        )
        paired_indices = (
            random.integers(
                0,
                len(complete_seeds),
                size=(10_000, len(complete_seeds)),
            )
            if complete_seeds
            else None
        )
        for method in METHOD_ORDER:
            method_frame = (
                case_frame[
                    (case_frame["method"] == method)
                    & case_frame["data_seed"].isin(complete_seeds)
                ]
                .sort_values("data_seed")
                .copy()
            )
            row: dict[str, object] = {
                "case": case,
                "case_label": CASE_LABELS[case],
                "method": method,
                "n_paired_seeds": len(complete_seeds),
                "planned_seeds": 20,
                "failure_rate": 1.0 - len(complete_seeds) / 20.0,
                "main_claim_eligible": claim_eligible,
            }
            for metric in DISTANCE_METRICS:
                if not claim_eligible:
                    summary = {
                        "mean": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "status": "unavailable_lt18_converged",
                    }
                else:
                    summary = summarize_values(
                        numeric_series(method_frame, metric),
                        paired_indices,
                    )
                for key, value in summary.items():
                    row[f"{metric}_{key}"] = value
            rows.append(row)
    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(
        METRICS / "main_distance_summary.csv", index=False
    )
    return summary_frame


def aggregate_grid(
    input_name: str,
    output_name: str,
    groups: list[str],
    planned_seeds: int,
) -> pd.DataFrame:
    raw = pd.read_csv(METRICS / input_name)
    raw["metric_available_bool"] = (
        raw["metric_available"].astype(str).str.lower() == "true"
    )
    metric_columns = [
        "reverse_kl",
        "normalized_reverse_kl",
        "forward_kl",
        "tv",
        "average_marginal_w1",
        "sliced_w1",
        "validation_elbo",
        "normalized_gradient",
        "wall_time_seconds",
        "jacobian_log_q01",
        "jacobian_log_q50",
        "jacobian_log_q99",
    ]
    metric_columns = [
        column for column in metric_columns if column in raw.columns
    ]
    rows: list[dict[str, object]] = []
    for group_values, frame in raw.groupby(groups, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        available = frame[frame["metric_available_bool"]].copy()
        seed_count = int(available["data_seed"].nunique())
        random = np.random.default_rng(
            913_337
            + sum(
                zlib_like(value)
                for value in group_values
            )
        )
        indices = (
            random.integers(
                0, len(available), size=(10_000, len(available))
            )
            if len(available)
            else None
        )
        row = {
            key: value for key, value in zip(groups, group_values)
        }
        row.update(
            {
                "available_seeds": seed_count,
                "planned_seeds": planned_seeds,
                "failure_rate": 1.0 - seed_count / planned_seeds,
            }
        )
        for metric in metric_columns:
            summary = summarize_values(
                numeric_series(available, metric), indices
            )
            for key, value in summary.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(METRICS / output_name, index=False)
    return result


def zlib_like(value: object) -> int:
    encoded = str(value).encode("utf-8")
    total = 0
    for byte in encoded:
        total = (total * 131 + byte) % 100_000
    return total


def format_estimate(
    mean: object,
    low: object,
    high: object,
    status: str,
) -> str:
    if status == "infinite":
        return "∞"
    if status != "finite":
        return "unavailable"
    mean_value = float(mean)
    low_value = float(low)
    high_value = float(high)
    maximum = max(abs(mean_value), abs(low_value), abs(high_value))
    if maximum < 0.01:
        pattern = "{:.4f}"
    elif maximum < 1.0:
        pattern = "{:.3f}"
    else:
        pattern = "{:.2f}"
    return (
        f"{pattern.format(mean_value)} "
        f"[{pattern.format(low_value)}, {pattern.format(high_value)}]"
    )


def write_main_tables(summary: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    header = (
        "| Case | Method | Reverse KL | Forward KL | TV | "
        "Avg. marginal W1 | Sliced W1 | n |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for _, row in summary.iterrows():
        cells = []
        for metric in DISTANCE_METRICS:
            cells.append(
                format_estimate(
                    row[f"{metric}_mean"],
                    row[f"{metric}_ci95_low"],
                    row[f"{metric}_ci95_high"],
                    str(row[f"{metric}_status"]),
                )
            )
        lines.append(
            "| {case} | {method} | {values} | {n}/20 |\n".format(
                case=row["case_label"],
                method=row["method"],
                values=" | ".join(cells),
                n=int(row["n_paired_seeds"]),
            )
        )
    notes = (
        "\nReverse KL is infinite for non-support-faithful Gaussian "
        "baselines. Basis GP is a derivative-space point mass, hence both "
        "KL directions are infinite and TV is one. Intervals are paired "
        "10,000-replicate bootstrap 95% CIs over common data seeds. "
        "`unavailable` is retained whenever fewer than 18/20 paired targets "
        "pass all reliability and VI convergence checks.\n"
    )
    (TABLES / "table5_direct_distance.md").write_text(
        "".join(lines) + notes, encoding="utf-8"
    )


def write_capacity_table(summary: pd.DataFrame) -> None:
    lines = [
        "| Case | $m_d$ | Family | Reverse KL/$m_d$ | TV | "
        "Avg. marginal W1 | Failure rate |\n",
        "|---|---:|---|---:|---:|---:|---:|\n",
    ]
    ordering = {name: index for index, name in enumerate(FAMILY_ORDER)}
    summary = summary.copy()
    summary["_family_order"] = summary["family"].map(ordering)
    summary = summary.sort_values(
        ["case", "m_constraints", "_family_order"]
    )
    for _, row in summary.iterrows():
        cells = []
        for metric in [
            "normalized_reverse_kl",
            "tv",
            "average_marginal_w1",
        ]:
            cells.append(
                format_estimate(
                    row[f"{metric}_mean"],
                    row[f"{metric}_ci95_low"],
                    row[f"{metric}_ci95_high"],
                    str(row[f"{metric}_status"]),
                )
            )
        lines.append(
            f"| {row['case']} | {int(row['m_constraints'])} | "
            f"{row['family']} | {' | '.join(cells)} | "
            f"{float(row['failure_rate']):.0%} |\n"
        )
    (TABLES / "table_capacity.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def write_map_table(summary: pd.DataFrame) -> None:
    lines = [
        "| Case | $m_d$ | Map | Reverse KL | TV | Avg. marginal W1 | "
        "Jacobian log-tail (1%) | Failure rate |\n",
        "|---|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    ordering = {name: index for index, name in enumerate(MAP_ORDER)}
    summary = summary.copy()
    summary["_map_order"] = summary["map"].map(ordering)
    summary = summary.sort_values(
        ["case", "m_constraints", "_map_order"]
    )
    for _, row in summary.iterrows():
        cells = []
        for metric in [
            "reverse_kl",
            "tv",
            "average_marginal_w1",
            "jacobian_log_q01",
        ]:
            cells.append(
                format_estimate(
                    row[f"{metric}_mean"],
                    row[f"{metric}_ci95_low"],
                    row[f"{metric}_ci95_high"],
                    str(row[f"{metric}_status"]),
                )
            )
        lines.append(
            f"| {row['case']} | {int(row['m_constraints'])} | "
            f"{row['map']} | {' | '.join(cells)} | "
            f"{float(row['failure_rate']):.0%} |\n"
        )
    (TABLES / "table_map_effect.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Aggregate whatever experiment files currently exist.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    TABLES.mkdir(parents=True, exist_ok=True)
    main_path = METRICS / "main_distance_raw.csv"
    capacity_path = METRICS / "capacity_raw.csv"
    map_path = METRICS / "map_raw.csv"
    if main_path.exists():
        main_summary = aggregate_main()
        write_main_tables(main_summary)
    elif not arguments.allow_partial:
        raise FileNotFoundError(main_path)
    if capacity_path.exists():
        capacity_summary = aggregate_grid(
            "capacity_raw.csv",
            "capacity_summary.csv",
            ["case", "m_constraints", "family"],
            5,
        )
        write_capacity_table(capacity_summary)
    elif not arguments.allow_partial:
        raise FileNotFoundError(capacity_path)
    if map_path.exists():
        map_summary = aggregate_grid(
            "map_raw.csv",
            "map_summary.csv",
            ["case", "m_constraints", "map"],
            5,
        )
        write_map_table(map_summary)
    elif not arguments.allow_partial:
        raise FileNotFoundError(map_path)
    print("Aggregation complete.")


if __name__ == "__main__":
    main()
