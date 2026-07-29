"""Strict completeness, isolation, traceability, and inventory validation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"
EXPECTED = {
    "main_restart_metrics.csv": 180,
    "main_distance_raw.csv": 300,
    "capacity_restart_metrics.csv": 360,
    "capacity_raw.csv": 120,
    "map_restart_metrics.csv": 180,
    "map_raw.csv": 60,
}
EXPECTED_FIGURES = [
    "distance_by_method",
    "kl_vs_md_by_rank",
    "map_effect_md",
    "runtime_scaling_md_rank",
    "memory_scaling_md_rank",
    "optimization_stability",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def validate() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: Any) -> None:
        checks.append(
            {"check": name, "passed": bool(passed), "details": details}
        )

    for filename, expected_rows in EXPECTED.items():
        path = METRICS / filename
        exists = path.exists()
        row_count = len(pd.read_csv(path)) if exists else 0
        record(
            f"row_count:{filename}",
            exists and row_count == expected_rows,
            {"expected": expected_rows, "observed": row_count},
        )

    restart_files = [
        METRICS / "main_restart_metrics.csv",
        METRICS / "capacity_restart_metrics.csv",
        METRICS / "map_restart_metrics.csv",
    ]
    manifest_frames: list[pd.DataFrame] = []
    for path in restart_files:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        manifest_frames.append(frame)
        converged = bool_series(frame["converged"])
        for _, row in frame.loc[~converged].iterrows():
            failures.append(
                {
                    "failure_type": "vi_restart",
                    "experiment": row.get("experiment", ""),
                    "case": row.get("case", ""),
                    "data_seed": row.get("data_seed", ""),
                    "m_constraints": row.get("m_constraints", ""),
                    "family": row.get("family", ""),
                    "map": row.get("map", ""),
                    "restart": row.get("restart", ""),
                    "reason": row.get("failure_reason", ""),
                }
            )
    if manifest_frames:
        manifest = pd.concat(manifest_frames, ignore_index=True)
        manifest.insert(
            0,
            "run_id",
            [
                f"{row.experiment}:{row.case}:data{row.data_seed}:"
                f"md{row.m_constraints}:{getattr(row, 'family', '')}:"
                f"{getattr(row, 'map', '')}:restart{row.restart}"
                for row in manifest.itertuples()
            ],
        )
        manifest.to_csv(ROOT / "run_manifest.csv", index=False)

    main_restarts = (
        pd.read_csv(METRICS / "main_restart_metrics.csv")
        if (METRICS / "main_restart_metrics.csv").exists()
        else pd.DataFrame()
    )
    if not main_restarts.empty:
        matrix = main_restarts.groupby(["case", "data_seed"]).size()
        record(
            "main_20x3_matrix",
            len(matrix) == 60 and bool((matrix == 3).all()),
            {
                "targets": int(len(matrix)),
                "minimum_restarts": int(matrix.min()),
                "maximum_restarts": int(matrix.max()),
            },
        )
        target_success = (
            main_restarts.assign(
                converged_bool=bool_series(main_restarts["converged"])
            )
            .groupby(["case", "data_seed"])["converged_bool"]
            .any()
            .groupby("case")
            .sum()
        )
        record(
            "main_claim_threshold_each_case",
            bool((target_success >= 18).all()),
            target_success.to_dict(),
        )

    for filename in [
        "main_distance_raw.csv",
        "capacity_raw.csv",
        "map_raw.csv",
    ]:
        path = METRICS / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        unavailable = ~bool_series(frame["metric_available"])
        explicit = frame.loc[unavailable, "unavailable_reason"].notna().all()
        record(
            f"explicit_unavailable:{filename}",
            bool(explicit),
            {"unavailable_rows": int(unavailable.sum())},
        )
        unreliable = (
            ~bool_series(frame["normalizer_reliable"])
            | ~bool_series(frame["reference_reliable"])
        )
        if unreliable.any():
            for _, row in frame.loc[unreliable].iterrows():
                failures.append(
                    {
                        "failure_type": "normalizer_or_reference",
                        "experiment": row.get("experiment", ""),
                        "case": row.get("case", ""),
                        "data_seed": row.get("data_seed", ""),
                        "m_constraints": row.get("m_constraints", ""),
                        "family": row.get("family", ""),
                        "map": row.get("map", ""),
                        "restart": "",
                        "reason": row.get("unavailable_reason", ""),
                    }
                )

    nesting_path = METRICS / "capacity_nesting_checks.csv"
    if nesting_path.exists():
        nesting = pd.read_csv(nesting_path)
        passed = bool_series(nesting["nesting_check_passed"])
        for _, row in nesting.loc[~passed].iterrows():
            failures.append(
                {
                    "failure_type": "rank_nesting_observed_failure",
                    "experiment": "capacity",
                    "case": row["case"],
                    "data_seed": row["data_seed"],
                    "m_constraints": row["m_constraints"],
                    "family": (
                        f"{row['smaller_family']}->{row['larger_family']}"
                    ),
                    "map": "",
                    "restart": "",
                    "reason": row["interpretation"],
                }
            )
        record(
            "nesting_failures_are_explicit",
            True,
            {
                "checks": len(nesting),
                "failed": int((~passed).sum()),
            },
        )

    figure_status = {}
    for name in EXPECTED_FIGURES:
        pdf = FIGURES / f"{name}.pdf"
        png = FIGURES / f"{name}.png"
        figure_status[name] = {
            "pdf": pdf.exists() and pdf.stat().st_size > 1000,
            "png": png.exists() and png.stat().st_size > 1000,
        }
    record(
        "all_figure_pairs",
        all(
            status["pdf"] and status["png"]
            for status in figure_status.values()
        ),
        figure_status,
    )

    forbidden_tokens = [
        "rebuttal_artifacts",
        "ctvgp_v3_experiments",
        "submitted_module",
    ]
    source_violations: list[str] = []
    for path in (ROOT / "code").glob("*.py"):
        # The validator necessarily names the forbidden legacy tokens that it
        # searches for; exclude only this audit script from its own scan.
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                source_violations.append(f"{path.name}:{token}")
    record(
        "source_isolation_no_legacy_import",
        not source_violations,
        source_violations,
    )

    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    record(
        "mandatory_tests",
        test_result.returncode == 0,
        (test_result.stdout + test_result.stderr).strip(),
    )

    theory = (
        (ROOT / "complexity_and_fidelity_theory.md").read_text(
            encoding="utf-8"
        )
        if (ROOT / "complexity_and_fidelity_theory.md").exists()
        else ""
    )
    conditional_language = all(
        phrase in theory
        for phrase in [
            "conditional Gaussian-surrogate",
            "not a universal theorem",
            "dimension alone",
        ]
    )
    record(
        "theory_scope_not_overclaimed",
        conditional_language,
        {
            "required_phrases_present": conditional_language,
        },
    )

    failures_path = ROOT / "failure_list.csv"
    fields = [
        "failure_type",
        "experiment",
        "case",
        "data_seed",
        "m_constraints",
        "family",
        "map",
        "restart",
        "reason",
    ]
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)

    report = {
        "scope_root": str(ROOT.resolve()),
        "checks": checks,
        "all_checks_passed": all(check["passed"] for check in checks),
        "explicit_failure_rows": len(failures),
    }
    (ROOT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report, failures


def inventory() -> None:
    output = ROOT / "artifact_inventory.csv"
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != output
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "bytes",
                "sha256",
            ],
        )
        writer.writeheader()
        for path in paths:
            writer.writerow(
                {
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )


def main() -> None:
    report, _ = validate()
    inventory()
    print(
        json.dumps(
            {
                "all_checks_passed": report["all_checks_passed"],
                "checks": len(report["checks"]),
                "explicit_failure_rows": report[
                    "explicit_failure_rows"
                ],
            },
            indent=2,
        )
    )
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
