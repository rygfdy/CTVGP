

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
TABLES = ROOT / "tables"
CASE_LABEL = {"linear": "(a)", "log": "(b)", "sigmoid": "(c)"}


def boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def interval(row: pd.Series, metric: str) -> str:
    status = str(row.get(f"{metric}_status", "unavailable"))
    if status == "infinite":
        return "∞"
    if status != "finite":
        return "unavailable"
    mean = float(row[f"{metric}_mean"])
    low = float(row[f"{metric}_ci95_low"])
    high = float(row[f"{metric}_ci95_high"])
    precision = 4 if max(abs(mean), abs(low), abs(high)) < 0.01 else 3
    return (
        f"{mean:.{precision}f} "
        f"[{low:.{precision}f}, {high:.{precision}f}]"
    )


def write_complexity_table() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    text = """| Component | Diagonal | Diag + rank \(r\) | Full |
|---|---:|---:|---:|
| Variational parameters | \(O(m_d)\) | \(O(m_d r)\) | \(O(m_d^2)\) |
| Auxiliary covariance storage | \(O(m_d)\) | \(O(m_d r)\) | \(O(m_d^2)\) |
| Per-batch target score | \(O(Bm_d^2)\) | \(O(Bm_d^2)\) | \(O(Bm_d^2)\) |
| Auxiliary entropy/inverse | \(O(m_d)\) | \(O(m_dr^2+r^3)\) | \(O(m_d^2)\) |
| Reparameterized sampling | \(O(Sm_d)\) | \(O(Sm_dr)\) | \(O(Sm_d^2)\) |
| Total low-rank iteration | \(O(Bm_d^2)\) | \(O(Bm_d^2+Bm_dr+m_dr^2+r^3)\) | \(O(Bm_d^2)\) |

Exact-GP preprocessing, shared by all families, is
\(O(n^3+n^2m_d+nm_d^2+m_d^3)\). The full-family iteration has the
same leading dense-target order as low rank but a larger parameter and
sampling constant; the measured microbenchmark separates these components.
"""
    (TABLES / "table_complexity.md").write_text(
        text, encoding="utf-8"
    )


def write_seed_analysis() -> None:
    restarts = pd.read_csv(METRICS / "main_restart_metrics.csv")
    restarts["converged_bool"] = boolean(restarts["converged"])
    target = (
        restarts.groupby(["case", "data_seed"])
        .agg(
            converged_restarts=("converged_bool", "sum"),
            elbo_min=("validation_elbo", "min"),
            elbo_max=("validation_elbo", "max"),
            gradient_min=("normalized_gradient", "min"),
        )
        .reset_index()
    )
    target["elbo_range"] = target["elbo_max"] - target["elbo_min"]
    lines = [
        "# Seed Analysis\n\n",
        "## Answer to the seed question\n\n",
        "Multiple data seeds are necessary even though CTVGP is not a neural "
        "network. A data seed changes the realized observation noise, fitted "
        "GP hyperparameters, posterior derivative mean/covariance, hard "
        "normalizer, and therefore the target distribution itself. These 20 "
        "paired data seeds are the independent units used by the bootstrap "
        "confidence interval.\n\n",
        "A VI restart measures a different source of variation: the stochastic "
        "AMSGrad path, randomized-QMC stream, and the local numerical optimum "
        "within one fixed posterior target. It is not an additional data "
        "replicate. We run three restarts, choose the report-facing fit by an "
        "independent validation ELBO among converged fits, and report restart "
        "dispersion separately.\n\n",
        "For a fixed dataset, Standard GP, Projection GP, and Basis GP are "
        "deterministic. Gaussian-site EP is deterministic given its site "
        "softness and update rule. Assigning artificial optimizer seeds to "
        "these baselines would manufacture uncertainty and is therefore not "
        "done. Normalizer, hard-reference, and distance Monte Carlo streams "
        "are independent of both data and VI streams; their MC SE is stored "
        "separately rather than folded into the data-seed interval.\n\n",
        "## Observed stability\n\n",
        "| Case | Targets with ≥1 converged fit | Converged restarts | "
        "Median ELBO restart range | Median best normalized gradient |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for case in ["linear", "log", "sigmoid"]:
        case_target = target[target["case"] == case]
        case_restart = restarts[restarts["case"] == case]
        lines.append(
            f"| {CASE_LABEL[case]} {case} | "
            f"{int((case_target['converged_restarts'] > 0).sum())}/20 | "
            f"{int(case_restart['converged_bool'].sum())}/60 | "
            f"{float(case_target['elbo_range'].median()):.4g} | "
            f"{float(case_target['gradient_min'].median()):.3g} |\n"
        )
    lines.extend(
        [
            "\n## Interval construction\n\n",
            "Within each case, only data seeds for which all five method rows "
            "and the selected CTVGP fit are reliable form the common paired "
            "matrix. The same 10,000 bootstrap index arrays are applied to "
            "every method in that case. A main claim is emitted only when at "
            "least 18/20 common targets remain. Failed restarts and failed "
            "targets stay in `failure_list.csv`; they are never replaced by a "
            "different seed or a hard-reference fit.\n",
        ]
    )
    (ROOT / "seed_analysis.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def write_results() -> None:
    main = pd.read_csv(METRICS / "main_distance_summary.csv")
    capacity = pd.read_csv(METRICS / "capacity_summary.csv")
    map_frame = pd.read_csv(METRICS / "map_summary.csv")
    restart = pd.read_csv(METRICS / "main_restart_metrics.csv")
    restart["converged_bool"] = boolean(restart["converged"])
    lines = [
        "# CTVGP Rebuttal v2 Results\n\n",
        "## Compact outline\n\n",
        "- Directly compare each paper baseline and rank-2 softplus CTVGP with "
        "the same hard derivative posterior.\n",
        "- Separate data-seed uncertainty from VI-restart and Monte Carlo "
        "uncertainty.\n",
        "- Measure the effects of \(m_d\), auxiliary rank, and positive map.\n",
        "- Validate the Woodbury complexity change with a component-level "
        "microbenchmark.\n",
        "- Preserve every convergence, normalizer, reference, and nesting "
        "failure explicitly.\n\n",
        "## 1. Direct derivative-posterior distances\n\n",
        "The full interval table is in "
        "[`tables/table5_direct_distance.md`](tables/table5_direct_distance.md). "
        "Reverse KL is support-sensitive: it is infinite for Standard GP, "
        "Projection GP, and Gaussian EP, and both KL directions are infinite "
        "for deterministic Basis GP. CTVGP is the only paper method with "
        "finite reverse KL by construction.\n\n",
    ]
    for case in ["linear", "log", "sigmoid"]:
        case_frame = main[main["case"] == case]
        ctvgp = case_frame[case_frame["method"] == "CTVGP"].iloc[0]
        standard = case_frame[
            case_frame["method"] == "Standard GP"
        ].iloc[0]
        ep = case_frame[case_frame["method"] == "EP-GP"].iloc[0]
        lines.append(
            f"- **{CASE_LABEL[case]} {case}.** CTVGP reverse KL "
            f"{interval(ctvgp, 'reverse_kl')}, TV "
            f"{interval(ctvgp, 'tv')}, and average marginal W1 "
            f"{interval(ctvgp, 'average_marginal_w1')}. Standard GP has "
            f"TV {interval(standard, 'tv')}; EP-GP has forward KL "
            f"{interval(ep, 'forward_kl')} but infinite reverse KL. "
            f"The paired table uses {int(ctvgp['n_paired_seeds'])}/20 "
            "targets.\n"
        )
    lines.extend(
        [
            "\nThese metrics answer different questions. A Gaussian baseline "
            "can be close in forward KL or TV when the original GP already "
            "places little mass outside the orthant, while still failing the "
            "hard-support requirement and having infinite reverse KL. The "
            "support theorem should therefore be read together with TV and "
            "Wasserstein, not as a claim that every baseline distance is large.\n\n",
            "## 2. Capacity as \(m_d\) and rank change\n\n",
            "The full table is in "
            "[`tables/table_capacity.md`](tables/table_capacity.md). "
            "The rank-family comparison is made on common log and sigmoid "
            "targets, with one nested warm start and two independent starts. "
            "Observed rank-nesting violations are labeled optimization "
            "failures rather than structural counterexamples.\n\n",
        ]
    )
    for case in ["log", "sigmoid"]:
        subset = capacity[
            (capacity["case"] == case)
            & (capacity["reverse_kl_status"] == "finite")
        ]
        if subset.empty:
            lines.append(
                f"- **{case}:** no reliable capacity aggregate was available.\n"
            )
            continue
        for dimension in [5, 10, 20]:
            rows = subset[subset["m_constraints"] == dimension]
            if rows.empty:
                continue
            best = rows.loc[rows["reverse_kl_mean"].idxmin()]
            lines.append(
                f"- **{case}, \(m_d={dimension}\):** the smallest observed "
                f"reverse KL is {float(best['reverse_kl_mean']):.3g} for "
                f"{best['family']} (failure rate "
                f"{float(best['failure_rate']):.0%}).\n"
            )
    lines.extend(
        [
            "\nThe theory does not claim that fixed rank must worsen for every "
            "target sequence. It proves rank nesting for a fixed target and "
            "gives an \u03a9(\(m_d-r\)) lower bound only under explicit spectral "
            "and covariance-residual growth assumptions.\n\n",
            "## 3. Positive-map effect\n\n",
            "The map table is in "
            "[`tables/table_map_effect.md`](tables/table_map_effect.md). "
            "Softplus remains the main paper-protocol map. The map study uses "
            "rank 2 and fixes squareplus \u03b1/\(s_d\)=0.25.\n\n",
        ]
    )
    for case in ["log", "sigmoid"]:
        for dimension in [5, 20]:
            rows = map_frame[
                (map_frame["case"] == case)
                & (map_frame["m_constraints"] == dimension)
                & (map_frame["reverse_kl_status"] == "finite")
            ]
            if rows.empty:
                continue
            best = rows.loc[rows["reverse_kl_mean"].idxmin()]
            lines.append(
                f"- **{case}, \(m_d={dimension}\):** {best['map']} has the "
                f"lowest mean reverse KL ({float(best['reverse_kl_mean']):.3g}); "
                f"its mean TV is {float(best['tv_mean']):.3g}.\n"
            )
    lines.extend(
        [
            "\nSoftplus/exp induce exponential exact auxiliary left tails, "
            "whereas squareplus induces a polynomial left tail. This changes "
            "boundary geometry and Jacobian gradients but does not imply a "
            "universal map ordering.\n\n",
            "## 4. Complexity and measured scaling\n\n",
            "The proof and limitations are in "
            "[`complexity_and_fidelity_theory.md`](complexity_and_fidelity_theory.md), "
            "and the compact order table is in "
            "[`tables/table_complexity.md`](tables/table_complexity.md). "
            "After Woodbury, a low-rank batch iteration costs "
            "\(O(Bm_d^2+Bm_dr+m_dr^2+r^3)\). The dense target score retains "
            "the \(Bm_d^2\) term, so low rank does not remove all quadratic "
            "work. The microbenchmark records median/IQR timing and peak RSS "
            "for every registered dimension/rank configuration.\n\n",
            "## 5. Failures and scope\n\n",
        ]
    )
    target_success = (
        restart.groupby(["case", "data_seed"])["converged_bool"]
        .any()
        .groupby("case")
        .sum()
    )
    for case, count in target_success.items():
        lines.append(
            f"- {case}: {int(count)}/20 targets have at least one formally "
            "converged restart.\n"
        )
    lines.extend(
        [
            "- Every restart failure, unreliable normalizer/reference, and "
            "observed nesting failure is listed in `failure_list.csv`.\n",
            "- An individual unavailable metric is printed as `unavailable`; "
            "it is never clipped, imputed, or replaced.\n",
            "- The main table is generated only if every case reaches the "
            "pre-registered 18/20 target threshold.\n\n",
            "## Claim–evidence map\n\n",
            "| Claim | Evidence | Status |\n",
            "|---|---|---|\n",
            "| CTVGP is support faithful | Positive-map construction and 1M-coordinate test | Supported |\n",
            "| Gaussian baselines have finite reverse KL | Support theorem | Rejected; reverse KL is infinite |\n",
            "| Main distances are stable across observations | 20 paired data seeds and bootstrap CIs | Supported if 18/20 threshold passes |\n",
            "| Larger rank must empirically win every run | Population nesting versus optimizer audit | Rejected; failures are labeled |\n",
            "| Fixed rank universally worsens with dimension | Counterexample boundary and conditional theorem | Rejected; conditional claim only |\n",
            "| Woodbury removes all \(m_d^2\) work | Dense target-score decomposition | Rejected |\n\n",
            "## Adversarial self-review\n\n",
            "- **Contribution:** direct support-aware fidelity is measured rather "
            "than inferred from predictive moments.\n",
            "- **Clarity:** support, seed units, map parameters, and failure "
            "rules are explicit.\n",
            "- **Experimental strength:** full results include all planned "
            "targets/restarts and do not suppress negative outcomes.\n",
            "- **Evaluation completeness:** five paper methods, four covariance "
            "families, three maps, and component timing are present.\n",
            "- **Method soundness:** claims are limited by support direction, "
            "optimizer convergence, target curvature, and conditional "
            "covariance assumptions.\n",
        ]
    )
    (ROOT / "rebuttal_v2_results.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def write_rebuttal_snippet() -> None:
    main = pd.read_csv(METRICS / "main_distance_summary.csv")
    snippets = []
    for case in ["linear", "log", "sigmoid"]:
        row = main[
            (main["case"] == case) & (main["method"] == "CTVGP")
        ].iloc[0]
        snippets.append(
            f"{CASE_LABEL[case]} reverse KL {interval(row, 'reverse_kl')}, "
            f"TV {interval(row, 'tv')}, and marginal W1 "
            f"{interval(row, 'average_marginal_w1')}"
        )
    text = r"""### Rebuttal-ready English text

**Direct fidelity and uncertainty.** We added a direct comparison between the
variational derivative marginal \(q_\theta(d)\) and the common hard target
\(p_h(d)=N(m_d,\Sigma_{dd})\mid d\ge0\), using the three Table-5 functions,
the same five method labels, 20 paired data realizations, and three genuine VI
restarts per target. The reported 95% intervals use 10,000 paired bootstrap
replicates over data seeds; VI-restart dispersion and Monte Carlo SE are kept
separate. For CTVGP (rank-2 softplus), DIRECT_SUMMARY. Standard GP,
Projection GP, and Gaussian-site EP have infinite
reverse KL because they place positive probability outside the hard orthant.
The deterministic Basis GP is mutually singular with the continuous hard
target (both KL directions are infinite and TV is one). We therefore also
report forward KL, TV, marginal \(W_1\), and sliced \(W_1\), which remain
informative when reverse KL is support-infinite.

**Why multiple seeds are required.** Although CTVGP is not a neural network,
the data seed changes the observed noise, fitted GP hyperparameters, and the
entire derivative posterior target. It is therefore the independent unit for
uncertainty. A VI restart instead measures optimization/QMC variability for one
fixed target. Fixed-data Standard/Projection/Basis GP and deterministic
Gaussian-site EP are not assigned artificial optimizer seeds.

**Rank, dimension, map, and complexity.** We additionally varied
\(m_d\in\{5,10,20\}\), auxiliary covariance family
\(\{{\rm diag},r=2,r=4,{\rm full}\}\), and positive map
\(\{{\rm softplus},\exp,{\rm squareplus}\}\). Rank families are nested at the
population optimum; an observed lower ELBO for a larger family is thus labeled
optimization failure. Dimension alone does *not* imply that fixed-rank KL must
increase. We prove a conditional Gaussian-surrogate lower bound
\(\mathrm{KL}\ge c\delta_{m,r}^2\), which becomes
\(\Omega(m_d-r)\) only when the best diagonal-plus-rank residual satisfies
\(\delta_{m,r}^2=\Omega(m_d-r)\) under bounded spectra. Finally, v2 replaces
the previous dense low-rank inverse with Woodbury identities. The per-batch
cost is
\(O(Bm_d^2+Bm_dr+m_dr^2+r^3)\); the dense target score retains
\(Bm_d^2\), so we do not claim that low rank removes all quadratic work.
"""
    text = text.replace(
        "DIRECT_SUMMARY",
        f"{snippets[0]}; {snippets[1]}; and {snippets[2]}",
    )
    (ROOT / "rebuttal_snippet.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    write_complexity_table()
    write_seed_analysis()
    write_results()
    write_rebuttal_snippet()
    print("Reports generated.")


if __name__ == "__main__":
    main()
