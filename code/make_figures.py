
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"
OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]
METHODS = [
    "Standard GP",
    "Projection GP",
    "Basis GP",
    "EP-GP",
    "CTVGP",
]
FAMILIES = ["diag", "rank2", "rank4", "full"]
BENCHMARK_FAMILIES = ["diag", "rank2", "rank4", "rank8", "full"]
MAPS = ["softplus", "exp", "squareplus"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)


def finite_errorbar(row: pd.Series, metric: str) -> tuple[float, float, float] | None:
    if row.get(f"{metric}_status") != "finite":
        return None
    mean = float(row[f"{metric}_mean"])
    low = float(row[f"{metric}_ci95_low"])
    high = float(row[f"{metric}_ci95_high"])
    return mean, mean - low, high - mean


def distance_by_method() -> None:
    frame = pd.read_csv(METRICS / "main_distance_summary.csv")
    cases = ["linear", "log", "sigmoid"]
    case_labels = ["(a) Linear", "(b) Log", "(c) Sigmoid"]
    metrics = [
        ("forward_kl", "Forward KL"),
        ("tv", "Total variation"),
        ("average_marginal_w1", "Avg. marginal $W_1$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    x = np.arange(len(cases), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(METHODS))
    for panel, (metric, ylabel) in enumerate(metrics):
        ax = axes[panel]
        for index, method in enumerate(METHODS):
            subset = (
                frame[frame["method"] == method]
                .set_index("case")
                .reindex(cases)
            )
            xs: list[float] = []
            ys: list[float] = []
            low_errors: list[float] = []
            high_errors: list[float] = []
            for case_index, (_, row) in enumerate(subset.iterrows()):
                estimate = finite_errorbar(row, metric)
                if estimate is None:
                    continue
                mean, low, high = estimate
                xs.append(x[case_index] + offsets[index])
                ys.append(mean)
                low_errors.append(low)
                high_errors.append(high)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=np.asarray([low_errors, high_errors]),
                    color=OKABE_ITO[index],
                    marker=MARKERS[index],
                    linestyle="none",
                    capsize=1.8,
                    label=method,
                    zorder=3,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + panel)})", loc="left", fontweight="bold")
        if metric in {"forward_kl", "average_marginal_w1"}:
            positive = [
                line.get_ydata()
                for line in ax.lines
                if len(line.get_ydata())
            ]
            if positive:
                ax.set_yscale("symlog", linthresh=1e-3)
                ax.set_ylim(bottom=0.0)
    # Basis GP is absent from the forward-KL panel because its value is
    # infinite, but it is finite and displayed in the TV/W1 panels.
    legend_by_label: dict[str, object] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend_by_label.update(zip(labels, handles))
    labels = [method for method in METHODS if method in legend_by_label]
    handles = [legend_by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.subplots_adjust(wspace=0.48, top=0.80, bottom=0.25)
    save_figure(fig, "distance_by_method")


def kl_vs_md_by_rank() -> None:
    frame = pd.read_csv(METRICS / "capacity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=False)
    for panel, case in enumerate(["log", "sigmoid"]):
        ax = axes[panel]
        subset = frame[frame["case"] == case]
        for index, family in enumerate(FAMILIES):
            rows = subset[subset["family"] == family].sort_values(
                "m_constraints"
            )
            rows = rows[
                rows["reverse_kl_status"].astype(str) == "finite"
            ]
            if rows.empty:
                continue
            x = rows["m_constraints"].to_numpy(float)
            mean = rows["reverse_kl_mean"].to_numpy(float)
            low = rows["reverse_kl_ci95_low"].to_numpy(float)
            high = rows["reverse_kl_ci95_high"].to_numpy(float)
            ax.plot(
                x,
                mean,
                color=OKABE_ITO[index],
                marker=MARKERS[index],
                linestyle=LINESTYLES[index],
                label=family,
            )
            ax.fill_between(
                x, low, high, color=OKABE_ITO[index], alpha=0.13
            )
        ax.set_xlabel("Derivative dimension $m_d$")
        ax.set_ylabel("Reverse KL")
        ax.set_title(
            f"({chr(97 + panel)}) {case.capitalize()}",
            loc="left",
            fontweight="bold",
        )
        ax.set_xticks([5, 10, 20])
        ax.set_yscale("log")
        for dimension in (5, 10, 20):
            dimension_rows = subset[
                subset["m_constraints"] == dimension
            ]
            if (
                not dimension_rows.empty
                and not (
                    dimension_rows["reverse_kl_status"].astype(str)
                    == "finite"
                ).any()
            ):
                ax.text(
                    dimension,
                    0.93,
                    "unavailable",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    rotation=90,
                    color="0.4",
                    fontsize=6.2,
                )
    axes[0].legend(ncol=2, loc="upper left")
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "kl_vs_md_by_rank")


def map_effect_md() -> None:
    frame = pd.read_csv(METRICS / "map_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
    for panel, case in enumerate(["log", "sigmoid"]):
        ax = axes[panel]
        subset = frame[frame["case"] == case]
        for index, map_name in enumerate(MAPS):
            rows = subset[subset["map"] == map_name].sort_values(
                "m_constraints"
            )
            rows = rows[
                rows["reverse_kl_status"].astype(str) == "finite"
            ]
            if rows.empty:
                continue
            x = rows["m_constraints"].to_numpy(float)
            mean = rows["reverse_kl_mean"].to_numpy(float)
            low = rows["reverse_kl_ci95_low"].to_numpy(float)
            high = rows["reverse_kl_ci95_high"].to_numpy(float)
            ax.plot(
                x,
                mean,
                color=OKABE_ITO[index],
                marker=MARKERS[index],
                linestyle=LINESTYLES[index],
                label=map_name,
            )
            ax.fill_between(
                x, low, high, color=OKABE_ITO[index], alpha=0.13
            )
        ax.set_xlabel("Derivative dimension $m_d$")
        ax.set_ylabel("Reverse KL")
        ax.set_xticks([5, 20])
        ax.set_yscale("log")
        ax.set_title(
            f"({chr(97 + panel)}) {case.capitalize()}",
            loc="left",
            fontweight="bold",
        )
        unavailable = subset[
            subset["m_constraints"] == 20
        ]["reverse_kl_status"].astype(str)
        if not unavailable.empty and not (unavailable == "finite").any():
            ax.text(
                20,
                0.52,
                "unavailable\n(0/5 targets)",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                color="0.35",
                fontsize=6.5,
            )
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, "map_effect_md")


def scaling_figure(metric: str, ylabel: str, name: str) -> None:
    frame = pd.read_csv(METRICS / "microbenchmark_summary.csv")
    frame = frame[frame["component"] == metric]
    fig, ax = plt.subplots(figsize=(5.5, 2.65))
    for index, family in enumerate(BENCHMARK_FAMILIES):
        rows = frame[frame["family"] == family].sort_values("m_constraints")
        if rows.empty:
            continue
        x = rows["m_constraints"].to_numpy(float)
        mean = rows["median"].to_numpy(float)
        q25 = rows["q25"].to_numpy(float)
        q75 = rows["q75"].to_numpy(float)
        valid = np.isfinite(mean) & (mean > 0)
        x, mean, q25, q75 = x[valid], mean[valid], q25[valid], q75[valid]
        if len(x) == 0:
            continue
        slope = (
            float(np.polyfit(np.log(x), np.log(mean), 1)[0])
            if len(x) >= 3
            else float("nan")
        )
        label = (
            f"{family} (slope {slope:.2f})"
            if np.isfinite(slope)
            else family
        )
        ax.plot(
            x,
            mean,
            color=OKABE_ITO[index],
            marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            label=label,
        )
        ax.fill_between(
            x, q25, q75, color=OKABE_ITO[index], alpha=0.13
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    registered_dimensions = sorted(
        frame["m_constraints"].dropna().astype(int).unique()
    )
    ax.set_xticks(registered_dimensions)
    ax.set_xticklabels([str(value) for value in registered_dimensions])
    ax.set_xlabel("Derivative dimension $m_d$")
    ax.set_ylabel(ylabel)
    if metric == "objective_gradient":
        theory = (
            r"Theory: $O(Bm_d^2+Bm_dr+m_dr^2+r^3)$"
        )
    else:
        theory = (
            "Model storage: low rank $O(m_dr)$; full $O(m_d^2)$\n"
            "Peak RSS includes fixed Python/BLAS overhead"
        )
    ax.text(
        0.98,
        0.04,
        theory,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 1.5,
        },
    )
    ax.legend(ncol=2, loc="best")
    ax.grid(True, which="both", alpha=0.20, linestyle="--")
    fig.tight_layout()
    save_figure(fig, name)


def optimization_stability() -> None:
    frame = pd.read_csv(METRICS / "main_restart_metrics.csv")
    frame["converged_bool"] = (
        frame["converged"].astype(str).str.lower() == "true"
    )
    grouped = (
        frame.groupby(["case", "data_seed"])
        .agg(
            elbo_range=(
                "validation_elbo",
                lambda values: float(np.max(values) - np.min(values)),
            ),
            max_gradient=("normalized_gradient", "max"),
            converged_restarts=("converged_bool", "sum"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    for index, case in enumerate(["linear", "log", "sigmoid"]):
        values = grouped[grouped["case"] == case]
        axes[0].scatter(
            np.full(len(values), index)
            + np.linspace(-0.08, 0.08, max(len(values), 1)),
            values["elbo_range"],
            color=OKABE_ITO[index],
            marker=MARKERS[index],
            s=14,
            alpha=0.8,
            label=case,
        )
        counts = (
            values["converged_restarts"]
            .value_counts()
            .reindex([0, 1, 2, 3], fill_value=0)
        )
        axes[1].plot(
            [0, 1, 2, 3],
            counts.to_numpy() / max(len(values), 1),
            color=OKABE_ITO[index],
            marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            label=case,
        )
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["linear", "log", "sigmoid"])
    axes[0].set_ylabel("Validation-ELBO restart range")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Restart dispersion", loc="left", fontweight="bold")
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_xlabel("Converged restarts per target")
    axes[1].set_ylabel("Fraction of data seeds")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("(b) Convergence", loc="left", fontweight="bold")
    axes[1].legend(loc="best")
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, "optimization_stability")


def main() -> None:
    configure_style()
    distance_by_method()
    kl_vs_md_by_rank()
    map_effect_md()
    optimization_stability()
    benchmark_path = METRICS / "microbenchmark_summary.csv"
    if benchmark_path.exists():
        scaling_figure(
            "objective_gradient",
            "Time per objective/gradient (s)",
            "runtime_scaling_md_rank",
        )
        scaling_figure(
            "peak_memory",
            "Peak RSS (MiB)",
            "memory_scaling_md_rank",
        )
    print("Figures generated in", FIGURES)


if __name__ == "__main__":
    main()
