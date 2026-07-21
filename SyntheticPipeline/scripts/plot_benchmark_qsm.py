"""Plot successful QSM benchmark rows to PNGs (non-interactive).

Always writes:
  1) F1 / distance overview (--out)
  2) Precision / Recall comparison (*_precision_recall.png)
  3) Algorithm x condition summary (*_by_condition.png)

Primary scores are high-res wood → low-poly cylinder abstraction fidelity.
Leaf-on / Leaf-off / Wood-only share one hue family with light→dark tints.
TreeQSM and SmartQSM are faceted so condition groups stay readable.
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
import textwrap

# Must be set before pyplot imports a GUI backend (common hang on Windows).
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

CONDITION_ORDER = ["leaf_on", "rgi", "oracle_wood", "gbseparation"]

CONDITION_LABELS = {
    "leaf_on": "Leaf-on",
    "rgi": "Leaf-off (RGI)",
    "oracle_wood": "Wood-only (oracle)",
    "gbseparation": "Leaf-off (GBSeparation)",
}

CONDITION_EXPLANATIONS = {
    "leaf_on": (
        "Leaf-on: no leaf separator. Points after normalize + CSF ground removal "
        "+ intensity filter; foliage kept."
    ),
    "rgi": (
        "Leaf-off (RGI): SegmentRGI wood/leaf separation with production "
        "EcomodelLite / pipeline_lite parameters (intensity + region growing)."
    ),
    "oracle_wood": (
        "Wood-only (oracle): QSM built from GT wood labels; still scored as a "
        "low-poly abstraction of the high-res trunk mesh / wood surface."
    ),
    "gbseparation": (
        "Leaf-off (GBSeparation): optional geometry-based separator "
        "(shortest-path wood extraction on a single Z-up tree)."
    ),
}

# Teal family: lighter = leafy/noisy input, darker = cleaner wood-only.
CONDITION_TINTS = {
    "leaf_on": "#A8D5C8",
    "rgi": "#4FA88F",
    "oracle_wood": "#1B5E4B",
    "gbseparation": "#7BB8A6",
}


def _sibling_out_path(out_path: str, suffix: str) -> str:
    root, ext = os.path.splitext(out_path)
    return f"{root}_{suffix}{ext or '.png'}"


def _y_limit(values, pad_ratio: float = 0.12, *, metric: str | None = None) -> float:
    """Scale bars to the data; 0-1 scores stay capped, distances do not."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    peak = float(np.nanmax(finite))
    if peak <= 0:
        return 0.1
    padded = peak * (1.0 + pad_ratio)
    is_distance = bool(metric and ("Dist" in metric or metric.endswith("_m")))
    if is_distance:
        return max(padded, peak + 0.05)
    # Keep some headroom for bar labels; never force a full 0-1 axis for tiny scores.
    if peak < 0.05:
        return max(padded, 0.05)
    if peak < 0.2:
        return max(padded, peak + 0.02)
    return min(1.05, max(padded, peak + 0.03))


def _ordered_conditions(values) -> list[str]:
    present = list(dict.fromkeys(values))
    known = [c for c in CONDITION_ORDER if c in present]
    extra = [c for c in present if c not in CONDITION_ORDER]
    return known + extra


def _condition_labels(order: list[str]) -> list[str]:
    return [CONDITION_LABELS.get(c, c) for c in order]


def _tint_palette(order: list[str]) -> dict[str, str]:
    palette = {}
    extras = [c for c in order if c not in CONDITION_TINTS]
    for condition in order:
        if condition in CONDITION_TINTS:
            palette[CONDITION_LABELS.get(condition, condition)] = CONDITION_TINTS[condition]
    if extras:
        # Fallback: generate nearby tints for unexpected conditions.
        for i, condition in enumerate(extras):
            lightness = 0.75 - 0.15 * i
            rgb = colorsys.hls_to_rgb(0.45, max(0.25, lightness), 0.45)
            palette[CONDITION_LABELS.get(condition, condition)] = matplotlib.colors.to_hex(rgb)
    return palette


def _add_input_legend(fig: plt.Figure, conditions: list[str], palette: dict[str, str]) -> None:
    handles = [
        Patch(
            facecolor=palette[CONDITION_LABELS.get(c, c)],
            edgecolor="0.3",
            label=CONDITION_LABELS.get(c, c),
        )
        for c in conditions
    ]
    legend = fig.legend(
        handles=handles,
        title="Input condition (light → dark = leafy → wood-only)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(4, len(handles)),
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )

    lines = ["How inputs were built:"]
    for condition in conditions:
        lines.append(f"• {CONDITION_EXPLANATIONS.get(condition, condition)}")
    lines.append(
        "Primary score: high-res wood model vs low-poly QSM cylinders "
        "(abstraction fidelity). Panels split QSM methods; within each panel, "
        "bars for one species are Leaf-on / Leaf-off / Wood-only side by side."
    )
    text = "\n".join(textwrap.fill(line, width=110) for line in lines)
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=8,
        linespacing=1.35,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.95,
        },
    )
    return legend


def _save_faceted_metrics(
    df: pd.DataFrame,
    metrics: list[str],
    titles: dict[str, str],
    out_path: str,
    *,
    condition_order: list[str],
) -> None:
    """One row of panels per metric; columns = QSM algorithm; hue = input tint."""
    label_order = _condition_labels(condition_order)
    palette = _tint_palette(condition_order)
    algorithms = sorted(df["Algorithm"].unique())
    species_order = sorted(df["Species"].unique())

    n_metrics = len(metrics)
    n_algos = max(1, len(algorithms))
    width = max(11.0, 1.1 * len(species_order) * n_algos)
    fig, axes = plt.subplots(
        n_metrics,
        n_algos,
        figsize=(width, 3.8 * n_metrics + 1.5),
        sharey="row",
        squeeze=False,
    )

    for row, metric in enumerate(metrics):
        for col, algorithm in enumerate(algorithms):
            ax = axes[row][col]
            subset = df[df["Algorithm"] == algorithm]
            sns.barplot(
                data=subset,
                x="Species",
                y=metric,
                hue="Input",
                order=species_order,
                hue_order=label_order,
                palette=palette,
                ax=ax,
                errorbar=None,
            )
            if row == 0:
                ax.set_title(algorithm, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(titles.get(metric, metric), fontsize=10)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_ylim(0, _y_limit(subset[metric], metric=metric))
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", padding=1, fontsize=6)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # Shared y-scale per metric row so TreeQSM/SmartQSM stay comparable.
        row_max = _y_limit(df[metric], metric=metric)
        for col in range(n_algos):
            axes[row][col].set_ylim(0, row_max)

    _add_input_legend(fig, condition_order, palette)
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved QSM benchmark plot to {out_path}")


def _save_condition_summary(
    df: pd.DataFrame,
    metrics: list[str],
    titles: dict[str, str],
    out_path: str,
    *,
    condition_order: list[str],
) -> None:
    """Compact overview: QSM method on x, tinted Leaf-on/Leaf-off/Wood-only grouped."""
    label_order = _condition_labels(condition_order)
    palette = _tint_palette(condition_order)
    summary = (
        df.groupby(["Algorithm", "Input", "Condition"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    algo_order = sorted(summary["Algorithm"].unique())

    nrows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.6 * nrows + 1.2))
    axes = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=summary,
            x="Algorithm",
            y=metric,
            hue="Input",
            order=algo_order,
            hue_order=label_order,
            palette=palette,
            ax=ax,
            errorbar=None,
        )
        ax.set_title(titles.get(metric, metric), fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Distance (m)" if "Dist" in metric or metric.endswith("_m") else "Score")
        ax.set_ylim(0, _y_limit(summary[metric], metric=metric))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    _add_input_legend(fig, condition_order, palette)
    fig.tight_layout(rect=(0, 0.18, 1, 0.92))
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved QSM benchmark plot to {out_path}")


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Plot QSM benchmark results")
    parser.add_argument(
        "--csv",
        default=os.path.join(
            root_dir, "SyntheticPipeline", "output", "qsm_benchmark", "benchmark_results_qsm.csv"
        ),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            root_dir, "SyntheticPipeline", "output", "visualizations", "benchmark_results_qsm.png"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)
    if "Status" in df.columns:
        df = df[df["Status"] == "ok"].copy()
    if "CylinderCount" in df.columns:
        df = df[df["CylinderCount"].fillna(0) > 0]

    overview_metrics = [
        m for m in ["Whole_F1", "Whole_IoU", "Whole_MedianDist_m", "Whole_P90Dist_m"]
        if m in df.columns
    ]
    if "Whole_F1" not in overview_metrics:
        print("CSV is missing Whole_F1.", file=sys.stderr)
        return 1
    # Fall back to secondary trunk/branch when present (older CSVs).
    for optional in ("Trunk_F1", "Branch_F1"):
        if optional in df.columns and df[optional].notna().any():
            overview_metrics.append(optional)

    pr_metrics = [
        m for m in [
            "Whole_Precision",
            "Whole_Recall",
            "Trunk_Precision",
            "Trunk_Recall",
            "Branch_Precision",
            "Branch_Recall",
        ]
        if m in df.columns and (m.startswith("Whole_") or df[m].notna().any())
    ]
    needed = [m for m in overview_metrics + pr_metrics if m.startswith("Whole_")]
    missing = [m for m in needed if m not in df.columns]
    if missing:
        print(f"CSV is missing metric columns: {missing}", file=sys.stderr)
        return 1

    df = df.dropna(subset=needed)
    if df.empty:
        print("No successful benchmark rows with metrics to plot.")
        return 1

    condition_order = _ordered_conditions(df["Condition"])
    df = df.copy()
    df["Input"] = df["Condition"].map(lambda c: CONDITION_LABELS.get(c, c))

    print(f"Plotting {len(df)} rows...")
    plot_cols = list(dict.fromkeys(overview_metrics + pr_metrics))
    species_df = (
        df.groupby(["Species", "Algorithm", "Input", "Condition"], as_index=False)[plot_cols]
        .mean(numeric_only=True)
    )

    titles = {
        "Whole_F1": "Abstraction F1 (hi-res wood ↔ cylinders)",
        "Whole_IoU": "Abstraction IoU",
        "Whole_MedianDist_m": "Median wood→model distance (m)",
        "Whole_P90Dist_m": "P90 wood→model distance (m)",
        "Trunk_F1": "Trunk F1 (secondary CylGT)",
        "Branch_F1": "Branch F1 (secondary CylGT)",
        "Whole_Precision": "Abstraction precision",
        "Whole_Recall": "Abstraction recall",
        "Trunk_Precision": "Trunk Precision (CylGT)",
        "Trunk_Recall": "Trunk Recall (CylGT)",
        "Branch_Precision": "Branch Precision (CylGT)",
        "Branch_Recall": "Branch Recall (CylGT)",
    }

    _save_faceted_metrics(
        species_df,
        overview_metrics,
        titles,
        args.out,
        condition_order=condition_order,
    )

    _save_faceted_metrics(
        species_df,
        pr_metrics,
        titles,
        _sibling_out_path(args.out, "precision_recall"),
        condition_order=condition_order,
    )

    summary_metrics = [
        m for m in overview_metrics + ["Whole_Precision", "Whole_Recall"]
        if m in species_df.columns
    ]
    _save_condition_summary(
        species_df,
        summary_metrics,
        {k: f"{v} (species-averaged)" for k, v in titles.items()},
        _sibling_out_path(args.out, "by_condition"),
        condition_order=condition_order,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
