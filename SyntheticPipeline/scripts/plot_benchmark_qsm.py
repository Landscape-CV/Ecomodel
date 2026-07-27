"""Plot successful QSM benchmark rows to PNGs (non-interactive).

Always writes:
  1) F1 / distance overview (--out)
  2) Precision / Recall comparison (*_precision_recall.png)
  3) Algorithm x condition summary (*_by_condition.png)

Primary scores are high-res wood → low-poly cylinder abstraction fidelity.
Leaf-on / Leaf-off / Wood-only share one hue family with light→dark tints.
Each QSM algorithm (TreeQSM, SmartQSM, AdTree, aRchi, …) is faceted so
condition groups stay readable.
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

# title, direction label, one-line definition for subplot annotations
METRIC_INFO = {
    "Whole_F1": (
        "Abstraction F1 (hi-res wood ↔ cylinders)",
        "higher is better",
        "Harmonic mean of precision and recall vs the wood surface.",
    ),
    "Whole_IoU": (
        "Abstraction IoU",
        "higher is better",
        "F1 rewritten as an IoU-like overlap score.",
    ),
    "Whole_MedianDist_m": (
        "Median wood→model distance (m)",
        "lower is better",
        "Median distance from coarse wood points to nearest cylinder surface.",
    ),
    "Whole_P90Dist_m": (
        "P90 wood→model distance (m)",
        "lower is better",
        "90th-percentile wood→cylinder distance (tail / missed branches).",
    ),
    "Whole_Precision": (
        "Abstraction precision",
        "higher is better",
        "Fraction of QSM surface within tolerance of dense high-res wood "
        "(does not invent geometry).",
    ),
    "Whole_Recall": (
        "Abstraction recall",
        "higher is better",
        "Fraction of coarse wood support covered by cylinders "
        "(completeness of the low-poly fit).",
    ),
    "Trunk_F1": (
        "Trunk F1 (secondary CylGT)",
        "higher is better",
        "Secondary score vs weak mesh-OBB cylinder GT (large-radius parts).",
    ),
    "Branch_F1": (
        "Branch F1 (secondary CylGT)",
        "higher is better",
        "Secondary score vs weak mesh-OBB cylinder GT (small-radius parts).",
    ),
    "Trunk_Precision": (
        "Trunk Precision (CylGT)",
        "higher is better",
        "Secondary trunk precision vs mesh-OBB cylinder GT.",
    ),
    "Trunk_Recall": (
        "Trunk Recall (CylGT)",
        "higher is better",
        "Secondary trunk recall vs mesh-OBB cylinder GT.",
    ),
    "Branch_Precision": (
        "Branch Precision (CylGT)",
        "higher is better",
        "Secondary branch precision vs mesh-OBB cylinder GT.",
    ),
    "Branch_Recall": (
        "Branch Recall (CylGT)",
        "higher is better",
        "Secondary branch recall vs mesh-OBB cylinder GT.",
    ),
}

BENCHMARK_METHOD_LINES = [
    "How the benchmark is run:",
    "1) Preprocess each tile: normalize → CSF ground removal → intensity filter "
    "(mirrors pipeline_lite).",
    "2) Build three QSM inputs from that cloud: Leaf-on (foliage kept), "
    "Leaf-off/RGI (SegmentRGI wood), Wood-only/oracle (GT wood labels).",
    "3) Run each QSM backend on the chosen input; translate cylinders back to "
    "world coords.",
    "4) Score as high-res wood → low-poly cylinder abstraction: precision vs "
    "dense *_trunk.ply; recall + distances vs an 8 cm–coarsened wood target "
    "(τ = distance tolerance, default 5 cm).",
    "5) Bars are species-averaged means of successful (Status=ok) runs; "
    "CylGT trunk/branch panels are secondary diagnostics only.",
]


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
        for i, condition in enumerate(extras):
            lightness = 0.75 - 0.15 * i
            rgb = colorsys.hls_to_rgb(0.45, max(0.25, lightness), 0.45)
            palette[CONDITION_LABELS.get(condition, condition)] = matplotlib.colors.to_hex(rgb)
    return palette


def _metric_title(metric: str, *, species_averaged: bool = False) -> str:
    title, direction, _ = METRIC_INFO.get(metric, (metric, "", ""))
    if species_averaged:
        title = f"{title} (species-averaged)"
    if direction:
        return f"{title}\n({direction})"
    return title


def _metric_ylabel(metric: str) -> str:
    _, direction, _ = METRIC_INFO.get(metric, (metric, "", ""))
    if "Dist" in metric or metric.endswith("_m"):
        base = "Distance (m)"
    else:
        base = "Score"
    return f"{base}\n[{direction}]" if direction else base


def _add_figure_guides(
    fig: plt.Figure,
    conditions: list[str],
    palette: dict[str, str],
    metrics: list[str],
) -> None:
    """Top legend for input conditions + footer describing how the benchmark works."""
    handles = [
        Patch(
            facecolor=palette[CONDITION_LABELS.get(c, c)],
            edgecolor="0.3",
            label=CONDITION_LABELS.get(c, c),
        )
        for c in conditions
    ]
    fig.legend(
        handles=handles,
        title="Input condition (light → dark = leafy → wood-only)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(4, len(handles)),
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )

    lines = list(BENCHMARK_METHOD_LINES)
    lines.append("")
    lines.append("Input conditions (bar colors):")
    for condition in conditions:
        lines.append(f"• {CONDITION_EXPLANATIONS.get(condition, condition)}")

    metric_bits = []
    for metric in metrics:
        info = METRIC_INFO.get(metric)
        if not info:
            continue
        short = info[0].split("(")[0].strip()
        metric_bits.append(f"{short}: {info[1]}")
    if metric_bits:
        lines.append("")
        lines.append("Panel guide — " + " | ".join(metric_bits))

    text = "\n".join(textwrap.fill(line, width=118) for line in lines)
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=7.5,
        linespacing=1.3,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.95,
        },
    )


def _save_faceted_metrics(
    df: pd.DataFrame,
    metrics: list[str],
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
    width = max(11.0, 1.15 * len(species_order) * n_algos)
    fig, axes = plt.subplots(
        n_metrics,
        n_algos,
        figsize=(width, 4.4 * n_metrics + 2.4),
        sharey="row",
        squeeze=False,
    )

    for row, metric in enumerate(metrics):
        _, direction, definition = METRIC_INFO.get(metric, (metric, "", ""))
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
            else:
                ax.set_title("")
            if col == 0:
                ax.set_ylabel(_metric_ylabel(metric), fontsize=9)
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

        row_max = _y_limit(df[metric], metric=metric)
        for col in range(n_algos):
            axes[row][col].set_ylim(0, row_max)

        # Row banner: metric name + direction + short definition on the left panel.
        banner = _metric_title(metric)
        if definition:
            banner = f"{banner}\n{definition}"
        axes[row][0].annotate(
            banner,
            xy=(0.0, 1.08),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="0.15",
            annotation_clip=False,
        )

    _add_figure_guides(fig, condition_order, palette, metrics)
    fig.tight_layout(rect=(0, 0.24, 1, 0.94))
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved QSM benchmark plot to {out_path}")


def _save_condition_summary(
    df: pd.DataFrame,
    metrics: list[str],
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
    fig, axes = plt.subplots(nrows, 2, figsize=(12.5, 5.1 * nrows + 2.2))
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
        title, direction, definition = METRIC_INFO.get(metric, (metric, "", ""))
        ax.set_title(f"{title} (species-averaged)", fontsize=11, pad=30)
        caption_bits = []
        if direction:
            caption_bits.append(f"({direction})")
        if definition:
            caption_bits.append(definition)
        if caption_bits:
            ax.text(
                0.0,
                1.015,
                textwrap.fill(" ".join(caption_bits), width=72),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.5,
                color="0.3",
                clip_on=False,
            )
        ax.set_xlabel("")
        ax.set_ylabel(_metric_ylabel(metric), fontsize=9)
        ax.set_ylim(0, _y_limit(summary[metric], metric=metric))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    _add_figure_guides(fig, condition_order, palette, metrics)
    fig.tight_layout(rect=(0, 0.26, 1, 0.93))
    fig.subplots_adjust(hspace=0.55)
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

    _save_faceted_metrics(
        species_df,
        overview_metrics,
        args.out,
        condition_order=condition_order,
    )

    _save_faceted_metrics(
        species_df,
        pr_metrics,
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
        _sibling_out_path(args.out, "by_condition"),
        condition_order=condition_order,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
