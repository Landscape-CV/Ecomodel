"""Plot instance-segmentation benchmark CSV to PNGs (non-interactive).

Always writes (under --out stem):
  1) F1 / PQ overview by density (* .png)
  2) Precision / Recall comparison (*_precision_recall.png)
  3) Algorithm x density summary (*_by_density.png)
  4) Mixed vs mono composition split (*_by_composition.png)

Mirrors the structure of plot_benchmark_qsm.py for SyntheticPipeline results.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

DENSITY_ORDER = ["sparse", "moderate", "dense", "extreme"]

DENSITY_LABELS = {
    "sparse": "Sparse (~2 trees)",
    "moderate": "Moderate (~6 trees)",
    "dense": "Dense (~15 trees)",
    "extreme": "Extreme (~30 trees)",
}

DENSITY_EXPLANATIONS = {
    "sparse": "Sparse: low overlap; stem cues usually clear.",
    "moderate": "Moderate: typical TLS plot density with light crown contact.",
    "dense": "Dense: frequent crown overlap; harder instance boundaries.",
    "extreme": "Extreme: heavy occlusion / overlap stress test.",
}

# Cool→warm density ramp (readable on light backgrounds; not purple defaults).
DENSITY_TINTS = {
    "sparse": "#7EB8A8",
    "moderate": "#4C8C7A",
    "dense": "#C47A3A",
    "extreme": "#8B3A2F",
}

ALGORITHM_LABELS = {
    "scanline": "Scanline",
    "treelearn": "TreeLearn",
    "pointsam": "Point-SAM (auto)",
    "pointsam_oracle": "Point-SAM (oracle)",
    "snap": "SNAP (auto)",
    "snap_oracle": "SNAP (oracle)",
}

METRIC_INFO = {
    "f1": (
        "Instance F1 @ IoU 0.5",
        "higher is better",
        "Harmonic mean of instance precision and recall after bipartite matching.",
    ),
    "pq": (
        "Panoptic Quality (PQ)",
        "higher is better",
        "SQ × RQ: matched IoU quality times detection F1-like recognition quality.",
    ),
    "precision": (
        "Instance precision",
        "higher is better",
        "Fraction of predicted trees matched to a GT tree (IoU ≥ threshold).",
    ),
    "recall": (
        "Instance recall",
        "higher is better",
        "Fraction of GT trees recovered by a matched prediction.",
    ),
    "sq": (
        "Segmentation Quality (SQ)",
        "higher is better",
        "Mean IoU over matched GT–pred pairs only.",
    ),
    "rq": (
        "Recognition Quality (RQ)",
        "higher is better",
        "Detection-style F1 over matched / unmatched instances.",
    ),
    "mean_matched_iou": (
        "Mean matched IoU",
        "higher is better",
        "Average IoU of successfully matched tree instances.",
    ),
}


def _algo_label(name: str) -> str:
    return ALGORITHM_LABELS.get(str(name), str(name))


def _density_label(name: str) -> str:
    return DENSITY_LABELS.get(str(name), str(name))


def _ordered_densities(series: pd.Series) -> list[str]:
    present = {str(x) for x in series.dropna().unique()}
    ordered = [d for d in DENSITY_ORDER if d in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _y_limit(values: pd.Series, *, metric: str) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return 1.0
    hi = float(vals.max())
    if metric in {"f1", "pq", "precision", "recall", "sq", "rq", "mean_matched_iou"}:
        return min(1.05, max(0.25, hi * 1.15 + 0.05))
    return max(hi * 1.15, 1e-3)


def _metric_ylabel(metric: str) -> str:
    title, direction, _ = METRIC_INFO.get(metric, (metric, "", ""))
    if direction:
        return f"{title}\n({direction})"
    return title


def _add_figure_guides(
    fig: plt.Figure,
    density_order: list[str],
    palette: dict[str, str],
    metrics: list[str],
) -> None:
    handles = [
        Patch(facecolor=palette[_density_label(d)], edgecolor="0.3", label=_density_label(d))
        for d in density_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(4, len(handles)),
        frameon=True,
        fontsize=9,
        title="Density class",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.995),
    )

    lines = ["Density guide"]
    for d in density_order:
        lines.append(f"• {DENSITY_EXPLANATIONS.get(d, d)}")
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


def _prepare_success(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "error" in out.columns:
        err = out["error"].astype(str).str.strip()
        out = out[err.isna() | (err == "") | (err == "nan")].copy()
    for col in ("f1", "pq", "precision", "recall"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "f1" in out.columns:
        out = out[out["f1"].notna()].copy()
    out["Algorithm"] = out["algorithm"].map(_algo_label)
    out["Density"] = out["density_class"].map(_density_label)
    if "composition" in out.columns:
        out["Composition"] = out["composition"].astype(str).str.title()
    if "prompt_mode" in out.columns:
        out["Prompt"] = out["prompt_mode"].astype(str)
    return out


def _save_overview_by_density(
    df: pd.DataFrame,
    metrics: list[str],
    out_path: str,
    *,
    density_order: list[str],
) -> None:
    label_order = [_density_label(d) for d in density_order]
    palette = {_density_label(d): DENSITY_TINTS.get(d, "#888888") for d in density_order}
    summary = (
        df.groupby(["Algorithm", "Density", "density_class"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    algo_order = sorted(summary["Algorithm"].unique())

    nrows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13.0, 5.0 * nrows + 2.2))
    axes = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=summary,
            x="Algorithm",
            y=metric,
            hue="Density",
            order=algo_order,
            hue_order=label_order,
            palette=palette,
            ax=ax,
            errorbar=None,
        )
        title, direction, definition = METRIC_INFO.get(metric, (metric, "", ""))
        ax.set_title(f"{title} (tile-averaged)", fontsize=11, pad=28)
        caption = " ".join(x for x in (f"({direction})" if direction else "", definition) if x)
        if caption:
            ax.text(
                0.0,
                1.015,
                textwrap.fill(caption, width=72),
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
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=7)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    _add_figure_guides(fig, density_order, palette, metrics)
    fig.tight_layout(rect=(0, 0.26, 1, 0.93))
    fig.subplots_adjust(hspace=0.55)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance benchmark plot to {out_path}")


def _save_faceted_by_composition(
    df: pd.DataFrame,
    metrics: list[str],
    out_path: str,
    *,
    density_order: list[str],
) -> None:
    """Columns = algorithm; hue = density; rows = metrics; optional facet via composition panels."""
    if "Composition" not in df.columns:
        return
    label_order = [_density_label(d) for d in density_order]
    palette = {_density_label(d): DENSITY_TINTS.get(d, "#888888") for d in density_order}
    compositions = sorted(df["Composition"].dropna().unique())
    algorithms = sorted(df["Algorithm"].unique())

    n_metrics = len(metrics)
    n_comp = max(1, len(compositions))
    fig, axes = plt.subplots(
        n_metrics,
        n_comp,
        figsize=(max(10.0, 5.5 * n_comp), 4.2 * n_metrics + 2.4),
        sharey="row",
        squeeze=False,
    )

    for row, metric in enumerate(metrics):
        _, _, definition = METRIC_INFO.get(metric, (metric, "", ""))
        for col, composition in enumerate(compositions):
            ax = axes[row][col]
            subset = df[df["Composition"] == composition]
            summary = (
                subset.groupby(["Algorithm", "Density"], as_index=False)[metric]
                .mean(numeric_only=True)
            )
            sns.barplot(
                data=summary,
                x="Algorithm",
                y=metric,
                hue="Density",
                order=algorithms,
                hue_order=label_order,
                palette=palette,
                ax=ax,
                errorbar=None,
            )
            if row == 0:
                ax.set_title(composition, fontsize=13, fontweight="bold")
            else:
                ax.set_title("")
            if col == 0:
                ax.set_ylabel(_metric_ylabel(metric), fontsize=9)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_ylim(0, _y_limit(df[metric], metric=metric))
            ax.tick_params(axis="x", rotation=25, labelsize=7)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", padding=1, fontsize=6)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        banner = METRIC_INFO.get(metric, (metric, "", ""))[0]
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

    _add_figure_guides(fig, density_order, palette, metrics)
    fig.tight_layout(rect=(0, 0.24, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance benchmark plot to {out_path}")


def _save_heatmap_f1(df: pd.DataFrame, out_path: str, *, density_order: list[str]) -> None:
    if "f1" not in df.columns:
        return
    pivot = (
        df.groupby(["Algorithm", "density_class"], as_index=False)["f1"]
        .mean(numeric_only=True)
        .pivot(index="Algorithm", columns="density_class", values="f1")
    )
    cols = [c for c in density_order if c in pivot.columns] + [
        c for c in pivot.columns if c not in density_order
    ]
    pivot = pivot.reindex(columns=cols)
    pivot.columns = [_density_label(c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(pivot.columns) + 3), max(4.5, 0.55 * len(pivot) + 2)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean F1 @ IoU 0.5"},
    )
    ax.set_title("Instance F1 by algorithm × density", fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved instance benchmark plot to {out_path}")


def main() -> int:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_csv = os.path.join(
        root_dir, "SyntheticPipeline", "output", "benchmark_instance_4method_full.csv"
    )
    default_out = os.path.join(
        root_dir,
        "SyntheticPipeline",
        "output",
        "visualizations",
        "benchmark_instance_4method_full.png",
    )
    parser = argparse.ArgumentParser(description="Plot instance segmentation benchmark results")
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--out", default=default_out)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)
    df = _prepare_success(df)
    if df.empty:
        print("No successful benchmark rows with metrics to plot.")
        return 1

    density_order = _ordered_densities(df["density_class"])
    overview_metrics = [m for m in ["f1", "pq", "mean_matched_iou"] if m in df.columns]
    pr_metrics = [m for m in ["precision", "recall", "sq", "rq"] if m in df.columns]
    if "f1" not in overview_metrics:
        print("CSV is missing f1.", file=sys.stderr)
        return 1

    print(f"Plotting {len(df)} successful rows...")
    stem, ext = os.path.splitext(args.out)
    if not ext:
        ext = ".png"
        args.out = stem + ext

    _save_overview_by_density(df, overview_metrics, args.out, density_order=density_order)
    if pr_metrics:
        _save_overview_by_density(
            df,
            pr_metrics,
            f"{stem}_precision_recall{ext}",
            density_order=density_order,
        )
    _save_heatmap_f1(df, f"{stem}_by_density{ext}", density_order=density_order)
    if "Composition" in df.columns:
        _save_faceted_by_composition(
            df,
            [m for m in ["f1", "pq"] if m in df.columns],
            f"{stem}_by_composition{ext}",
            density_order=density_order,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
