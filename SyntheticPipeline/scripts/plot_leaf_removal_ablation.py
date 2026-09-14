"""Plot leaf-on vs leaf-off (--leaf_removal) instance-seg ablation.

Compares the Phase-0 ablation CSV against matching tiles/algorithms from the
leaf-on full benchmark CSV.

Writes under --out stem:
  1) F1 / PQ by density  (* .png)
  2) Delta F1 heatmap    (*_delta_f1.png)
  3) Mixed vs mono split (*_by_composition.png)
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

ALGORITHM_LABELS = {
    "scanline": "Scanline",
    "treelearn": "TreeLearn",
    "pointsam": "Point-SAM (auto)",
    "pointsam_oracle": "Point-SAM (oracle)",
    "snap": "SNAP (auto)",
    "snap_oracle": "SNAP (oracle)",
}

LEAF_LABELS = {
    "leaf_on": "Leaf-on (default)",
    "leaf_off": "Leaf-off (--leaf_removal)",
}

LEAF_COLORS = {
    "leaf_on": "#4C8C7A",
    "leaf_off": "#C47A3A",
}


def _algo_label(name: str) -> str:
    return ALGORITHM_LABELS.get(str(name), str(name))


def _density_label(name: str) -> str:
    return DENSITY_LABELS.get(str(name), str(name).title())


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "error" in out.columns:
        out = out[out["error"].isna() | (out["error"].astype(str).str.strip() == "")]
    if "prompt_mode" in out.columns:
        out = out[out["prompt_mode"].astype(str) == "auto"]
    for col in ("f1", "pq", "precision", "recall"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_comparison(leaf_off: pd.DataFrame, leaf_on: pd.DataFrame) -> pd.DataFrame:
    tiles = set(leaf_off["tile"].unique())
    algs = set(leaf_off["algorithm"].unique())
    on = leaf_on[leaf_on["tile"].isin(tiles) & leaf_on["algorithm"].isin(algs)].copy()
    off = leaf_off.copy()
    on["leaf_setting"] = "leaf_on"
    off["leaf_setting"] = "leaf_off"
    cols = [
        c
        for c in [
            "tile",
            "algorithm",
            "density_class",
            "composition",
            "prompt_mode",
            "f1",
            "pq",
            "precision",
            "recall",
            "leaf_setting",
        ]
        if c in on.columns or c in off.columns
    ]
    merged = pd.concat([on[cols], off[cols]], ignore_index=True)
    merged["Algorithm"] = merged["algorithm"].map(_algo_label)
    merged["Density"] = merged["density_class"].map(_density_label)
    merged["Leaf"] = merged["leaf_setting"].map(LEAF_LABELS)
    if "composition" in merged.columns:
        merged["Composition"] = merged["composition"].astype(str).str.title()
    return merged


def _save_by_density(df: pd.DataFrame, out_path: str, metrics: list[str]) -> None:
    density_order = [d for d in DENSITY_ORDER if d in set(df["density_class"])]
    label_order = [_density_label(d) for d in density_order]
    leaf_order = [LEAF_LABELS["leaf_on"], LEAF_LABELS["leaf_off"]]
    summary = (
        df.groupby(["Algorithm", "Density", "Leaf", "density_class"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    algos = sorted(df["Algorithm"].unique())
    fig, axes = plt.subplots(
        len(metrics),
        len(algos),
        figsize=(3.6 * len(algos) + 1.5, 3.8 * len(metrics) + 1.4),
        sharey="row",
        squeeze=False,
    )

    for r, metric in enumerate(metrics):
        for c, algo in enumerate(algos):
            ax = axes[r][c]
            sub = summary[summary["Algorithm"] == algo]
            sns.barplot(
                data=sub,
                x="Density",
                y=metric,
                hue="Leaf",
                order=label_order,
                hue_order=leaf_order,
                palette={LEAF_LABELS[k]: LEAF_COLORS[k] for k in ("leaf_on", "leaf_off")},
                ax=ax,
                errorbar=None,
            )
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("")
            if c == 0:
                ax.set_ylabel("F1 @ IoU 0.5" if metric == "f1" else metric.upper(), fontsize=9)
            else:
                ax.set_ylabel("")
            if r == 0:
                ax.set_title(algo, fontsize=11, pad=8)
            ax.tick_params(axis="x", rotation=25, labelsize=7.5)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", padding=2, fontsize=6.5)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    handles = [
        Patch(facecolor=LEAF_COLORS["leaf_on"], label=LEAF_LABELS["leaf_on"]),
        Patch(facecolor=LEAF_COLORS["leaf_off"], label=LEAF_LABELS["leaf_off"]),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Leaf-on vs leaf-off instance segmentation (matched tiles)",
        fontsize=13,
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        textwrap.fill(
            "Leaf-off = same stratified tiles with --leaf_removal (RGI). "
            "Leaf-on = matching tiles from the full leaf-on benchmark CSV.",
            width=110,
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_delta_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Delta F1 = leaf_off − leaf_on (positive => leaf removal helped)."""
    density_order = [d for d in DENSITY_ORDER if d in set(df["density_class"])]
    piv = (
        df.groupby(["Algorithm", "density_class", "leaf_setting"], as_index=False)["f1"]
        .mean(numeric_only=True)
        .pivot_table(index="Algorithm", columns=["density_class", "leaf_setting"], values="f1")
    )
    rows = []
    for algo in piv.index:
        for dens in density_order:
            try:
                on = float(piv.loc[algo, (dens, "leaf_on")])
                off = float(piv.loc[algo, (dens, "leaf_off")])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({"Algorithm": algo, "density_class": dens, "delta_f1": off - on})
    if not rows:
        print("No delta F1 rows to plot.", file=sys.stderr)
        return
    delta = pd.DataFrame(rows)
    mat = delta.pivot(index="Algorithm", columns="density_class", values="delta_f1")
    mat = mat.reindex(columns=density_order)
    mat.columns = [_density_label(c) for c in mat.columns]

    lim = max(0.15, float(np.nanmax(np.abs(mat.values))))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(mat.columns) + 3), max(3.8, 0.7 * len(mat) + 2)))
    sns.heatmap(
        mat,
        annot=True,
        fmt="+.2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-lim,
        vmax=lim,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Δ F1 (leaf-off − leaf-on)"},
    )
    ax.set_title("Leaf removal effect on instance F1 (positive = leaf-off better)", fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_by_composition(df: pd.DataFrame, out_path: str) -> None:
    if "Composition" not in df.columns or "f1" not in df.columns:
        return
    density_order = [d for d in DENSITY_ORDER if d in set(df["density_class"])]
    label_order = [_density_label(d) for d in density_order]
    leaf_order = [LEAF_LABELS["leaf_on"], LEAF_LABELS["leaf_off"]]
    comps = sorted(df["Composition"].dropna().unique())
    algos = sorted(df["Algorithm"].unique())

    fig, axes = plt.subplots(
        len(comps),
        len(algos),
        figsize=(3.6 * len(algos) + 1.2, 3.6 * len(comps) + 1.2),
        sharey=True,
        squeeze=False,
    )
    summary = (
        df.groupby(["Composition", "Algorithm", "Density", "Leaf"], as_index=False)["f1"]
        .mean(numeric_only=True)
    )
    for r, comp in enumerate(comps):
        for c, algo in enumerate(algos):
            ax = axes[r][c]
            sub = summary[(summary["Composition"] == comp) & (summary["Algorithm"] == algo)]
            sns.barplot(
                data=sub,
                x="Density",
                y="f1",
                hue="Leaf",
                order=label_order,
                hue_order=leaf_order,
                palette={LEAF_LABELS[k]: LEAF_COLORS[k] for k in ("leaf_on", "leaf_off")},
                ax=ax,
                errorbar=None,
            )
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel("F1" if c == 0 else "")
            if r == 0:
                ax.set_title(algo, fontsize=11)
            if c == 0:
                ax.annotate(
                    comp,
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.tick_params(axis="x", rotation=25, labelsize=7)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    handles = [
        Patch(facecolor=LEAF_COLORS["leaf_on"], label=LEAF_LABELS["leaf_on"]),
        Patch(facecolor=LEAF_COLORS["leaf_off"], label=LEAF_LABELS["leaf_off"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Leaf-on vs leaf-off F1 by composition", fontsize=13, y=1.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sp_out = os.path.join(root, "SyntheticPipeline", "output")
    parser = argparse.ArgumentParser(description="Plot leaf-on vs leaf-off ablation")
    parser.add_argument(
        "--leaf_off_csv",
        default=os.path.join(sp_out, "benchmark_instance_leaf_removal_ablation.csv"),
    )
    parser.add_argument(
        "--leaf_on_csv",
        default=os.path.join(sp_out, "benchmark_instance_4method_full.csv"),
        help="Leaf-on baseline (matched tiles/algorithms pulled automatically)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            sp_out, "visualizations", "benchmark_instance_leaf_removal_ablation.png"
        ),
    )
    args = parser.parse_args()

    for path, label in ((args.leaf_off_csv, "leaf-off"), (args.leaf_on_csv, "leaf-on")):
        if not os.path.isfile(path):
            print(f"{label} CSV not found: {path}", file=sys.stderr)
            return 1

    off = _prepare(pd.read_csv(args.leaf_off_csv))
    on = _prepare(pd.read_csv(args.leaf_on_csv))
    if off.empty:
        print("No successful leaf-off rows.", file=sys.stderr)
        return 1

    df = _build_comparison(off, on)
    print(
        f"Comparing {df[df.leaf_setting=='leaf_off'].tile.nunique()} tiles × "
        f"{df.algorithm.nunique()} algorithms "
        f"({len(df)} rows total)"
    )

    stem, ext = os.path.splitext(args.out)
    if not ext:
        ext = ".png"
        args.out = stem + ext

    metrics = [m for m in ("f1", "pq") if m in df.columns]
    _save_by_density(df, args.out, metrics)
    _save_delta_heatmap(df, f"{stem}_delta_f1{ext}")
    _save_by_composition(df, f"{stem}_by_composition{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
