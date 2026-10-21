"""Grouped F1 / precision / recall charts for the real TLS val benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORESTS = [
    ("l1w", "L1W"),
    ("wytham_isprs_test", "Wytham ISPRS"),
    ("wytham_qutuwu", "Wytham qutuwu"),
]
METHOD_ORDER = [
    "forestformer3d", "treelearn", "pointsam", "scanline", "treex", "tls2trees",
]


def method_label(algorithm: str, weights: str) -> str:
    weights = (weights or "").strip()
    if algorithm.startswith("pointsam") and weights:
        return f"pointsam ({weights})"
    return algorithm


def _grouped_bars(ok, ran, color_of, forest_keys, forest_names, column, ylabel, title, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(forest_names))
    width = 0.8 / max(len(ran), 1)
    for i, label in enumerate(ran):
        heights = []
        for key in forest_keys:
            sub = ok[(ok["label"] == label) & (ok["density_class"] == key)]
            vals = sub[column].dropna() if column in sub.columns else pd.Series(dtype=float)
            heights.append(float(vals.mean()) if len(vals) else np.nan)
        offset = (i - (len(ran) - 1) / 2) * width
        bars = ax.bar(x + offset, [0 if np.isnan(h) else h for h in heights], width,
                      label=label, color=color_of[label])
        for bar, h in zip(bars, heights):
            if np.isnan(h):
                bar.set_height(0)
                bar.set_alpha(0)
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(forest_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot(csv_path: Path, f1_path: Path, pr_path: Path, tree_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if "error" not in df.columns:
        df["error"] = ""
    ok = df[df["error"].fillna("").astype(str).str.strip() == ""].copy()
    if "weights" not in ok.columns:
        ok["weights"] = ""
    ok["weights"] = ok["weights"].fillna("").astype(str)
    ok["label"] = [method_label(a, w) for a, w in zip(ok["algorithm"], ok["weights"])]

    ran = []
    for alg in METHOD_ORDER:
        labels = ok.loc[ok["algorithm"] == alg, "label"].unique().tolist()
        ran.extend(labels)
    for label in ok["label"].unique():
        if label not in ran:
            ran.append(label)
    if not ran:
        raise SystemExit(f"No successful rows in {csv_path}")

    palette = plt.cm.tab10(np.linspace(0, 1, max(len(ran), 1)))
    color_of = {label: palette[i] for i, label in enumerate(ran)}
    forest_keys = [k for k, _ in FORESTS]
    forest_names = [n for _, n in FORESTS]

    _grouped_bars(
        ok, ran, color_of, forest_keys, forest_names, "f1",
        "Plot-level F1 (IoU 0.5)", "Plot-level F1 by forest", f1_path,
    )
    if "tree_f1" in ok.columns and ok["tree_f1"].notna().any():
        tree_col, tree_ylabel = "tree_f1", "Tree-level mean F1"
    elif "tree_iou" in ok.columns and ok["tree_iou"].notna().any():
        tree_col, tree_ylabel = "tree_iou", "Tree-level mean IoU"
    else:
        tree_col, tree_ylabel = "mean_matched_iou", "Tree-level mean IoU"
    tree_ran = [
        label for label in ran
        if tree_col in ok.columns and ok.loc[ok["label"] == label, tree_col].notna().any()
    ]
    _grouped_bars(
        ok, tree_ran, color_of, forest_keys, forest_names, tree_col, tree_ylabel,
        "Tree-level score of matched trees", tree_path,
    )

    fig, ax = plt.subplots(figsize=(8, 4.6))
    means = ok.groupby("label", sort=False)[["precision", "recall"]].mean()
    means = means.reindex([m for m in ran if m in means.index])
    x = np.arange(len(means))
    w = 0.36
    p_bars = ax.bar(x - w / 2, means["precision"], w, label="precision", color="#4C78A8")
    r_bars = ax.bar(x + w / 2, means["recall"], w, label="recall", color="#F58518")
    for bars in (p_bars, r_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(means.index), rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean over scored tiles")
    ax.set_title("Plot-level precision and recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {f1_path}")
    print(f"Wrote {pr_path}")
    print(f"Wrote {tree_path}")


def main():
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=here / "output" / "benchmark_real_tls_val.csv")
    ap.add_argument("--f1", type=Path, default=here / "output" / "benchmark_real_tls_val_f1.png")
    ap.add_argument("--pr", type=Path, default=here / "output" / "benchmark_real_tls_val_pr.png")
    ap.add_argument("--tree", type=Path, default=here / "output" / "benchmark_real_tls_val_tree_f1.png")
    args = ap.parse_args()
    plot(args.csv, args.f1, args.pr, args.tree)


if __name__ == "__main__":
    main()
