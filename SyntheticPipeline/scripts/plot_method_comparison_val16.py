"""
Merge synthetic val-16 results across methods into one comparison pack + PNGs.

Sources (frozen val-16 / same tiles):
  - pointsam_ft_compare/compare_raw.csv  (vanilla, decoder_ft, full_ft × auto)
  - treelearn_ft_compare/compare_raw.csv (vanilla, ft)
  - benchmark_treex_tls2trees_val16.csv  (treex, tls2trees, scanline)

Usage:
  python scripts/plot_method_comparison_val16.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SP = Path(__file__).resolve().parents[1]
_OUT = _SP / "output" / "method_comparison_val16"

# Display order / labels for the methods the user asked about
METHOD_ORDER = [
    "treelearn_vanilla",
    "treelearn_ft",
    "pointsam_vanilla",
    "pointsam_decoder_ft",
    "treex",
    "tls2trees",
]
METHOD_LABELS = {
    "treelearn_vanilla": "TreeLearn vanilla",
    "treelearn_ft": "TreeLearn FT",
    "pointsam_vanilla": "Point-SAM vanilla",
    "pointsam_decoder_ft": "Point-SAM decoder FT",
    "treex": "treeX",
    "tls2trees": "TLS2trees",
}
DENSITY_ORDER = ["sparse", "moderate", "dense", "extreme"]


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in df.columns:
        return df.copy()
    return df[df["error"].fillna("") == ""].copy()


def load_unified() -> pd.DataFrame:
    rows = []

    ps = _ok(pd.read_csv(_SP / "output" / "pointsam_ft_compare" / "compare_raw.csv"))
    ps = ps[(ps["prompt_mode"] == "auto") & (ps["model"].isin(["vanilla", "decoder_ft"]))]
    for _, r in ps.iterrows():
        key = "pointsam_vanilla" if r["model"] == "vanilla" else "pointsam_decoder_ft"
        rows.append({**r.to_dict(), "method": key})

    tl = _ok(pd.read_csv(_SP / "output" / "treelearn_ft_compare" / "compare_raw.csv"))
    for _, r in tl.iterrows():
        key = "treelearn_vanilla" if r["model"] == "vanilla" else "treelearn_ft"
        rows.append({**r.to_dict(), "method": key})

    tx = _ok(pd.read_csv(_SP / "output" / "benchmark_treex_tls2trees_val16.csv"))
    tx = tx[tx["algorithm"].isin(["treex", "tls2trees"])]
    for _, r in tx.iterrows():
        rows.append({**r.to_dict(), "method": r["algorithm"]})

    df = pd.DataFrame(rows)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["method_label"] = df["method"].map(METHOD_LABELS)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    hard = df["density_class"].isin(["dense", "extreme"])
    rows = []
    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        rows.append(
            {
                "method": m,
                "method_label": METHOD_LABELS[m],
                "f1_all": float(sub["f1"].mean()),
                "pq_all": float(sub["pq"].mean()),
                "f1_dense_extreme": float(sub.loc[hard[sub.index], "f1"].mean())
                if hard[sub.index].any()
                else float("nan"),
                "pq_dense_extreme": float(sub.loc[hard[sub.index], "pq"].mean())
                if hard[sub.index].any()
                else float("nan"),
                "n_tiles": int(sub["tile"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def by_density(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["method", "method_label", "density_class"], as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "f1_mean"})
    )
    return g


def plot_overview(summary: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(summary))
    w = 0.38
    ax.bar(x - w / 2, summary["f1_all"], w, label="F1 all strata", color="#4C8C7A")
    ax.bar(
        x + w / 2,
        summary["f1_dense_extreme"],
        w,
        label="F1 dense+extreme",
        color="#C47A3A",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method_label"], rotation=25, ha="right")
    ax.set_ylabel("Mean instance F1 @ IoU 0.5")
    ymax = float(np.nanmax(summary[["f1_all", "f1_dense_extreme"]].to_numpy()))
    ax.set_ylim(0, max(0.35, ymax * 1.15))
    ax.set_title("Synthetic val-16 — method comparison")
    ax.legend(frameon=False)
    ax.axhline(0.15, color="#8B3A2F", ls="--", lw=1, label=None)
    ax.text(
        0.99,
        0.15,
        " dense+extreme kill bar 0.15",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=9,
        color="#8B3A2F",
    )
    fig.tight_layout()
    fig.savefig(out / "f1_overview.png", dpi=160)
    plt.close(fig)


def plot_density_heatmap(dens: pd.DataFrame, out: Path) -> None:
    mat = dens.pivot(index="method_label", columns="density_class", values="f1_mean")
    mat = mat.reindex(
        index=[METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in mat.index],
        columns=[d for d in DENSITY_ORDER if d in mat.columns],
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        vmin=0,
        vmax=0.7,
        ax=ax,
        cbar_kws={"label": "Mean F1"},
    )
    ax.set_title("Mean F1 by density × method (val-16)")
    ax.set_xlabel("Density class")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out / "f1_by_density_heatmap.png", dpi=160)
    plt.close(fig)


def plot_density_bars(dens: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    order_labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    sns.barplot(
        data=dens,
        x="density_class",
        y="f1_mean",
        hue="method_label",
        order=DENSITY_ORDER,
        hue_order=order_labels,
        ax=ax,
        palette="colorblind",
    )
    ax.set_ylabel("Mean F1 @ IoU 0.5")
    ax.set_xlabel("Density")
    ax.set_title("F1 by density (synthetic val-16)")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "f1_by_density_bars.png", dpi=160)
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, dens: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Method comparison — synthetic val-16",
        "",
        "Same frozen 16-tile split. Auto prompts for Point-SAM. "
        "TreeLearn FT = heads-only synthetic FT. Point-SAM FT = decoder_ft.",
        "",
        "## Mean F1",
        "",
        "| method | F1 all | F1 dense+extreme | PQ all |",
        "|--------|-------:|-----------------:|-------:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['method_label']} | {r['f1_all']:.3f} | {r['f1_dense_extreme']:.3f} | {r['pq_all']:.3f} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `f1_overview.png` — all vs hard strata",
        "- `f1_by_density_heatmap.png`",
        "- `f1_by_density_bars.png`",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_unified()
    df.to_csv(out / "unified_raw.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    dens = by_density(df)
    dens.to_csv(out / "by_density.csv", index=False)

    plot_overview(summary, out)
    plot_density_heatmap(dens, out)
    plot_density_bars(dens, out)
    write_markdown(summary, dens, out)

    # JSON for canvas / other consumers
    payload = {
        "summary": summary.to_dict(orient="records"),
        "by_density": dens.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", out)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
