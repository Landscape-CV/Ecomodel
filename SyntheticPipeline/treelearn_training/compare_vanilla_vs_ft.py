"""
Compare vanilla TreeLearn vs fine-tuned checkpoint on frozen val-16 tiles.

Usage (from SyntheticPipeline/):
  python treelearn_training/compare_vanilla_vs_ft.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_SP = _HERE.parent
_ROOT = _SP.parent
for p in (str(_ROOT), str(_SP), str(_SP / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_instance_segmentation import evaluate_tile  # noqa: E402


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.copy()
    if "error" in ok.columns:
        err = ok["error"]
        bad = err.notna() & err.astype(str).str.strip().ne("") & err.astype(str).str.lower().ne("nan")
        ok = ok.loc[~bad]
    if "f1" in ok.columns:
        ok = ok.loc[ok["f1"].notna()]
    if ok.empty:
        return pd.DataFrame()
    keys = ["model", "algorithm", "density_class", "composition"]
    for k in keys:
        if k not in ok.columns:
            ok[k] = "n/a"
    return (
        ok.groupby(keys, dropna=False)[["precision", "recall", "f1", "pq"]]
        .mean()
        .reset_index()
    )


def decide_next(summary: pd.DataFrame) -> str:
    if summary.empty:
        return (
            "No comparison metrics. Re-run after training. "
            "Do not proceed to SNAP FT until TreeLearn baseline compare works."
        )
    auto = summary[summary["algorithm"] == "treelearn"]
    if auto.empty:
        auto = summary

    def mean_f1(model: str, densities: List[str]) -> float:
        sub = auto[(auto["model"] == model) & (auto["density_class"].isin(densities))]
        if sub.empty:
            return float("nan")
        return float(sub["f1"].mean())

    v_all = mean_f1("vanilla", ["sparse", "moderate", "dense", "extreme"])
    f_all = mean_f1("ft", ["sparse", "moderate", "dense", "extreme"])
    v_hard = mean_f1("vanilla", ["dense", "extreme"])
    f_hard = mean_f1("ft", ["dense", "extreme"])
    d_all = f_all - v_all if np.isfinite(f_all) and np.isfinite(v_all) else float("nan")
    d_hard = f_hard - v_hard if np.isfinite(f_hard) and np.isfinite(v_hard) else float("nan")

    lines = [
        "## Kill / go criterion (TreeLearn FT)",
        "",
        f"- mean F1 (all): vanilla={v_all:.3f} → FT={f_all:.3f} (Δ={d_all:+.3f})",
        f"- mean F1 (dense+extreme): vanilla={v_hard:.3f} → FT={f_hard:.3f} (Δ={d_hard:+.3f})",
        "",
    ]
    success_hard = np.isfinite(d_hard) and d_hard >= 0.10
    success_overall = (
        np.isfinite(d_all)
        and d_all >= 0.05
        and (not np.isfinite(d_hard) or d_hard >= 0.0)
    )
    if np.isfinite(d_hard) and d_hard < -0.02:
        lines.append(
            "**Verdict: REGRESSION.** Keep vanilla `ecomodel.yaml` weights. "
            "Do not FT SNAP with the same recipe."
        )
    elif success_hard:
        lines.append(
            "**Verdict: SUCCESS (hard strata).** Dense+extreme F1 cleared +0.10. "
            "Keep `treelearn_checkpoints/best.pth`. SNAP FT still not the default next step."
        )
    elif success_overall:
        lines.append(
            "**Verdict: PARTIAL SUCCESS (overall only).** Overall F1 improved ≥ +0.05 "
            "without dense regression, but dense+extreme remains weak. "
            "Keep FT weights for easier strata; do **not** treat dense separation as solved. "
            "Do not auto-FT SNAP; next leverage is wood-only / hybrid / real TLS, not another "
            "heads-only synthetic FT."
        )
    else:
        lines.append(
            "**Verdict: WEAK / STOP.** Trivial or insufficient gain (Point-SAM-like). "
            "Even with ~112 synthetic train tiles, domain adapt did not fix dense/extreme. "
            "**Do not** immediately fine-tune SNAP the same way. Next: hybrid classical "
            "pipelines, better wood-only inputs, or real TLS — not another blind FT."
        )
    return "\n".join(lines)


def markdown_report(summary: pd.DataFrame, decision: str) -> str:
    lines = [
        "# TreeLearn vanilla vs fine-tuned",
        "",
        "## Mean metrics by model / stratum",
        "",
    ]
    if summary.empty:
        lines.append("_No successful rows._")
    else:
        lines.append("| model | density | composition | P | R | F1 | PQ |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for _, r in summary.iterrows():
            lines.append(
                f"| {r['model']} | {r['density_class']} | {r['composition']} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['pq']:.3f} |"
            )
    lines.extend(["", decision, ""])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, default=_SP / "testdataset" / "instance")
    ap.add_argument(
        "--split",
        type=Path,
        default=_SP / "treelearn_prepared" / "split.json",
    )
    ap.add_argument(
        "--vanilla_config",
        type=Path,
        default=_ROOT / "TreeLearn" / "configs" / "pipeline" / "ecomodel.yaml",
    )
    ap.add_argument(
        "--ft_config",
        type=Path,
        default=_ROOT / "TreeLearn" / "configs" / "pipeline" / "ecomodel_ft.yaml",
    )
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_SP / "output" / "treelearn_ft_compare",
    )
    args = ap.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    val_tiles = list(split.get("val") or split.get("val_files") or [])
    if not val_tiles:
        # fallback to pointsam split
        ps = json.loads((_SP / "pointsam_voxelized" / "split.json").read_text())
        val_tiles = list(ps["val"])

    models = {
        "vanilla": args.vanilla_config,
        "ft": args.ft_config,
    }
    for name, cfg in models.items():
        if not Path(cfg).is_file():
            raise SystemExit(f"Missing TreeLearn config for {name}: {cfg}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_folder = str(args.out_dir / "tmp_results")
    os.makedirs(results_folder, exist_ok=True)

    rows: List[dict] = []
    for model_name, cfg_path in models.items():
        print(f"\n=== Model: {model_name} ({cfg_path}) ===", flush=True)
        ckpt_kwargs = {
            "treelearn_config_path": str(cfg_path),
            "treelearn_use_gpu": True,
            "pointsam_ckpt": "",
            "pointsam_config": "large",
            "pointsam_use_gpu": True,
            "snap_ckpt": "",
            "snap_domain": "Outdoor",
            "snap_grid_size": 0.05,
            "snap_use_gpu": True,
        }
        for tile in val_tiles:
            prefix = str(args.dataset_dir / tile)
            print(f"  tile {tile} ...", flush=True)
            tile_rows = evaluate_tile(
                prefix,
                ["treelearn"],
                iou_thresh=args.iou_thresh,
                run_leaf_removal=False,
                results_folder=results_folder,
                ckpt_kwargs=ckpt_kwargs,
                save_predictions=False,
            )
            for r in tile_rows:
                r["model"] = model_name
                r["config"] = str(cfg_path)
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "compare_raw.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.out_dir / "compare_summary.csv", index=False)
    decision = decide_next(summary)
    (args.out_dir / "compare_report.md").write_text(
        markdown_report(summary, decision), encoding="utf-8"
    )
    (args.out_dir / "next_steps.md").write_text(decision + "\n", encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
