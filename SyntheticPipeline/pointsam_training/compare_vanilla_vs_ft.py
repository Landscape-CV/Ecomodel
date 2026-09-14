"""
Compare vanilla Point-SAM vs fine-tuned checkpoint(s) on val tiles.

Usage (from SyntheticPipeline/):
  python pointsam_training/compare_vanilla_vs_ft.py
  python pointsam_training/compare_vanilla_vs_ft.py --ft_ckpts pointsam_checkpoints/decoder_ft/best.safetensors
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

from benchmark_instance_segmentation import (  # noqa: E402
    evaluate_tile,
    expand_algorithms,
)


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
    keys = ["model", "algorithm", "prompt_mode", "density_class", "composition"]
    for k in keys:
        if k not in ok.columns:
            ok[k] = "n/a"
    return (
        ok.groupby(keys, dropna=False)[["precision", "recall", "f1", "pq"]]
        .mean()
        .reset_index()
    )


def markdown_report(summary: pd.DataFrame, decision: str) -> str:
    lines = [
        "# Point-SAM vanilla vs fine-tuned",
        "",
        "## Mean metrics by model / stratum",
        "",
    ]
    if summary.empty:
        lines.append("_No successful rows._")
    else:
        lines.append(
            "| model | algorithm | density | composition | P | R | F1 | PQ |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for _, r in summary.iterrows():
            lines.append(
                f"| {r['model']} | {r['algorithm']} | {r['density_class']} | "
                f"{r['composition']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['pq']:.3f} |"
            )
    lines.extend(["", "## Next-step gate", "", decision, ""])
    return "\n".join(lines)


def decide_next(summary: pd.DataFrame) -> str:
    """Go/no-go for SNAP / TreeLearn based on dense/extreme F1 deltas."""
    if summary.empty:
        return (
            "No comparison metrics available. Re-run after training completes. "
            "Default next: inspect training logs before trying SNAP/TreeLearn."
        )

    auto = summary[
        (summary["prompt_mode"] == "auto")
        | (summary["algorithm"].astype(str).str.endswith("_oracle") == False)
    ]
    # Prefer rows tagged algorithm pointsam (auto)
    auto = summary[summary["algorithm"] == "pointsam"]
    if auto.empty:
        auto = summary

    def mean_f1(model: str, densities: List[str]) -> float:
        sub = auto[
            (auto["model"] == model) & (auto["density_class"].isin(densities))
        ]
        if sub.empty:
            return float("nan")
        return float(sub["f1"].mean())

    models = sorted(auto["model"].unique())
    vanilla = "vanilla"
    fts = [m for m in models if m != vanilla]
    if not fts:
        return "Only vanilla present; cannot decide. Train a fine-tuned checkpoint first."

    best_ft = max(fts, key=lambda m: mean_f1(m, ["dense", "extreme"]))
    v_hard = mean_f1(vanilla, ["dense", "extreme"])
    f_hard = mean_f1(best_ft, ["dense", "extreme"])
    v_all = mean_f1(vanilla, ["sparse", "moderate", "dense", "extreme"])
    f_all = mean_f1(best_ft, ["sparse", "moderate", "dense", "extreme"])

    delta_hard = f_hard - v_hard if np.isfinite(f_hard) and np.isfinite(v_hard) else float("nan")
    delta_all = f_all - v_all if np.isfinite(f_all) and np.isfinite(v_all) else float("nan")

    lines = [
        f"Best FT model on dense/extreme: **{best_ft}**",
        f"- mean auto F1 (all strata): vanilla={v_all:.3f} → FT={f_all:.3f} (Δ={delta_all:+.3f})",
        f"- mean auto F1 (dense+extreme): vanilla={v_hard:.3f} → FT={f_hard:.3f} (Δ={delta_hard:+.3f})",
        "",
    ]
    if np.isfinite(delta_hard) and delta_hard >= 0.03:
        lines.append(
            "**Recommendation:** Point-SAM FT helps on hard strata. "
            "Next: iterate Point-SAM (more crops / longer decoder FT) or try "
            "**TreeLearn domain adaptation** (already weak on dense/extreme)."
        )
    elif np.isfinite(delta_all) and delta_all >= 0.02:
        lines.append(
            "**Recommendation:** Modest overall gain. Keep the FT weights for "
            "promptable use; next try **SNAP Outdoor** adaptation on the same "
            "voxelized GT, then TreeLearn if still needed."
        )
    else:
        lines.append(
            "**Recommendation:** Little/no gain from Point-SAM FT on this synthetic "
            "set. Next: try **SNAP Outdoor** fine-tune / prompt adaptation on the "
            "same `pointsam_voxelized` labels, then TreeLearn domain adapt."
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        type=Path,
        default=_SP / "testdataset" / "instance",
    )
    ap.add_argument(
        "--split",
        type=Path,
        default=_SP / "pointsam_voxelized" / "split.json",
    )
    ap.add_argument(
        "--vanilla_ckpt",
        type=Path,
        default=_ROOT / "thirdparty" / "checkpoints" / "point_sam" / "model.safetensors",
    )
    ap.add_argument(
        "--ft_ckpts",
        type=str,
        nargs="*",
        default=None,
        help="Fine-tuned safetensors paths (default: decoder_ft then full_ft if exist)",
    )
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--prompt_mode", type=str, default="both", choices=["auto", "oracle", "both"])
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_SP / "output" / "pointsam_ft_compare",
    )
    ap.add_argument("--max_val_tiles", type=int, default=None, help="Debug subset of val tiles")
    args = ap.parse_args()

    split = json.loads(args.split.read_text())
    val_tiles = list(split["val"])
    if args.max_val_tiles:
        val_tiles = val_tiles[: args.max_val_tiles]

    ft_ckpts: Dict[str, Path] = {}
    if args.ft_ckpts:
        for p in args.ft_ckpts:
            path = Path(p)
            if not path.is_absolute():
                path = (_SP / path).resolve()
            ft_ckpts[path.parent.name] = path
    else:
        for name in ("decoder_ft", "full_ft"):
            cand = _SP / "pointsam_checkpoints" / name / "best.safetensors"
            if cand.is_file():
                ft_ckpts[name] = cand

    models = {"vanilla": args.vanilla_ckpt, **ft_ckpts}
    for name, ckpt in models.items():
        if not Path(ckpt).is_file():
            raise SystemExit(f"Missing checkpoint for {name}: {ckpt}")

    algorithms = expand_algorithms(["pointsam"], args.prompt_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_folder = str(args.out_dir / "tmp_results")
    os.makedirs(results_folder, exist_ok=True)

    rows: List[dict] = []
    for model_name, ckpt in models.items():
        print(f"\n=== Model: {model_name} ({ckpt}) ===")
        ckpt_kwargs = {
            "pointsam_ckpt": str(ckpt),
            "pointsam_config": "large",
            "pointsam_use_gpu": True,
            "treelearn_config_path": "",
            "treelearn_use_gpu": True,
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
                algorithms,
                iou_thresh=args.iou_thresh,
                run_leaf_removal=False,
                results_folder=results_folder,
                ckpt_kwargs=ckpt_kwargs,
                save_predictions=False,
            )
            for r in tile_rows:
                r["model"] = model_name
                r["ckpt"] = str(ckpt)
                rows.append(r)

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "compare_raw.csv"
    df.to_csv(csv_path, index=False)
    summary = summarize(df)
    summary_path = args.out_dir / "compare_summary.csv"
    summary.to_csv(summary_path, index=False)

    decision = decide_next(summary)
    md = markdown_report(summary, decision)
    md_path = args.out_dir / "compare_report.md"
    md_path.write_text(md, encoding="utf-8")
    (args.out_dir / "next_steps.md").write_text(decision + "\n", encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print("\n" + decision)


if __name__ == "__main__":
    main()
