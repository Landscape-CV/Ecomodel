"""
Benchmark tree instance segmentation on multi-tree synthetic tiles.

Expects tiles produced by ``generate_instance_benchmark.py``:
  {tile}_scan.laz, {tile}_instances.npy, {tile}_meta.json

Metrics (Hungarian match, ignore ground/clutter labels < 0):
  - precision / recall / F1 / PQ at IoU 0.5
  - mean matched IoU
  - counts of TP / FP / FN instances

Results are broken down by density_class and composition from meta.json.

Usage:
  python scripts/benchmark_instance_segmentation.py \\
      --dataset_dir testdataset/instance \\
      --out_csv output/benchmark_instance.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import laspy
except ImportError:
    print("Please pip install laspy lazrs pandas scipy")
    sys.exit(1)

# Repo roots
_SP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_SP_DIR)
for p in (_ROOT, _SP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.instance_metrics import match_instances, pairwise_iou_matrix


GROUND_ID = -1
CLUTTER_ID = -2


def discover_tiles(dataset_dir: str) -> List[str]:
    """Return tile prefixes that have both scan.laz and instances.npy."""
    prefixes = []
    for laz in glob.glob(os.path.join(dataset_dir, "*_scan.laz")):
        base = laz[: -len("_scan.laz")]
        if os.path.exists(base + "_instances.npy"):
            prefixes.append(base)
    return sorted(prefixes)


def run_segmenter_on_tile(
    points_xyzi: np.ndarray,
    segmenter_type: str,
    treelearn_config_path: str = "",
    treelearn_use_gpu: bool = True,
    run_leaf_removal: bool = False,
    results_folder: str = "results",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run EcomodelLite preprocessing + instance segmentation.

    Leaf removal is off by default so instance tools are scored on leafy clouds.

    Returns:
        filtered_points (M×4), pred_instance_ids (M,), orig_indices (M,)
        or (None, None, None) on failure.
    """
    from ecomodel_lite import EcomodelLite

    kwargs = {"segmenter_type": segmenter_type, "results_folder": results_folder}
    if segmenter_type == "treelearn":
        kwargs["treelearn_config_path"] = treelearn_config_path
        kwargs["treelearn_use_gpu"] = treelearn_use_gpu

    model = EcomodelLite(**kwargs)

    pts = np.hstack([
        points_xyzi,
        np.arange(len(points_xyzi), dtype=float).reshape(-1, 1),
    ])
    pts = model.normalize_point_cloud(pts)
    pts = model.remove_ground(pts)
    if pts is None or len(pts) < 50:
        return None, None, None

    pts = model.filter_intensity(pts, model.intensity_threshold)
    if pts is None or len(pts) < 50:
        return None, None, None

    work = pts[:, :4]
    orig_indices = pts[:, 4].astype(int)

    if run_leaf_removal:
        wood = model.remove_leaves_rgi(work)
        if wood is None or len(wood) < 50:
            return None, None, None
        # RGI drops points — recover orig indices via nearest-neighbor on xyz
        from scipy.spatial import cKDTree
        tree = cKDTree(work[:, :3])
        _, nn = tree.query(wood[:, :3], k=1)
        work = wood
        orig_indices = orig_indices[nn]

    out_dir = os.path.join(results_folder, "instance_seg_tmp")
    os.makedirs(out_dir, exist_ok=True)
    work_out, labels = model.perform_instance_segmentation(work, output_dir=out_dir)
    if work_out is None or labels is None:
        return None, None, None

    # Segmenters may drop/reorder points; remap by NN if shapes diverge
    if len(labels) != len(orig_indices):
        from scipy.spatial import cKDTree
        tree = cKDTree(work[:, :3])
        _, nn = tree.query(work_out[:, :3], k=1)
        orig_indices = orig_indices[nn]
        work = work_out
    else:
        work = work_out

    return work, labels.astype(np.int32), orig_indices


def evaluate_tile(
    tile_prefix: str,
    algorithms: List[str],
    iou_thresh: float,
    treelearn_config_path: str,
    treelearn_use_gpu: bool,
    run_leaf_removal: bool,
    results_folder: str,
) -> List[Dict]:
    laz_path = tile_prefix + "_scan.laz"
    gt_path = tile_prefix + "_instances.npy"
    meta_path = tile_prefix + "_meta.json"

    tile_name = os.path.basename(tile_prefix)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    las = laspy.read(laz_path)
    intensity = las.intensity / 65535.0
    points = np.vstack([las.x, las.y, las.z, intensity]).T
    gt_full = np.load(gt_path).astype(np.int32)
    if len(gt_full) != len(points):
        return [{
            "tile": tile_name,
            "algorithm": "n/a",
            "error": f"GT length {len(gt_full)} != points {len(points)}",
            "density_class": meta.get("density_class"),
            "composition": meta.get("composition"),
        }]

    rows = []
    for alg in algorithms:
        t0 = time.time()
        try:
            filtered, pred_ids, orig_idx = run_segmenter_on_tile(
                points,
                segmenter_type=alg,
                treelearn_config_path=treelearn_config_path,
                treelearn_use_gpu=treelearn_use_gpu,
                run_leaf_removal=run_leaf_removal,
                results_folder=results_folder,
            )
        except Exception as e:
            rows.append({
                "tile": tile_name,
                "algorithm": alg,
                "error": str(e),
                "duration_sec": time.time() - t0,
                "density_class": meta.get("density_class"),
                "composition": meta.get("composition"),
                "num_trees_meta": meta.get("num_trees"),
            })
            continue

        if filtered is None or pred_ids is None:
            rows.append({
                "tile": tile_name,
                "algorithm": alg,
                "error": "segmentation returned no points",
                "duration_sec": time.time() - t0,
                "density_class": meta.get("density_class"),
                "composition": meta.get("composition"),
                "num_trees_meta": meta.get("num_trees"),
            })
            continue

        # Align GT to the evaluated subset
        gt_sub = gt_full[orig_idx]
        # Ignore ground/clutter in both for IoU matrix construction
        eval_mask = gt_sub >= 0
        # Pred clutter/noise stays as-is; unmatched pred on ground-only points still count as FP
        # Evaluate IoU only over points where GT is a tree OR pred claims a tree
        iou, gt_ids, pred_ids_u = pairwise_iou_matrix(gt_sub, pred_ids)
        metrics = match_instances(iou, iou_thresh=iou_thresh)

        row = {
            "tile": tile_name,
            "algorithm": alg,
            "density_class": meta.get("density_class"),
            "composition": meta.get("composition"),
            "num_trees_meta": meta.get("num_trees"),
            "num_eval_points": int(len(pred_ids)),
            "num_gt_tree_points": int(np.sum(eval_mask)),
            "iou_thresh": iou_thresh,
            "duration_sec": time.time() - t0,
            "error": "",
            **metrics,
        }
        rows.append(row)
        print(
            f"  [{alg}] {tile_name}: F1={metrics['f1']:.3f} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"PQ={metrics['pq']:.3f} (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})"
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Stratum-wise means for successful rows."""
    ok = df[df["error"].fillna("") == ""].copy()
    if ok.empty:
        return ok
    group_cols = ["algorithm", "density_class", "composition"]
    metric_cols = ["precision", "recall", "f1", "pq", "sq", "rq", "mean_matched_iou", "tp", "fp", "fn"]
    present = [c for c in metric_cols if c in ok.columns]
    return ok.groupby(group_cols, dropna=False)[present].mean().reset_index()


def main():
    parser = argparse.ArgumentParser(description="Benchmark multi-tree instance segmentation")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(_SP_DIR, "testdataset", "instance"),
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join(_SP_DIR, "output", "benchmark_instance.csv"),
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="scanline",
        help="Comma-separated: scanline,treelearn",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--treelearn_config", type=str, default="")
    parser.add_argument("--treelearn_use_gpu", action="store_true", default=True)
    parser.add_argument("--no_treelearn_gpu", action="store_true")
    parser.add_argument(
        "--leaf_removal",
        action="store_true",
        help="Run RGI wood/leaf separation before instance segmentation (off by default)",
    )
    parser.add_argument("--sample_n", type=int, default=None, help="Random subset of tiles")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    if "treelearn" in algorithms and not args.treelearn_config:
        print("WARNING: treelearn requested but --treelearn_config not set; skipping treelearn")
        algorithms = [a for a in algorithms if a != "treelearn"]
    if not algorithms:
        print("No algorithms to run.")
        sys.exit(1)

    tiles = discover_tiles(args.dataset_dir)
    if not tiles:
        print(f"No tiles with *_scan.laz + *_instances.npy in {args.dataset_dir}")
        sys.exit(1)

    if args.sample_n is not None and args.sample_n < len(tiles):
        rng = np.random.default_rng(args.seed)
        tiles = list(rng.choice(tiles, size=args.sample_n, replace=False))

    print(f"Evaluating {len(tiles)} tiles with algorithms={algorithms}")
    results_folder = os.path.join(_SP_DIR, "output", "benchmark_instance_workdir")
    os.makedirs(results_folder, exist_ok=True)

    all_rows: List[Dict] = []
    for i, prefix in enumerate(tiles):
        print(f"\n[{i+1}/{len(tiles)}] {os.path.basename(prefix)}")
        all_rows.extend(
            evaluate_tile(
                prefix,
                algorithms=algorithms,
                iou_thresh=args.iou_thresh,
                treelearn_config_path=args.treelearn_config,
                treelearn_use_gpu=not args.no_treelearn_gpu,
                run_leaf_removal=args.leaf_removal,
                results_folder=results_folder,
            )
        )

    df = pd.DataFrame(all_rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)
    print(f"\nWrote {out_csv}")

    summary = summarize(df)
    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    summary.to_csv(str(summary_path), index=False)
    print(f"Wrote {summary_path}")
    if not summary.empty:
        print("\nStratum summary (mean):")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
