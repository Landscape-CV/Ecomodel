"""
Benchmark tree instance segmentation on multi-tree synthetic tiles.

Expects tiles produced by ``generate_instance_benchmark.py``:
  {tile}_scan.laz, {tile}_instances.npy, {tile}_meta.json

Algorithms: scanline, treelearn, pointsam, snap
  (+ pointsam_oracle / snap_oracle when --prompt_mode oracle|both)

Usage:
  python scripts/benchmark_instance_segmentation.py \\
      --dataset_dir testdataset/instance \\
      --algorithms scanline,treelearn,pointsam,snap \\
      --prompt_mode both --stratify 16 \\
      --out_csv output/benchmark_instance_4method.csv
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

_SP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_SP_DIR)
for p in (_ROOT, _SP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.instance_metrics import match_instances, pairwise_iou_matrix
from Utils.promptable_instance import oracle_stem_prompts


def _instance_colors(labels: np.ndarray) -> np.ndarray:
    """Deterministic RGB in [0,1] for instance IDs (-1 → gray)."""
    rgb = np.full((len(labels), 3), 0.55, dtype=np.float64)
    ids = labels.astype(np.int64)
    pos = ids >= 0
    if not np.any(pos):
        return rgb
    # Stable hash-ish coloring per instance id
    u = ids[pos]
    h = (u * 2654435761) % (2**32)
    rgb[pos, 0] = ((h % 256) / 255.0) * 0.85 + 0.1
    rgb[pos, 1] = (((h // 256) % 256) / 255.0) * 0.85 + 0.1
    rgb[pos, 2] = (((h // 65536) % 256) / 255.0) * 0.85 + 0.1
    return rgb


def save_prediction_cloud(
    out_dir: str,
    tile_name: str,
    algorithm: str,
    xyz: np.ndarray,
    intensity: np.ndarray,
    pred_ids: np.ndarray,
    gt_ids: np.ndarray,
) -> Dict[str, str]:
    """
    Write viewable prediction artifacts for one tile×algorithm:
      - LAZ with extra dims instance_pred / instance_gt
      - colored PLY (pred) and PLY (gt) for quick inspection
    """
    tile_dir = Path(out_dir) / tile_name
    tile_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tile_name}_{algorithm}"
    paths: Dict[str, str] = {}

    xyz = np.asarray(xyz, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_ids, dtype=np.int32).reshape(-1)
    gt = np.asarray(gt_ids, dtype=np.int32).reshape(-1)
    n = len(xyz)
    if not (len(inten) == n and len(pred) == n and len(gt) == n):
        raise ValueError(
            f"length mismatch xyz={n} inten={len(inten)} pred={len(pred)} gt={len(gt)}"
        )

    # Intensity stored as uint16-ish scale for LAZ viewers
    inten_u16 = np.clip(np.round(inten * 65535.0), 0, 65535).astype(np.uint16)

    laz_path = tile_dir / f"{stem}_pred.laz"
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance_gt", type=np.int32))
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = inten_u16
    # Classification: 1=unassigned, 2+=tree instances (clamped)
    cls = np.ones(n, dtype=np.uint8)
    cls[pred >= 0] = np.clip(pred[pred >= 0] + 2, 2, 31).astype(np.uint8)
    las.classification = cls
    las.instance_pred = pred
    las.instance_gt = gt
    las.write(str(laz_path))
    paths["laz"] = str(laz_path)

    # Colored PLYs (binary) — open in CloudCompare / MeshLab.
    # Cap PLY size for viewers; full-resolution labels stay in LAZ/npy.
    max_ply = 500_000
    if n > max_ply:
        rng = np.random.default_rng(0)
        ply_idx = rng.choice(n, size=max_ply, replace=False)
        ply_idx.sort()
    else:
        ply_idx = np.arange(n)

    for kind, labels in (("pred", pred), ("gt", gt)):
        ply_path = tile_dir / f"{stem}_{kind}.ply"
        xyz_p = xyz[ply_idx]
        colors = (_instance_colors(labels[ply_idx]) * 255.0).astype(np.uint8)
        m = len(ply_idx)
        with open(ply_path, "wb") as f:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {m}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode("ascii")
            f.write(header)
            # struct: 3 float32 + 3 uint8
            verts = np.empty(m, dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ])
            verts["x"] = xyz_p[:, 0]
            verts["y"] = xyz_p[:, 1]
            verts["z"] = xyz_p[:, 2]
            verts["r"] = colors[:, 0]
            verts["g"] = colors[:, 1]
            verts["b"] = colors[:, 2]
            f.write(verts.tobytes())
        paths[kind] = str(ply_path)

    npy_path = tile_dir / f"{stem}_pred_ids.npy"
    np.save(str(npy_path), pred)
    paths["pred_npy"] = str(npy_path)
    return paths


def discover_tiles(dataset_dir: str) -> List[str]:
    prefixes = []
    for laz in glob.glob(os.path.join(dataset_dir, "*_scan.laz")):
        base = laz[: -len("_scan.laz")]
        if os.path.exists(base + "_instances.npy"):
            prefixes.append(base)
    return sorted(prefixes)


def stratify_tiles(tiles: List[str], target_n: int, seed: int) -> List[str]:
    """
    Pick up to target_n tiles with balanced density × composition coverage.
    Default design for 16: 1 mixed + 1 mono per density (4 densities).
    """
    by_key: Dict[Tuple[str, str], List[str]] = {}
    for prefix in tiles:
        meta_path = prefix + "_meta.json"
        density, composition = "unknown", "unknown"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            density = meta.get("density_class", "unknown")
            composition = meta.get("composition", "unknown")
        by_key.setdefault((density, composition), []).append(prefix)

    rng = np.random.default_rng(seed)
    # Prefer one per (density, composition) cell first
    selected: List[str] = []
    cells = sorted(by_key.keys())
    for cell in cells:
        pool = by_key[cell]
        chosen = pool[int(rng.integers(0, len(pool)))]
        selected.append(chosen)
        if len(selected) >= target_n:
            return selected

    remaining = [t for t in tiles if t not in selected]
    if remaining and len(selected) < target_n:
        need = min(target_n - len(selected), len(remaining))
        extra = list(rng.choice(remaining, size=need, replace=False))
        selected.extend(extra)
    return selected


def expand_algorithms(base_algs: List[str], prompt_mode: str) -> List[str]:
    """Map base names + prompt_mode into concrete algorithm keys."""
    out = []
    for a in base_algs:
        if a in ("pointsam", "snap"):
            if prompt_mode in ("auto", "both"):
                out.append(a)
            if prompt_mode in ("oracle", "both"):
                out.append(f"{a}_oracle")
        else:
            out.append(a)
    # de-dupe preserving order
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def run_segmenter_on_tile(
    points_xyzi: np.ndarray,
    segmenter_type: str,
    *,
    gt_full: Optional[np.ndarray] = None,
    treelearn_config_path: str = "",
    treelearn_use_gpu: bool = True,
    pointsam_ckpt: str = "",
    pointsam_config: str = "large",
    pointsam_use_gpu: bool = True,
    snap_ckpt: str = "",
    snap_domain: str = "Outdoor",
    snap_grid_size: float = 0.05,
    snap_use_gpu: bool = True,
    run_leaf_removal: bool = False,
    results_folder: str = "results",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    from ecomodel_lite import EcomodelLite

    use_oracle = segmenter_type.endswith("_oracle")
    base_type = segmenter_type.replace("_oracle", "")
    kwargs = {"segmenter_type": base_type, "results_folder": results_folder}
    if base_type == "treelearn":
        kwargs["treelearn_config_path"] = treelearn_config_path
        kwargs["treelearn_use_gpu"] = treelearn_use_gpu
    elif base_type == "pointsam":
        kwargs["pointsam_ckpt"] = pointsam_ckpt
        kwargs["pointsam_config"] = pointsam_config
        kwargs["pointsam_use_gpu"] = pointsam_use_gpu
    elif base_type == "snap":
        kwargs["snap_ckpt"] = snap_ckpt
        kwargs["snap_domain"] = snap_domain
        kwargs["snap_grid_size"] = snap_grid_size
        kwargs["snap_use_gpu"] = snap_use_gpu

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
        from scipy.spatial import cKDTree
        tree = cKDTree(work[:, :3])
        _, nn = tree.query(wood[:, :3], k=1)
        work = wood
        orig_indices = orig_indices[nn]

    prompts = None
    if use_oracle and base_type in ("pointsam", "snap"):
        if gt_full is None:
            raise ValueError("gt_full required for oracle prompt mode")
        gt_sub = gt_full[orig_indices]
        prompts = oracle_stem_prompts(work[:, :3], gt_sub)

    out_dir = os.path.join(results_folder, f"instance_seg_tmp_{base_type}")
    os.makedirs(out_dir, exist_ok=True)
    work_out, labels = model.perform_instance_segmentation(
        work, output_dir=out_dir, prompts=prompts
    )
    if work_out is None or labels is None:
        return None, None, None

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
    run_leaf_removal: bool,
    results_folder: str,
    ckpt_kwargs: Dict,
    save_predictions: bool = False,
    pred_dir: Optional[str] = None,
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
        use_oracle = alg.endswith("_oracle")

        try:
            filtered, pred_ids, orig_idx = run_segmenter_on_tile(
                points,
                segmenter_type=alg,
                gt_full=gt_full,
                run_leaf_removal=run_leaf_removal,
                results_folder=results_folder,
                **ckpt_kwargs,
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
                "prompt_mode": "oracle" if use_oracle else "auto",
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
                "prompt_mode": "oracle" if use_oracle else "auto",
            })
            continue

        gt_sub = gt_full[orig_idx]
        eval_mask = gt_sub >= 0
        iou, _, _ = pairwise_iou_matrix(gt_sub, pred_ids)
        metrics = match_instances(iou, iou_thresh=iou_thresh)

        pred_paths = {}
        if save_predictions and pred_dir:
            try:
                inten_sub = filtered[:, 3] if filtered.shape[1] >= 4 else intensity[orig_idx]
                pred_paths = save_prediction_cloud(
                    pred_dir,
                    tile_name,
                    alg,
                    filtered[:, :3],
                    inten_sub,
                    pred_ids,
                    gt_sub,
                )
                print(f"    saved → {pred_paths.get('laz', pred_paths.get('pred', ''))}")
            except Exception as exc:
                print(f"    WARNING: failed to save predictions ({exc})")

        row = {
            "tile": tile_name,
            "algorithm": alg,
            "prompt_mode": "oracle" if use_oracle else "auto",
            "density_class": meta.get("density_class"),
            "composition": meta.get("composition"),
            "num_trees_meta": meta.get("num_trees"),
            "num_eval_points": int(len(pred_ids)),
            "num_gt_tree_points": int(np.sum(eval_mask)),
            "iou_thresh": iou_thresh,
            "duration_sec": time.time() - t0,
            "error": "",
            "pred_laz": pred_paths.get("laz", ""),
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
    ok = df[df["error"].fillna("") == ""].copy()
    if ok.empty:
        return ok
    group_cols = ["algorithm", "prompt_mode", "density_class", "composition"]
    group_cols = [c for c in group_cols if c in ok.columns]
    metric_cols = ["precision", "recall", "f1", "pq", "sq", "rq", "mean_matched_iou", "tp", "fp", "fn"]
    present = [c for c in metric_cols if c in ok.columns]
    return ok.groupby(group_cols, dropna=False)[present].mean().reset_index()


def summary_to_markdown(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No successful rows._\n"
    cols = [c for c in ["algorithm", "prompt_mode", "density_class", "composition",
                        "precision", "recall", "f1", "pq"] if c in summary.columns]
    sub = summary[cols].copy()
    for c in ["precision", "recall", "f1", "pq"]:
        if c in sub.columns:
            sub[c] = sub[c].map(lambda x: f"{float(x):.3f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Benchmark multi-tree instance segmentation")
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(_SP_DIR, "testdataset", "instance"))
    parser.add_argument("--out_csv", type=str,
                        default=os.path.join(_SP_DIR, "output", "benchmark_instance.csv"))
    parser.add_argument(
        "--algorithms",
        type=str,
        default="scanline,pointsam,snap",
        help="Comma-separated: scanline,treelearn,pointsam,snap",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="both",
        choices=["auto", "oracle", "both"],
        help="For pointsam/snap: automatic seeds, GT-oracle clicks, or both",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--treelearn_config", type=str, default="")
    parser.add_argument("--no_treelearn_gpu", action="store_true")
    parser.add_argument("--pointsam_ckpt", type=str,
                        default=os.path.join(_ROOT, "thirdparty", "checkpoints", "point_sam", "model.safetensors"))
    parser.add_argument("--pointsam_config", type=str, default="large")
    parser.add_argument("--no_pointsam_gpu", action="store_true")
    parser.add_argument("--snap_ckpt", type=str,
                        default=os.path.join(_ROOT, "thirdparty", "checkpoints", "snap", "SNAP_C.pth"))
    parser.add_argument("--snap_domain", type=str, default="Outdoor")
    parser.add_argument("--snap_grid_size", type=float, default=0.05)
    parser.add_argument("--no_snap_gpu", action="store_true")
    parser.add_argument("--leaf_removal", action="store_true")
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--stratify", type=int, default=None,
                        help="Stratified sample size (e.g. 16 = 1 mixed+1 mono per density)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Write per-tile LAZ/PLY predictions under --pred_dir",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=os.path.join(_SP_DIR, "output", "benchmark_instance_predictions"),
        help="Directory for saved prediction clouds (used with --save_predictions)",
    )
    args = parser.parse_args()

    base_algs = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    if "treelearn" in base_algs and not args.treelearn_config:
        print("WARNING: treelearn requested but --treelearn_config not set; skipping treelearn")
        base_algs = [a for a in base_algs if a != "treelearn"]
    if "pointsam" in base_algs and not os.path.isfile(args.pointsam_ckpt):
        print(f"WARNING: pointsam ckpt missing ({args.pointsam_ckpt}); skipping pointsam")
        base_algs = [a for a in base_algs if a != "pointsam"]
    elif "pointsam" in base_algs:
        try:
            import sys as _sys
            _ps = os.path.join(_ROOT, "thirdparty", "Point-SAM")
            if _ps not in _sys.path:
                _sys.path.insert(0, _ps)
            import torkit3d  # noqa: F401
            from pc_sam.model.pc_sam import PointCloudSAM  # noqa: F401
        except Exception as exc:
            print(f"WARNING: pointsam import failed ({exc}); skipping pointsam")
            base_algs = [a for a in base_algs if a != "pointsam"]
    if "snap" in base_algs and not os.path.isfile(args.snap_ckpt):
        print(f"WARNING: snap ckpt missing ({args.snap_ckpt}); skipping snap")
        base_algs = [a for a in base_algs if a != "snap"]
    elif "snap" in base_algs:
        try:
            import sys as _sys
            _sn = os.path.join(_ROOT, "thirdparty", "SNAP")
            if _sn not in _sys.path:
                _sys.path.insert(0, _sn)
            from src.snap import SNAP  # noqa: F401
        except Exception as exc:
            print(f"WARNING: snap import failed ({exc}); skipping snap")
            base_algs = [a for a in base_algs if a != "snap"]

    algorithms = expand_algorithms(base_algs, args.prompt_mode)
    if not algorithms:
        print("No algorithms to run (check checkpoints / configs).")
        sys.exit(1)

    tiles = discover_tiles(args.dataset_dir)
    if not tiles:
        print(f"No tiles with *_scan.laz + *_instances.npy in {args.dataset_dir}")
        sys.exit(1)

    if args.stratify is not None:
        tiles = stratify_tiles(tiles, args.stratify, args.seed)
    elif args.sample_n is not None and args.sample_n < len(tiles):
        rng = np.random.default_rng(args.seed)
        tiles = list(rng.choice(tiles, size=args.sample_n, replace=False))

    print(f"Evaluating {len(tiles)} tiles with algorithms={algorithms}")
    results_folder = os.path.join(_SP_DIR, "output", "benchmark_instance_workdir")
    os.makedirs(results_folder, exist_ok=True)
    if args.save_predictions:
        os.makedirs(args.pred_dir, exist_ok=True)
        print(f"Saving prediction clouds to {args.pred_dir}")

    ckpt_kwargs = {
        "treelearn_config_path": args.treelearn_config,
        "treelearn_use_gpu": not args.no_treelearn_gpu,
        "pointsam_ckpt": args.pointsam_ckpt,
        "pointsam_config": args.pointsam_config,
        "pointsam_use_gpu": not args.no_pointsam_gpu,
        "snap_ckpt": args.snap_ckpt,
        "snap_domain": args.snap_domain,
        "snap_grid_size": args.snap_grid_size,
        "snap_use_gpu": not args.no_snap_gpu,
    }

    all_rows: List[Dict] = []
    for i, prefix in enumerate(tiles):
        print(f"\n[{i+1}/{len(tiles)}] {os.path.basename(prefix)}")
        all_rows.extend(
            evaluate_tile(
                prefix,
                algorithms=algorithms,
                iou_thresh=args.iou_thresh,
                run_leaf_removal=args.leaf_removal,
                results_folder=results_folder,
                ckpt_kwargs=ckpt_kwargs,
                save_predictions=args.save_predictions,
                pred_dir=args.pred_dir if args.save_predictions else None,
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

    md_path = out_csv.with_name(out_csv.stem + "_summary.md")
    md_path.write_text(summary_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {md_path}")

    if not summary.empty:
        print("\nStratum summary (mean):")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
