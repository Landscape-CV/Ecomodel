"""Benchmark QSM reconstruction under leaf-on, separated, and oracle inputs.

The default execution is deliberately sequential and resource constrained.  Synthetic
tiles contain one known tree, so instance segmentation is bypassed to isolate the
effect of foliage on each QSM backend.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import laspy
import numpy as np
import pandas as pd
import psutil
from scipy.spatial import cKDTree

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecomodel_lite import EcomodelLite
from gui.smartqsm_runner import run_smartqsm_on_segments


RESULT_COLUMNS = [
    "Algorithm", "Condition", "Species", "File", "Status", "Error",
    "InputPoints", "PreprocessTime_s", "ExecTime_s", "PeakRSS_GB",
    "Seed", "PreprocessVoxelSize_m", "VoxelSize_m", "MetricVoxelSize_m",
    "DistanceTolerance_m", "Config",
    "CylinderCount", "Whole_Precision", "Whole_Recall", "Whole_F1",
    "Whole_IoU", "Whole_VolRatio", "Trunk_Precision", "Trunk_Recall",
    "Trunk_F1", "Trunk_IoU", "Trunk_VolRatio", "Branch_Precision",
    "Branch_Recall", "Branch_F1", "Branch_IoU", "Branch_VolRatio",
]


class MockConfig:
    def __init__(self, sq_dir, sq_py, sq_cfg, timeout=None, memory_limit_gb=None):
        self.smartqsm_dir = sq_dir
        self.smartqsm_python = sq_py
        self.smartqsm_config = sq_cfg
        self.smartqsm_timeout = timeout
        self.smartqsm_memory_limit_gb = memory_limit_gb


def sample_cylinders(cylinders, num_points=100000, rng=None):
    """Deterministically sample cylinder lateral surfaces when an RNG is supplied."""
    cylinders = np.asarray(cylinders, dtype=float)
    if cylinders.size == 0:
        return np.zeros((0, 3), dtype=float)
    cylinders = np.atleast_2d(cylinders)
    rng = rng or np.random.default_rng(0)
    areas = 2 * np.pi * cylinders[:, 3] * cylinders[:, 7]
    areas[areas <= 0] = 1e-6
    counts = rng.multinomial(num_points, areas / areas.sum())
    blocks = []
    for cyl, count in zip(cylinders, counts):
        if count == 0:
            continue
        start, radius, axis, length = cyl[:3], cyl[3], cyl[4:7], cyl[7]
        norm = np.linalg.norm(axis)
        if norm == 0 or length <= 0 or radius < 0:
            continue
        axis = axis / norm
        z = rng.uniform(0, length, count)
        theta = rng.uniform(0, 2 * np.pi, count)
        basis = np.array([0.0, 1.0, 0.0]) if abs(axis[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        basis -= np.dot(basis, axis) * axis
        basis /= np.linalg.norm(basis)
        basis2 = np.cross(axis, basis)
        blocks.append(
            start + np.outer(z, axis)
            + radius * (np.outer(np.cos(theta), basis) + np.outer(np.sin(theta), basis2))
        )
    return np.vstack(blocks) if blocks else np.zeros((0, 3), dtype=float)


def calculate_metrics(gt_pts, pred_pts, voxel_size=0.1):
    """Occupied-voxel precision, recall, F1 and IoU (legacy helper API)."""
    if len(gt_pts) == 0 and len(pred_pts) == 0:
        return 1.0, 1.0, 1.0, 1.0
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 0.0, 0.0, 0.0, 0.0
    gt_voxels = np.unique(np.floor(gt_pts / voxel_size).astype(np.int64), axis=0)
    pred_voxels = np.unique(np.floor(pred_pts / voxel_size).astype(np.int64), axis=0)
    gt_set = set(map(tuple, gt_voxels))
    pred_set = set(map(tuple, pred_voxels))
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return precision, recall, f1, iou


def distance_metrics(gt_pts, pred_pts, tolerance):
    """Symmetric surface completeness/correctness at a distance tolerance."""
    if len(gt_pts) == 0 and len(pred_pts) == 0:
        return 1.0, 1.0, 1.0, 1.0
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 0.0, 0.0, 0.0, 0.0
    precision = float(np.mean(cKDTree(gt_pts).query(pred_pts, workers=1)[0] <= tolerance))
    recall = float(np.mean(cKDTree(pred_pts).query(gt_pts, workers=1)[0] <= tolerance))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = f1 / (2.0 - f1) if f1 < 2.0 else 1.0
    return precision, recall, f1, iou


def get_cylinder_volume(cylinders):
    cylinders = np.asarray(cylinders)
    return float(np.sum(np.pi * cylinders[:, 3] ** 2 * cylinders[:, 7])) if cylinders.size else 0.0


def _treeqsm_worker(result_queue, points, model_kwargs):
    try:
        model = EcomodelLite(**model_kwargs)
        cylinders, _ = model.get_cylinders_for_tree(
            points, tree_instance=0, compute_metrics=False
        )
        result_queue.put({"status": "ok", "cylinders": cylinders, "error": ""})
    except BaseException:
        result_queue.put({
            "status": "error",
            "cylinders": np.empty((0, 9)),
            "error": traceback.format_exc(),
        })


def _sleep_worker(result_queue, points, model_kwargs):
    """Test-only probe used to verify timeout cleanup under spawn."""
    time.sleep(model_kwargs.get("sleep", 1.0))
    result_queue.put({"status": "ok", "cylinders": np.empty((0, 9)), "error": ""})


def _rss_gb(process):
    try:
        proc = psutil.Process(process.pid)
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss / (1024 ** 3)
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
        return 0.0


def run_treeqsm(points, model_kwargs, timeout, memory_limit_gb, worker_target=None):
    """Run production-lite TreeQSM in a killable worker without queue deadlock."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=worker_target or _treeqsm_worker,
        args=(result_queue, points, model_kwargs),
    )
    started = time.time()
    peak_rss = 0.0
    process.start()
    result = None
    status = "error"
    error = "TreeQSM worker exited without a result"
    try:
        while process.is_alive():
            peak_rss = max(peak_rss, _rss_gb(process))
            try:
                result = result_queue.get_nowait()
                break
            except queue.Empty:
                pass
            elapsed = time.time() - started
            if memory_limit_gb and peak_rss > memory_limit_gb:
                status, error = "memory_limit", f"Worker exceeded {memory_limit_gb:.2f} GB"
                break
            if timeout and elapsed > timeout:
                status, error = "timeout", f"TreeQSM exceeded {timeout} seconds"
                break
            time.sleep(0.2)
        if result is None:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                pass
        if result is not None:
            status = result["status"]
            error = result.get("error", "")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
        process.join(timeout=2)
        peak_rss = max(peak_rss, _rss_gb(process))
        result_queue.close()
        result_queue.join_thread()
    cylinders = result["cylinders"] if result is not None else np.empty((0, 9))
    return cylinders, status, error, time.time() - started, peak_rss


def _downsample(points, voxel_size, max_points, rng):
    if voxel_size and voxel_size > 0 and len(points):
        keys = np.floor(points[:, :3] / voxel_size).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(indices)]
    if max_points and len(points) > max_points:
        points = points[np.sort(rng.choice(len(points), max_points, replace=False))]
    return points


def prepare_conditions(tile_path, labels, args, rng):
    """Mirror production preprocessing while preserving original point indices."""
    las = laspy.read(tile_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(float)
    intensity = np.asarray(las.intensity, dtype=float) if hasattr(las, "intensity") else np.ones(len(xyz))
    if intensity.size and np.max(intensity) > 1:
        intensity /= 65535.0
    if len(labels) != len(xyz):
        raise ValueError(f"Label count {len(labels)} does not match point count {len(xyz)}")

    indexed = np.column_stack((xyz, intensity, np.arange(len(xyz), dtype=float)))
    prep_dir = os.path.join(args.out_dir, "preprocess")
    model = EcomodelLite(results_folder=prep_dir, intensity_threshold=args.intensity_threshold)
    started = time.time()
    indexed = model.normalize_point_cloud(indexed)
    indexed = model.remove_ground(indexed)
    if indexed is None or len(indexed) < 100:
        raise ValueError("Too few points after CSF ground removal")
    indexed = model.filter_intensity(indexed, model.intensity_threshold)
    if len(indexed) < 100:
        raise ValueError("Too few points after intensity filtering")
    indexed = _downsample(indexed, args.preprocess_voxel_size, None, rng)

    conditions = {}
    if "leaf_on" in args.conditions:
        conditions["leaf_on"] = indexed
    if "rgi" in args.conditions:
        wood_mask, _ = model.classify_wood_leaf_on_array(indexed[:, :4], model._rgi_params)
        if wood_mask is not None and np.any(wood_mask):
            conditions["rgi"] = indexed[wood_mask]
        else:
            conditions["rgi"] = None
    original_indices = indexed[:, 4].astype(np.int64)
    if "oracle_wood" in args.conditions:
        conditions["oracle_wood"] = indexed[labels[original_indices] == 1]

    if "gbseparation" in args.conditions:
        from GBSeparation.remove_leaves import LeafRemover
        gb_mask, _ = LeafRemover().process(indexed[:, :3], return_mask=True)
        conditions["gbseparation"] = indexed[gb_mask]

    for name, value in list(conditions.items()):
        if value is not None:
            conditions[name] = _downsample(value, args.qsm_voxel_size, args.max_points, rng)
    return conditions, model.mean.copy(), time.time() - started


def _metric_block(gt_cyls, pred_cyls, rng, args):
    gt_pts = sample_cylinders(gt_cyls, args.surface_samples, rng)
    pred_pts = sample_cylinders(pred_cyls, args.surface_samples, rng)
    precision, recall, f1, _ = distance_metrics(gt_pts, pred_pts, args.distance_tolerance)
    _, _, _, voxel_iou = calculate_metrics(gt_pts, pred_pts, args.metric_voxel_size)
    return (precision, recall, f1, voxel_iou,
            get_cylinder_volume(pred_cyls) / get_cylinder_volume(gt_cyls)
            if get_cylinder_volume(gt_cyls) > 0 else 0.0)


def evaluate_cylinders(gt_cyls, pred_cyls, seed, args):
    rng = np.random.default_rng(seed)
    whole = _metric_block(gt_cyls, pred_cyls, rng, args)
    gt_trunk = gt_cyls[gt_cyls[:, 3] >= args.trunk_radius_threshold]
    gt_branch = gt_cyls[gt_cyls[:, 3] < args.trunk_radius_threshold]
    pred_trunk = pred_cyls[pred_cyls[:, 3] >= args.trunk_radius_threshold]
    pred_branch = pred_cyls[pred_cyls[:, 3] < args.trunk_radius_threshold]
    trunk = _metric_block(gt_trunk, pred_trunk, rng, args)
    branch = _metric_block(gt_branch, pred_branch, rng, args)
    names = ("Precision", "Recall", "F1", "IoU", "VolRatio")
    metrics = {}
    for prefix, values in (("Whole", whole), ("Trunk", trunk), ("Branch", branch)):
        metrics.update({f"{prefix}_{name}": value for name, value in zip(names, values)})
    return metrics


def _world_cylinders(cylinders, mean):
    cylinders = np.asarray(cylinders, dtype=float)
    if cylinders.size == 0:
        return np.empty((0, 9))
    cylinders = np.atleast_2d(cylinders).copy()
    cylinders[:, :3] += mean[:3]
    return cylinders


def load_gt_cylinders(path):
    """Load legacy 9-column or current 10-column synthetic GT cylinders."""
    cylinders = np.atleast_2d(np.loadtxt(path)).astype(float)
    if cylinders.shape[1] not in (9, 10):
        raise ValueError(
            f"GT must contain 9 legacy columns or 10 current columns, got {cylinders.shape[1]}"
        )
    return cylinders


def _base_result(tile_path, condition, algorithm, config, input_count, prep_time, args, seed):
    filename = os.path.basename(tile_path)
    return {
        "Algorithm": algorithm,
        "Condition": condition,
        "Species": filename.split("_tile")[0],
        "File": filename,
        "Status": "error",
        "Error": "",
        "InputPoints": input_count,
        "PreprocessTime_s": prep_time,
        "ExecTime_s": 0.0,
        "PeakRSS_GB": 0.0,
        "Seed": seed,
        "PreprocessVoxelSize_m": args.preprocess_voxel_size,
        "VoxelSize_m": args.qsm_voxel_size,
        "MetricVoxelSize_m": args.metric_voxel_size,
        "DistanceTolerance_m": args.distance_tolerance,
        "Config": config,
        "CylinderCount": 0,
    }


def benchmark_tile(tile_path, args, completed=None):
    completed = completed or set()
    base_name = tile_path.removesuffix("_scan.laz")
    labels_path = base_name + "_labels.npy"
    gt_path = base_name + "_gt.txt"
    if not os.path.exists(labels_path) or not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing labels or GT for {tile_path}")
    labels = np.load(labels_path)
    gt_cyls = load_gt_cylinders(gt_path)

    tile_seed = args.seed + sum(os.path.basename(tile_path).encode("utf-8"))
    rng = np.random.default_rng(tile_seed)
    conditions, mean, prep_time = prepare_conditions(tile_path, labels, args, rng)
    rows = []

    for condition, point_data in conditions.items():
        if point_data is None or len(point_data) < 100:
            for algorithm in args.algorithms:
                row = _base_result(tile_path, condition, algorithm, "", 0, prep_time, args, tile_seed)
                row.update(Status="separation_failed", Error=f"{condition} produced fewer than 100 points")
                rows.append(row)
            continue

        points = point_data[:, :3]
        for algorithm in args.algorithms:
            config = ""
            row_seed = tile_seed + sum((condition + algorithm).encode("utf-8"))
            if algorithm == "TreeQSM":
                work_dir = os.path.join(args.out_dir, "treeqsm_temp", Path(base_name).name, condition)
                os.makedirs(work_dir, exist_ok=True)
                model_kwargs = {
                    "results_folder": work_dir,
                    "patch_diam1": args.patch_diam1,
                    "ball_rad1": args.ball_rad1,
                    "nmin1": args.nmin1,
                    "patch_diam2_min": args.patch_diam2_min,
                    "patch_diam2_max": args.patch_diam2_max,
                    "ball_rad2": args.ball_rad2,
                }
                config = json.dumps(model_kwargs, sort_keys=True)
                row = _base_result(tile_path, condition, algorithm, config, len(points), prep_time, args, row_seed)
                if _result_key(row) in completed:
                    continue
                cylinders, status, error, elapsed, peak = run_treeqsm(
                    points, model_kwargs, args.treeqsm_timeout, args.memory_limit_gb
                )
            elif algorithm == "SmartQSM":
                sq_cfg = args.sq_leafon_cfg if condition == "leaf_on" else args.sq_leafoff_cfg
                config = sq_cfg
                row = _base_result(tile_path, condition, algorithm, config, len(points), prep_time, args, row_seed)
                if _result_key(row) in completed:
                    continue
                temp_dir = os.path.join(args.out_dir, "smartqsm_temp", Path(base_name).name, condition)
                smart_config = MockConfig(
                    args.sq_dir, args.sq_py, sq_cfg, args.smartqsm_timeout, args.memory_limit_gb
                )
                logs = []
                started = time.time()
                cylinders = run_smartqsm_on_segments(
                    point_data[:, :4], np.zeros(len(points), dtype=np.int32),
                    temp_dir, smart_config, log=logs.append,
                )
                elapsed = time.time() - started
                peak = getattr(smart_config, "smartqsm_peak_rss_gb", 0.0)
                log_text = "".join(logs)
                if len(cylinders):
                    status, error = "ok", ""
                elif "timed out" in log_text:
                    status, error = "timeout", log_text[-4000:]
                elif "exceeded" in log_text:
                    status, error = "memory_limit", log_text[-4000:]
                else:
                    status, error = "error", log_text[-4000:]
            else:
                continue

            row.update(Status=status, Error=error, ExecTime_s=elapsed, PeakRSS_GB=peak,
                       CylinderCount=len(cylinders))
            if status == "ok" and len(cylinders):
                world_cylinders = _world_cylinders(cylinders, mean)
                row.update(evaluate_cylinders(gt_cyls, world_cylinders, row_seed, args))
            rows.append(row)

    if args.export_adqsm:
        export_dir = Path(args.out_dir) / "adqsm_manual" / Path(base_name).name
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for condition, point_data in conditions.items():
            if point_data is None:
                continue
            world_xyz = point_data[:, :3] + mean[:3]
            path = export_dir / f"{condition}.xyz"
            np.savetxt(path, world_xyz, fmt="%.6f")
            manifest.append({
                "file": str(path),
                "condition": condition,
                "points": len(world_xyz),
                "adqsm_defaults": {"Height_Segmentation": 0.5, "Cloud_Parameter": 0.003},
            })
        (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def _result_key(row):
    return (str(row["File"]), str(row["Condition"]), str(row["Algorithm"]), str(row["Config"]))


def _atomic_write(rows, csv_path):
    frame = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[RESULT_COLUMNS]
    temp_path = csv_path + ".tmp"
    frame.to_csv(temp_path, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temp_path, csv_path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(ROOT_DIR / "SyntheticPipeline" / "testdataset" / "single"))
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "SyntheticPipeline" / "output" / "qsm_benchmark"))
    parser.add_argument("--algorithms", default="TreeQSM,SmartQSM")
    parser.add_argument("--conditions", default="leaf_on,rgi,oracle_wood")
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--num-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-gbseparation", action="store_true")
    parser.add_argument("--export-adqsm", action="store_true")
    parser.add_argument("--intensity-threshold", type=float, default=0.0)
    parser.add_argument("--preprocess-voxel-size", type=float, default=0.0)
    parser.add_argument("--qsm-voxel-size", type=float, default=0.08)
    parser.add_argument("--max-points", type=int, default=400_000)
    parser.add_argument("--surface-samples", type=int, default=50_000)
    parser.add_argument("--distance-tolerance", type=float, default=0.05)
    parser.add_argument("--metric-voxel-size", type=float, default=0.10)
    parser.add_argument("--trunk-radius-threshold", type=float, default=0.05)
    parser.add_argument("--treeqsm-timeout", type=float, default=900)
    parser.add_argument("--smartqsm-timeout", type=float, default=900)
    parser.add_argument("--memory-limit-gb", type=float, default=15.0)
    parser.add_argument("--patch-diam1", type=float, default=0.10)
    parser.add_argument("--ball-rad1", type=float, default=0.12)
    parser.add_argument("--nmin1", type=int, default=15)
    parser.add_argument("--patch-diam2-min", type=float, default=0.05)
    parser.add_argument("--patch-diam2-max", type=float, default=0.12)
    parser.add_argument("--ball-rad2", type=float, default=0.14)
    parser.add_argument("--sq-dir", default=str(ROOT_DIR / "thirdparty" / "SmartQSM"))
    parser.add_argument("--sq-py", default=sys.executable)
    parser.add_argument(
        "--sq-leafon-cfg",
        default=str(ROOT_DIR / "thirdparty" / "SmartQSM" / "configs" / "spconv-contraction-LEAFON-GPU.yaml"),
    )
    parser.add_argument(
        "--sq-leafoff-cfg",
        default=str(ROOT_DIR / "thirdparty" / "SmartQSM" / "configs" / "space-colonization.yaml"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    args.out_dir = os.path.abspath(args.out_dir)
    args.algorithms = [value.strip() for value in args.algorithms.split(",") if value.strip()]
    args.conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    unknown = set(args.algorithms) - {"TreeQSM", "SmartQSM"}
    if unknown:
        raise ValueError(f"Unknown algorithms: {sorted(unknown)}")
    valid_conditions = {"leaf_on", "rgi", "oracle_wood", "gbseparation"}
    unknown_conditions = set(args.conditions) - valid_conditions
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {sorted(unknown_conditions)}")
    if args.include_gbseparation and "gbseparation" not in args.conditions:
        args.conditions.append("gbseparation")
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "benchmark_results_qsm.csv")

    existing_rows = []
    completed = set()
    if os.path.exists(csv_path):
        existing_rows = pd.read_csv(csv_path).to_dict("records")
        completed = {_result_key(row) for row in existing_rows}

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "*_scan.laz")))
    if args.sample_n and args.sample_n < len(files):
        files = random.Random(args.seed).sample(files, args.sample_n)
    print(f"Found {len(files)} QSM benchmark tiles; using {args.num_workers} worker(s).")

    rows = list(existing_rows)

    def process_tile(tile_path):
        try:
            return benchmark_tile(tile_path, args, completed)
        except Exception:
            error = traceback.format_exc()
            tile_rows = []
            for condition in args.conditions:
                for algorithm in args.algorithms:
                    cfg = (args.sq_leafon_cfg if condition == "leaf_on" else args.sq_leafoff_cfg) if algorithm == "SmartQSM" else ""
                    row = _base_result(tile_path, condition, algorithm, cfg, 0, 0.0, args, args.seed)
                    row.update(Status="preprocess_error", Error=error)
                    tile_rows.append(row)
            return tile_rows

    if args.num_workers == 1:
        completed_tiles = ((tile_path, process_tile(tile_path)) for tile_path in files)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
        future_to_tile = {executor.submit(process_tile, path): path for path in files}
        completed_tiles = (
            (future_to_tile[future], future.result())
            for future in concurrent.futures.as_completed(future_to_tile)
        )

    try:
        for index, (tile_path, tile_rows) in enumerate(completed_tiles, start=1):
            print(f"[{index}/{len(files)}] {os.path.basename(tile_path)}")
            for row in tile_rows:
                key = _result_key(row)
                if key in completed:
                    continue
                rows.append(row)
                completed.add(key)
                _atomic_write(rows, csv_path)
    finally:
        if args.num_workers > 1:
            executor.shutdown(wait=True)

    print(f"Benchmark completed: {csv_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
