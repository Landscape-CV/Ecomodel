import os
import sys
import glob
import time
import argparse
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import laspy
except ImportError:
    print("Please pip install laspy lazrs pandas")
    exit(1)

# Add the root of PyTLidar to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from SegmentRGI.SegmentRGI import classify_wood_leaf_point_cloud
from ecomodel_segmenters import SegmenterScanline
from GBSeparation.Graph_Path import array_to_graph, extract_path_info
from GBSeparation.LS_circle import getRootPt
from GBSeparation.ExtractInitWood import extract_init_wood
from GBSeparation.ExtractFinalWood import extract_final_wood
from gui.smartqsm_runner import run_smartqsm_on_segments

def compute_metrics(gt_labels, pred_labels):
    tp = np.sum((gt_labels == 1) & (pred_labels == 1))
    fp = np.sum((gt_labels == 0) & (pred_labels == 1))
    fn = np.sum((gt_labels == 1) & (pred_labels == 0))
    tn = np.sum((gt_labels == 0) & (pred_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return precision, recall, f1, iou


def run_segment_rgi(points, instance_ids, orig_indices, **kwargs):
    total_len = kwargs.get("total_len")
    pred_labels = np.full(total_len, -1, dtype=np.int32)
    unique_instances = np.unique(instance_ids)
    for inst_id in unique_instances:
        if inst_id < 0: continue
        inst_mask = (instance_ids == inst_id)
        inst_points = points[inst_mask]
        inst_orig_indices = orig_indices[inst_mask]
        
        if len(inst_points) < 50: continue
            
        try:
            wood_mask, leaf_mask = classify_wood_leaf_point_cloud(
                point_cloud=inst_points,
                noise_percentile=1
            )
            pred_labels[inst_orig_indices[wood_mask]] = 1
            pred_labels[inst_orig_indices[leaf_mask]] = 0
        except Exception as e:
            print(f"  [Warning] SegmentRGI failed on instance {inst_id}: {e}")
    return pred_labels


def run_gb_separation(points, instance_ids, orig_indices, **kwargs):
    total_len = kwargs.get("total_len")
    pred_labels = np.full(total_len, -1, dtype=np.int32)
    unique_instances = np.unique(instance_ids)
    for inst_id in unique_instances:
        if inst_id < 0: continue
        inst_mask = (instance_ids == inst_id)
        inst_points = points[inst_mask]
        inst_orig_indices = orig_indices[inst_mask]
        
        if len(inst_points) < 100: continue
            
        try:
            pcd_xyz = inst_points[:, :3]
            treeHeight = np.max(pcd_xyz[:, 2]) - np.min(pcd_xyz[:, 2])
            
            root, fit_seg = getRootPt(pcd_xyz, lower_h=0.0, upper_h=0.2)
            if root is None or len(root) == 0:
                root = pcd_xyz[np.argmin(pcd_xyz[:, 2])].reshape(1, 3)
                
            pcd_xyz_with_root = np.append(pcd_xyz, root, axis=0)
            root_id = pcd_xyz_with_root.shape[0] - 1
            
            G = array_to_graph(pcd_xyz_with_root, root_id, kpairs=3, knn=300, 
                               nbrs_threshold=treeHeight/30, nbrs_threshold_step=treeHeight/60)
            
            path_dis, path_list = extract_path_info(G, root_id, return_path=True)
            init_wood_ids = extract_init_wood(pcd_xyz_with_root, G, root_id, path_dis, path_list,
                                              split_interval=[0.1, 0.2, 0.3, 0.5, 1], max_angle=0.25*np.pi)
            
            final_wood_mask = extract_final_wood(pcd_xyz_with_root, root_id, path_dis, path_list, init_wood_ids, G)
            
            # Exclude the appended root point
            final_wood_mask = final_wood_mask[:-1]
            
            pred_labels[inst_orig_indices[final_wood_mask]] = 1
            pred_labels[inst_orig_indices[~final_wood_mask]] = 0
            
        except Exception as e:
            print(f"  [Warning] GBSeparation failed on instance {inst_id}: {e}")
            
    return pred_labels


class MockConfig:
    def __init__(self, sq_dir, sq_py, sq_cfg):
        self.smartqsm_dir = sq_dir
        self.smartqsm_python = sq_py
        self.smartqsm_config = sq_cfg


def run_smartqsm(points, instance_ids, orig_indices, **kwargs):
    total_len = kwargs.get("total_len")
    pred_labels = np.full(total_len, -1, dtype=np.int32)
    
    sq_dir = kwargs.get("sq_dir")
    sq_py = kwargs.get("sq_py")
    sq_cfg = kwargs.get("sq_cfg")
    temp_dir = kwargs.get("temp_dir")
    
    if not (sq_dir and sq_py and sq_cfg and temp_dir):
        print("  [Warning] SmartQSM dependencies missing. Skipping.")
        return pred_labels
        
    config = MockConfig(sq_dir, sq_py, sq_cfg)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        def log_func(msg): 
            print(msg, end="", flush=True)
            
        print(f"  [SmartQSM] Starting subprocess execution...", flush=True)
        cylinders = run_smartqsm_on_segments(points, instance_ids, temp_dir, config, log=log_func)
        if cylinders is None or len(cylinders) == 0:
            print("  [Warning] SmartQSM returned no cylinders.", flush=True)
            return pred_labels
            
        print(f"\n  [SmartQSM] Subprocess finished. Returned {len(cylinders)} cylinders. Starting distance calculation...", flush=True)
        # Point to cylinder distance
        xyz = points[:, :3]
        min_dist = np.full(len(xyz), np.inf)
        
        for i, cyl in enumerate(cylinders):
            if i % 1000 == 0:
                print(f"  [SmartQSM] Computed distance for {i}/{len(cylinders)} cylinders...", flush=True)
                
            start = cyl[0:3]
            radius = cyl[3]
            axis = cyl[4:7]
            length = cyl[7]
            
            # AABB filtering
            end = start + axis * length
            box_min = np.minimum(start, end) - radius - 0.05
            box_max = np.maximum(start, end) + radius + 0.05
            
            mask = (xyz[:, 0] >= box_min[0]) & (xyz[:, 0] <= box_max[0]) & \
                   (xyz[:, 1] >= box_min[1]) & (xyz[:, 1] <= box_max[1]) & \
                   (xyz[:, 2] >= box_min[2]) & (xyz[:, 2] <= box_max[2])
            
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
                
            sub_xyz = xyz[idx]
            v = sub_xyz - start
            t = np.sum(v * axis, axis=1)
            t = np.clip(t, 0, length)
            closest_points = start + np.outer(t, axis)
            dist = np.linalg.norm(sub_xyz - closest_points, axis=1)
            
            min_dist[idx] = np.minimum(min_dist[idx], dist - radius)
            
        is_wood = min_dist <= 0.05
        print(f"  [SmartQSM] Distance calculation finished. Wood points: {np.sum(is_wood)}/{len(xyz)}", flush=True)
        
        valid_mask = instance_ids >= 0
        valid_orig_indices = orig_indices[valid_mask]
        valid_is_wood = is_wood[valid_mask]
        
        pred_labels[valid_orig_indices[valid_is_wood]] = 1
        pred_labels[valid_orig_indices[~valid_is_wood]] = 0
        
    except Exception as e:
        print(f"  [Warning] SmartQSM evaluation failed: {e}", flush=True)
        
    return pred_labels


ALGORITHMS = {
    "SegmentRGI": run_segment_rgi,
    "GBSeparation": run_gb_separation,
    "SmartQSM": run_smartqsm
}

def process_tile(labels_file, log_dir, args):
    base_name = labels_file.replace("_labels.npy", "")
    laz_file = base_name + "_scan.laz"
    tile_name = os.path.basename(base_name)
    log_file = os.path.join(log_dir, f"{tile_name}.log")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    
    start_time = time.time()
    results = []
    
    try:
        with open(log_file, "w") as f:
            sys.stdout = f
            sys.stderr = f
            
            print(f"Starting processing for {tile_name}...")
            las = laspy.read(laz_file)
            intensity = las.intensity / 65535.0
            points = np.vstack([las.x, las.y, las.z, intensity]).T
            gt_labels = np.load(labels_file)
            
            print("  [Step 1] Running Tree Instance Segmentation (Scanline)...")
            segmenter = SegmenterScanline()
            filtered_points, instance_ids, orig_indices = segmenter.process_with_indices(points)
            
            if filtered_points is None or len(instance_ids) == 0:
                print("  [Warning] No trees found.")
                return []
            
            algs_to_run = args.algorithms.split(",") if args.algorithms else list(ALGORITHMS.keys())
            
            for alg_name in algs_to_run:
                if alg_name not in ALGORITHMS:
                    continue
                print(f"  [Step 2] Running {alg_name}...")
                alg_start = time.time()
                
                kwargs = {
                    "total_len": len(points),
                    "sq_dir": args.sq_dir,
                    "sq_py": args.sq_py,
                    "sq_cfg": args.sq_cfg,
                    "temp_dir": os.path.join(log_dir, f"temp_{tile_name}")
                }
                
                pred_labels = ALGORITHMS[alg_name](filtered_points, instance_ids, orig_indices, **kwargs)
                
                valid_mask = (pred_labels != -1)
                if not np.any(valid_mask):
                    print(f"  [Warning] No points valid for metric computation for {alg_name}.")
                    results.append({"tile": tile_name, "algorithm": alg_name, "error": "No valid points", "duration_sec": time.time() - alg_start})
                    continue
                
                valid_gt = gt_labels[valid_mask]
                valid_pred = pred_labels[valid_mask]
                
                p, r, f1, iou = compute_metrics(valid_gt, valid_pred)
                duration = time.time() - alg_start
                print(f"  -> {alg_name} | Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")
                
                results.append({
                    "tile": tile_name,
                    "algorithm": alg_name,
                    "precision": p,
                    "recall": r,
                    "f1_score": f1,
                    "iou": iou,
                    "duration_sec": duration
                })
                
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        results.append({"tile": tile_name, "error": str(e), "duration_sec": time.time() - start_time})
        with open(log_file, "a") as f:
            f.write(f"\nException occurred: {e}\n{tb}\n")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark Separation Algorithms.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(os.path.dirname(script_dir), "testdataset")
    default_out = os.path.join(os.path.dirname(script_dir), "output", "benchmark_results.csv")
    
    parser.add_argument("--dataset_dir", type=str, default=default_dataset)
    parser.add_argument("--out_csv", type=str, default=default_out)
    parser.add_argument("--num_workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--algorithms", type=str, default="SegmentRGI,GBSeparation,SmartQSM", help="Comma-separated algs")
    parser.add_argument("--sq_dir", type=str, default=os.path.join(root_dir, "thirdparty", "SmartQSM"))
    parser.add_argument("--sq_py", type=str, default=sys.executable)
    parser.add_argument("--sq_cfg", type=str, default=os.path.join(root_dir, "thirdparty", "SmartQSM", "configs", "spconv-contraction-LEAFON-GPU.yaml"))
    parser.add_argument("--plot", action="store_true", help="Generate performance comparison graph")
    
    args = parser.parse_args()
    base_dir = os.path.abspath(args.dataset_dir)
    labels_files = glob.glob(os.path.join(base_dir, "*_labels.npy"))
    
    if args.sample_n and args.sample_n < len(labels_files):
        import random
        random.seed(42)
        labels_files = random.sample(labels_files, args.sample_n)
    
    print(f"Found {len(labels_files)} labeled point clouds to benchmark.")
    
    log_dir = os.path.join(os.path.dirname(args.out_csv), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting ProcessPoolExecutor with {args.num_workers} workers...")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_tile, lf, log_dir, args): lf for lf in labels_files}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            tile_name = os.path.basename(futures[future]).replace("_labels.npy", "")
            print(f"[{i+1}/{len(futures)}] Finished {tile_name}")
            if res: results.extend(res)
                
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved benchmark results to {args.out_csv}")
    
    if args.plot and not df.empty:
        import matplotlib.pyplot as plt
        valid_df = df.dropna(subset=["f1_score"])
        summary = valid_df.groupby("algorithm")[["precision", "recall", "f1_score", "iou"]].mean()
        
        ax = summary.plot(kind="bar", figsize=(10, 6))
        plt.title("Separation Algorithm Performance Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.ylim(0, 1.05)
        plt.legend(loc="lower right")
        
        plot_path = os.path.join(os.path.dirname(args.out_csv), "visualizations", "benchmark_comparison.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
