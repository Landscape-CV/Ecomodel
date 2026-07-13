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
from scipy.spatial import cKDTree

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

def sample_cylinders(cylinders, num_points=100000):
    if len(cylinders) == 0:
        return np.zeros((0, 3))
    areas = 2 * np.pi * cylinders[:, 3] * cylinders[:, 7]
    areas[areas <= 0] = 1e-6
    probs = areas / np.sum(areas)
    counts = np.random.multinomial(num_points, probs)
    points = []
    for i, cyl in enumerate(cylinders):
        n = counts[i]
        if n == 0: continue
        start = cyl[0:3]
        radius = cyl[3]
        axis = cyl[4:7]
        length = cyl[7]
        
        z = np.random.uniform(0, length, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        
        if np.abs(axis[0]) > 0.9:
            v1 = np.array([0, 1, 0])
        else:
            v1 = np.array([1, 0, 0])
        v1 = v1 - np.dot(v1, axis) * axis
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(axis, v1)
        
        pts = start + np.outer(z, axis) + radius * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
        points.append(pts)
    return np.vstack(points)

def compute_voxel_metrics(pred_mask, gt_mask, points, voxel_size):
    if len(points) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(int)
    
    unique_voxels, inv_idx = np.unique(voxel_indices, axis=0, return_inverse=True)
    counts = np.bincount(inv_idx, minlength=len(unique_voxels))
    
    pred_counts = np.bincount(inv_idx, weights=pred_mask, minlength=len(unique_voxels))
    gt_counts = np.bincount(inv_idx, weights=gt_mask, minlength=len(unique_voxels))
    
    pred_is_voxel = (pred_counts / counts) > 0.5
    gt_is_voxel = (gt_counts / counts) > 0.5
    
    tp = np.sum(pred_is_voxel & gt_is_voxel)
    fp = np.sum(pred_is_voxel & ~gt_is_voxel)
    fn = np.sum(~pred_is_voxel & gt_is_voxel)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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
            
            cache_file = os.path.join(log_dir, f"{tile_name}_instances.npz")
            if os.path.exists(cache_file):
                print("  [Step 1] Loading Tree Instances from cache...", flush=True)
                npz = np.load(cache_file)
                filtered_points = npz['filtered_points']
                instance_ids = npz['instance_ids']
                orig_indices = npz['orig_indices']
            else:
                print("  [Step 1] Running EcomodelLite Preprocessing & Segmentation...", flush=True)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from ecomodel_lite import EcomodelLite
                from ecomodel_segmenters import SegmenterScanline
                
                model = EcomodelLite(segmenter_type="scanline")
                
                # Stack coordinates, intensity, and original index
                pts_with_idx = np.vstack([las.x, las.y, las.z, intensity, np.arange(len(points))]).T
                
                # 1. Normalize
                pts_with_idx = model.normalize_point_cloud(pts_with_idx)
                
                # 2. Remove ground
                pts_with_idx = model.remove_ground(pts_with_idx)
                
                # 3. Filter intensity
                if pts_with_idx is not None:
                    pts_with_idx = model.filter_intensity(pts_with_idx, model.intensity_threshold)
                
                if pts_with_idx is None or len(pts_with_idx) < 100:
                    filtered_points = None
                    instance_ids = np.array([])
                    orig_indices = np.array([])
                else:
                    # For benchmarking separation on single-tree tiles, bypass SegmenterScanline 
                    # because it is designed for wood-only skeletons and fails/drops points on leafy canopies.
                    filtered_points = pts_with_idx[:, :4]  # Keep xyz + intensity
                    instance_ids = np.zeros(len(filtered_points), dtype=np.int32)
                    orig_indices = pts_with_idx[:, 4].astype(int)
                np.savez_compressed(cache_file, filtered_points=filtered_points, instance_ids=instance_ids, orig_indices=orig_indices)
            
            if filtered_points is None or len(instance_ids) == 0:
                print("  [Warning] No trees found.")
                return []
                
            print("  [Step 1b] Preparing GT Skeletons...", flush=True)
            # Use the pre-computed _labels.npy (where 1 = Trunk, 0 = Canopy)
            # This perfectly matches the _trunk.ply meshes in the test dataset
            if len(gt_labels) != len(points):
                print(f"  [Warning] Mismatch in GT labels length ({len(gt_labels)}) vs points ({len(points)}). Skipping.", flush=True)
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
                                # 3. Compute Metrics
                duration = time.time() - alg_start
                
                # Default to 0
                tp, tr, tf1, tiou = 0.0, 0.0, 0.0, 0.0
                cp, cr, cf1, ciou = 0.0, 0.0, 0.0, 0.0
                
                if len(gt_labels) > 0:
                    gt_is_trunk = (gt_labels == 1)
                    gt_is_canopy = (gt_labels == 0)
                    
                    pred_is_wood = (pred_labels == 1)
                    pred_is_canopy = (pred_labels == 0)
                    
                    # Compute Voxel Metrics for Trunk vs Canopy
                    tp, tr, tf1, tiou = compute_voxel_metrics(pred_mask=pred_is_wood, gt_mask=gt_is_trunk, points=points, voxel_size=args.voxel_size)
                    cp, cr, cf1, ciou = compute_voxel_metrics(pred_mask=pred_is_canopy, gt_mask=gt_is_canopy, points=points, voxel_size=args.voxel_size)
                    
                    print(f"  -> {alg_name} [Trunk Voxel]  | P: {tp:.4f}, R: {tr:.4f}, F1: {tf1:.4f}, IoU: {tiou:.4f}", flush=True)
                    
                    if getattr(args, "visualize", False):
                        vis_dir = os.path.join(os.path.dirname(args.out_csv), "visualizations")
                        os.makedirs(vis_dir, exist_ok=True)
                        import open3d as o3d
                        
                        def save_ply(pts, color, name):
                            if len(pts) == 0: return
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pts)
                            pcd.paint_uniform_color(color)
                            o3d.io.write_point_cloud(os.path.join(vis_dir, name), pcd)
                            
                        # Colors: GT Trunk=Dark Green, GT Canopy=Light Green
                        # Pred Trunk=Dark Red, Pred Canopy=Light Red/Pink
                        save_ply(points[gt_is_trunk, :3], [0.0, 0.6, 0.0], f"{tile_name}_GT_Trunk.ply")
                        save_ply(points[gt_is_canopy, :3], [0.4, 0.9, 0.4], f"{tile_name}_GT_Canopy.ply")
                        save_ply(points[pred_is_wood, :3], [0.8, 0.0, 0.0], f"{tile_name}_{alg_name}_Pred_Trunk.ply")
                        save_ply(points[pred_is_canopy, :3], [1.0, 0.5, 0.5], f"{tile_name}_{alg_name}_Pred_Canopy.ply")
                        
                        print(f"  -> Saved separate visualization layers to {vis_dir}", flush=True)
                        
                else:
                    print("  [Warning] GT points missing, skipping metrics.")
                
                results.append({
                    "tile": tile_name,
                    "algorithm": alg_name,
                    "trunk_precision": tp,
                    "trunk_recall": tr,
                    "trunk_f1_score": tf1,
                    "trunk_iou": tiou,
                    "canopy_precision": cp,
                    "canopy_recall": cr,
                    "canopy_f1_score": cf1,
                    "canopy_iou": ciou,
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
    parser.add_argument("--trunk_radius_threshold", type=float, default=0.05, help="Radius threshold to distinguish trunk from canopy (meters)")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size for computing metrics (meters)")
    parser.add_argument("--visualize", action="store_true", help="Generate colored side-by-side .ply point clouds")
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
    
    df = pd.DataFrame()
    if os.path.exists(args.out_csv):
        try:
            df = pd.read_csv(args.out_csv)
            print(f"Loaded existing results from {args.out_csv}")
        except Exception as e:
            print(f"Could not load existing CSV: {e}")
            
    print(f"Starting ProcessPoolExecutor with {args.num_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_tile, lf, log_dir, args): lf for lf in labels_files}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            tile_name = os.path.basename(futures[future]).replace("_labels.npy", "")
            print(f"[{i+1}/{len(futures)}] Finished {tile_name}")
            
            if res:
                for r in res:
                    if not df.empty and 'tile' in df.columns and 'algorithm' in df.columns:
                        df = df[~((df['tile'] == r['tile']) & (df['algorithm'] == r['algorithm']))]
                    df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
                
                # Real-time save to CSV
                df.to_csv(args.out_csv, index=False)
    
    if args.plot and not df.empty:
        import matplotlib.pyplot as plt
        valid_df = df.dropna(subset=["trunk_f1_score"])
        summary = valid_df.groupby("algorithm")[["trunk_precision", "trunk_recall", "trunk_f1_score", "trunk_iou"]].mean()
        
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
