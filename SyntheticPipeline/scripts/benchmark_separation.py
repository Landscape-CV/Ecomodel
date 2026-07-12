import os
import sys
import glob
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
sys.path.append(root_dir)

from SegmentRGI.SegmentRGI import classify_wood_leaf_point_cloud
from ecomodel_segmenters import SegmenterScanline

def compute_metrics(gt_labels, pred_labels):
    """
    Computes Precision, Recall, F1, and IoU for the Wood class (label = 1).
    """
    tp = np.sum((gt_labels == 1) & (pred_labels == 1))
    fp = np.sum((gt_labels == 0) & (pred_labels == 1))
    fn = np.sum((gt_labels == 1) & (pred_labels == 0))
    tn = np.sum((gt_labels == 0) & (pred_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return precision, recall, f1, iou

def benchmark_segment_rgi(laz_path, labels_path, visualize=False):
    print(f"\nBenchmarking SegmentRGI on {os.path.basename(laz_path)}...")
    
    # Load point cloud
    las = laspy.read(laz_path)
    # The segmenter expects Nx4 array: [x, y, z, intensity]
    intensity = las.intensity / 65535.0  # Normalize back to 0-1 if it was scaled up
    points = np.vstack([las.x, las.y, las.z, intensity]).T
    
    # Load GT labels
    gt_labels = np.load(labels_path)
    
    # 1. Instance Segmentation
    print("  [Step 1] Running Tree Instance Segmentation (Scanline)...")
    try:
        segmenter = SegmenterScanline()
        filtered_points, instance_ids, orig_indices = segmenter.process_with_indices(points)
    except Exception as e:
        print(f"  [Error] Instance Segmentation failed: {e}")
        return None, None, None, None
        
    if filtered_points is None or len(instance_ids) == 0:
        print("  [Warning] No trees found.")
        return None, None, None, None

    # 2. Wood/Leaf Separation (per instance)
    print("  [Step 2] Running Wood/Leaf Separation (SegmentRGI)...")
    pred_labels = np.full(len(points), -1, dtype=np.int32)
    
    unique_instances = np.unique(instance_ids)
    for inst_id in unique_instances:
        if inst_id < 0:
            continue  # Skip noise points
            
        inst_mask = (instance_ids == inst_id)
        inst_points = filtered_points[inst_mask]
        inst_orig_indices = orig_indices[inst_mask]
        
        # Skip small instances
        if len(inst_points) < 50:
            continue
            
        try:
            wood_mask, leaf_mask = classify_wood_leaf_point_cloud(
                point_cloud=inst_points,
                noise_percentile=1
            )
            
            # Map back
            wood_global_indices = inst_orig_indices[wood_mask]
            leaf_global_indices = inst_orig_indices[leaf_mask]
            pred_labels[wood_global_indices] = 1
            pred_labels[leaf_global_indices] = 0
            
        except Exception as e:
            print(f"  [Warning] SegmentRGI failed on instance {inst_id}: {e}")
            
    # Compute metrics on valid points only
    valid_mask = (pred_labels != -1)
    if not np.any(valid_mask):
        print("  [Warning] No points valid for metric computation.")
        return None, None, None, None
        
    valid_gt = gt_labels[valid_mask]
    valid_pred = pred_labels[valid_mask]
    
    try:
        # Calculate metrics
        p, r, f1, iou = compute_metrics(valid_gt, valid_pred)
        print(f"  -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")
        
        if visualize:
            import open3d as o3d
            viz_points = points[valid_mask][:, :3]
            v_gt = valid_gt
            v_pred = valid_pred
            
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(viz_points)
            gt_colors = np.zeros((len(viz_points), 3))
            gt_colors[v_gt == 1] = [0, 1, 0] # Wood=Green
            gt_colors[v_gt == 0] = [1, 0, 0] # Leaf=Red
            gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)
            
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(viz_points)
            pred_colors = np.zeros((len(viz_points), 3))
            pred_colors[v_pred == 1] = [0, 1, 0]
            pred_colors[v_pred == 0] = [1, 0, 0]
            pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)
            
            bbox = gt_pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            pred_pcd.translate(np.array([extent[0] * 1.2, 0, 0]))
            
            print("Saving visualization: Ground Truth (Left) vs Prediction (Right). Green=Wood, Red=Leaf")
            
            # Combine point clouds
            combined_pcd = gt_pcd + pred_pcd
            
            # Create visualizations directory
            viz_dir = os.path.join(os.path.dirname(os.path.dirname(laz_path)), "output", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            out_file = os.path.join(viz_dir, os.path.basename(laz_path).replace(".laz", "_viz.ply"))
            o3d.io.write_point_cloud(out_file, combined_pcd)
            print(f"Saved side-by-side visualization to: {out_file}")
            
        return p, r, f1, iou
    except Exception as e:
        print(f"  [Error] Metrics computation failed: {e}")
        return None, None, None, None

def process_tile(labels_file, log_dir, visualize=False):
    import os, sys, time
    base_name = labels_file.replace("_labels.npy", "")
    tile_name = os.path.basename(base_name)
    laz_file = base_name + "_scan.laz"
    
    if not os.path.exists(laz_file):
        return None
        
    log_file = os.path.join(log_dir, f"{tile_name}.log")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    
    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            sys.stdout = f
            sys.stderr = f
            
            print(f"Starting processing for {tile_name}...")
            p, r, f1, iou = benchmark_segment_rgi(laz_file, labels_file, visualize=visualize)
            
            duration = time.time() - start_time
            if p is not None:
                return {
                    "tile": tile_name,
                    "algorithm": "SegmentRGI",
                    "precision": p,
                    "recall": r,
                    "f1_score": f1,
                    "iou": iou,
                    "duration_sec": duration
                }
            return {"tile": tile_name, "error": "No valid points", "duration_sec": duration}
    except Exception as e:
        duration = time.time() - start_time
        with open(log_file, "a") as f:
            f.write(f"\nException occurred: {e}\n")
        return {"tile": tile_name, "error": str(e), "duration_sec": duration}
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

def main():
    import argparse
    import concurrent.futures
    import multiprocessing
    import time
    
    parser = argparse.ArgumentParser(description="Benchmark Separation Algorithms.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(os.path.dirname(script_dir), "testdataset")
    default_out = os.path.join(os.path.dirname(script_dir), "output", "benchmark_results.csv")
    
    parser.add_argument("--dataset_dir", type=str, default=default_dataset, help="Path to testdataset dir")
    parser.add_argument("--out_csv", type=str, default=default_out, help="Output CSV path")
    
    # Default to cpu_count - 1 to leave room for the system
    default_workers = max(1, multiprocessing.cpu_count() - 1)
    parser.add_argument("--num_workers", type=int, default=default_workers, help="Number of worker processes")
    parser.add_argument("--sample_n", type=int, default=None, help="Randomly sample N tiles")
    parser.add_argument("--visualize", action="store_true", help="Visualize GT vs Pred side-by-side")
    
    args = parser.parse_args()
    
    if args.visualize:
        print("Visualization enabled. Side-by-side point clouds will be saved to output/visualizations/")
    
    base_dir = os.path.abspath(args.dataset_dir)
    labels_files = glob.glob(os.path.join(base_dir, "*_labels.npy"))
    
    if args.sample_n and args.sample_n < len(labels_files):
        import random
        random.seed(42)
        labels_files = random.sample(labels_files, args.sample_n)
        print(f"Randomly sampled {args.sample_n} point clouds.")
    
    print(f"Found {len(labels_files)} labeled point clouds to benchmark.")
    
    log_dir = os.path.join(os.path.dirname(args.out_csv), "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs for each tile will be saved to {log_dir}")
    
    results = []
    
    # We use ProcessPoolExecutor for CPU-bound multiprocessing
    print(f"Starting ProcessPoolExecutor with {args.num_workers} workers...")
    
    main_start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_tile, lf, log_dir, args.visualize): lf for lf in labels_files}
        
        # as_completed yields futures as they finish
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            
            # Print progress to the main console
            tile_file = futures[future]
            tile_name = os.path.basename(tile_file).replace("_labels.npy", "")
            
            tiles_finished = i + 1
            tiles_total = len(labels_files)
            tiles_remaining = tiles_total - tiles_finished
            
            elapsed_time = time.time() - main_start_time
            avg_time_per_tile = elapsed_time / tiles_finished
            eta_seconds = avg_time_per_tile * tiles_remaining
            
            eta_mins, eta_secs = divmod(int(eta_seconds), 60)
            eta_hours, eta_mins = divmod(eta_mins, 60)
            
            eta_str = ""
            if eta_hours > 0:
                eta_str += f"{eta_hours}h "
            if eta_mins > 0 or eta_hours > 0:
                eta_str += f"{eta_mins}m "
            eta_str += f"{eta_secs}s"
            
            if res is not None:
                tile_duration = res.get('duration_sec', 0)
                dur_mins, dur_secs = divmod(int(tile_duration), 60)
                
                print(f"[{tiles_finished}/{tiles_total}] Finished {tile_name} (took {dur_mins}m {dur_secs}s) - ETA: {eta_str}")
                
                # Only save valid results to CSV
                if "error" not in res:
                    results.append(res)
                    # Save progressively so data isn't lost if interrupted
                    pd.DataFrame(results).to_csv(args.out_csv, index=False)
            else:
                print(f"[{tiles_finished}/{tiles_total}] Finished {tile_name} (skipped/failed) - ETA: {eta_str}")
            
    # Print final results
    if results:
        df = pd.DataFrame(results)
        total_time = time.time() - main_start_time
        t_hours, remainder = divmod(int(total_time), 3600)
        t_mins, t_secs = divmod(remainder, 60)
        print(f"\nFinished processing in {t_hours}h {t_mins}m {t_secs}s. Saved benchmark results to {args.out_csv}")
        
        # Print summary
        print("\n--- Summary ---")
        summary = df.groupby("algorithm")[["precision", "recall", "f1_score", "iou", "duration_sec"]].mean()
        print(summary)

if __name__ == "__main__":
    main()
