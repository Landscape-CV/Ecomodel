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

def benchmark_segment_rgi(laz_path, labels_path):
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
        filtered_points, instance_ids, orig_indices = segmenter.segment(points, return_indices=True)
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
        return p, r, f1, iou
    except Exception as e:
        print(f"  [Error] Metrics computation failed: {e}")
        return None, None, None, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Separation Algorithms.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(os.path.dirname(script_dir), "testdataset")
    default_out = os.path.join(os.path.dirname(script_dir), "output", "benchmark_results.csv")
    
    parser.add_argument("--dataset_dir", type=str, default=default_dataset, help="Path to testdataset dir")
    parser.add_argument("--out_csv", type=str, default=default_out, help="Output CSV path")
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.dataset_dir)
    labels_files = glob.glob(os.path.join(base_dir, "*_labels.npy"))
    
    print(f"Found {len(labels_files)} labeled point clouds to benchmark.")
    
    results = []
    
    for labels_file in labels_files:
        base_name = labels_file.replace("_labels.npy", "")
        laz_file = base_name + "_scan.laz"
        
        if not os.path.exists(laz_file):
            continue
            
        p, r, f1, iou = benchmark_segment_rgi(laz_file, labels_file)
        
        if p is not None:
            results.append({
                "tile": os.path.basename(base_name),
                "algorithm": "SegmentRGI",
                "precision": p,
                "recall": r,
                "f1_score": f1,
                "iou": iou
            })
            
            # Save progressively so data isn't lost if interrupted
            pd.DataFrame(results).to_csv(args.out_csv, index=False)
            
    # Print final results
    if results:
        df = pd.DataFrame(results)
        print(f"\nFinished processing. Saved benchmark results to {args.out_csv}")
        
        # Print summary
        print("\n--- Summary ---")
        summary = df.groupby("algorithm")[["precision", "recall", "f1_score", "iou"]].mean()
        print(summary)

if __name__ == "__main__":
    main()
