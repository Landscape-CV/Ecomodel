import os
import glob
import numpy as np
import open3d as o3d
from pathlib import Path

try:
    import laspy
except ImportError:
    print("Please pip install laspy lazrs")
    exit(1)

def label_point_cloud(laz_path, trunk_ply_path, out_labels_path, distance_threshold=0.05):
    """
    Computes distances from each point in the LAZ to the Trunk PLY mesh.
    Points within distance_threshold are labeled as Wood (1), else Leaf (0).
    """
    if not os.path.exists(laz_path) or not os.path.exists(trunk_ply_path):
        print(f"[Warning] Missing LAZ or PLY for {laz_path}")
        return False
        
    print(f"Labeling {os.path.basename(laz_path)}...")
    
    # Load point cloud
    las = laspy.read(laz_path)
    points = np.vstack([las.x, las.y, las.z]).T
    
    # Load trunk mesh and create RaycastingScene for fast distance queries
    trunk_mesh = o3d.t.io.read_triangle_mesh(trunk_ply_path)
    if len(trunk_mesh.triangle.indices) == 0:
        print("  [Warning] Trunk mesh is empty.")
        labels = np.zeros(len(points), dtype=np.uint8)
        np.save(out_labels_path, labels)
        return True
        
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(trunk_mesh)
    
    # Compute shortest distance from points to mesh
    points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_distance(points_tensor).numpy()
    
    # Threshold to create binary mask (1 = Wood, 0 = Leaf)
    wood_mask = distances <= distance_threshold
    labels = np.zeros(len(points), dtype=np.uint8)
    labels[wood_mask] = 1
    
    # Save the labels
    np.save(out_labels_path, labels)
    
    wood_count = np.sum(wood_mask)
    leaf_count = len(points) - wood_count
    print(f"  -> Wood points: {wood_count}, Leaf points: {leaf_count}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label synthetic LiDAR points as Wood/Leaf using GT mesh.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(os.path.dirname(script_dir), "testdataset")
    
    parser.add_argument("--dataset_dir", type=str, default=default_dataset, help="Path to testdataset dir")
    parser.add_argument("--dist_thresh", type=float, default=0.05, help="Distance threshold in meters")
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.dataset_dir)
    laz_files = glob.glob(os.path.join(base_dir, "*_scan.laz"))
    
    print(f"Found {len(laz_files)} scans to label.")
    
    for laz_file in laz_files:
        base_name = laz_file.replace("_scan.laz", "")
        trunk_ply = base_name + "_trunk.ply"
        out_labels = base_name + "_labels.npy"
        
        # Skip if already labeled
        if os.path.exists(out_labels):
            continue
            
        label_point_cloud(laz_file, trunk_ply, out_labels, distance_threshold=args.dist_thresh)
