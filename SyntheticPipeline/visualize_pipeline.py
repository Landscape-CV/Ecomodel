import open3d as o3d
import numpy as np
import laspy
import json
import argparse
import colorsys

def create_cylinder_mesh(start, end, radius, color):
    # Calculate vector from start to end
    vec = end - start
    length = np.linalg.norm(vec)
    if length < 1e-4:
        return None
        
    # Create a basic cylinder along Z axis
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=12)
    
    # Open3D cylinder is centered at origin along Z axis [-height/2, height/2]
    # We want it to go from start to end.
    # 1. Translate so bottom is at origin
    cyl.translate(np.array([0, 0, length / 2]))
    
    # 2. Rotate to align Z axis with vec
    axis = vec / length
    z_axis = np.array([0, 0, 1])
    
    # Axis-angle representation for rotation
    v = np.cross(z_axis, axis)
    c = np.dot(z_axis, axis)
    k = 1.0 / (1.0 + c) if c != -1.0 else 0.0
    
    if np.linalg.norm(v) > 1e-6:
        R = np.array([
            [v[0]*v[0]*k + c,     v[0]*v[1]*k - v[2], v[0]*v[2]*k + v[1]],
            [v[1]*v[0]*k + v[2], v[1]*v[1]*k + c,     v[1]*v[2]*k - v[0]],
            [v[2]*v[0]*k - v[1], v[2]*v[1]*k + v[0], v[2]*v[2]*k + c]
        ])
        cyl.rotate(R, center=(0,0,0))
    elif c == -1.0: # 180 degree rotation
        cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0])), center=(0,0,0))
        
    # 3. Translate to actual start
    cyl.translate(start)
    
    # Paint it
    cyl.paint_uniform_color(color)
    return cyl

def main():
    parser = argparse.ArgumentParser(description="Visualize Point Cloud and GT Cylinders together")
    parser.add_argument("--laz_path", type=str, default="SyntheticPipeline/output/pointclouds/demo_forest_scan.laz")
    parser.add_argument("--gt_path", type=str, default="SyntheticPipeline/output/scenes/demo_forest_gt.json")
    parser.add_argument("--gt_mesh_path", type=str, default=None, help="Path to ground truth mesh (e.g. .ply) to render instead of JSON cylinders")
    parser.add_argument("--downsample", type=int, default=50, help="Keep 1 out of N points to make the cloud translucent")
    args = parser.parse_args()

    geometries = []

    # 1. Load point cloud
    print(f"Loading point cloud: {args.laz_path}")
    try:
        las = laspy.read(args.laz_path)
        pts = np.vstack((las.x, las.y, las.z)).transpose()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        # Apply intensity as grayscale color if available
        try:
            intensity = las.intensity / 65535.0
            colors = np.vstack((intensity, intensity, intensity)).transpose()
            pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception as e:
            print("No intensity found or failed to parse intensity, using default color.")
            
        if args.downsample > 1:
            print(f"Downsampling point cloud by factor of {args.downsample} to make it 'translucent'...")
            pcd = pcd.uniform_down_sample(args.downsample)
            
        geometries.append(pcd)
    except FileNotFoundError:
        print(f"Warning: Could not find point cloud {args.laz_path}")

    # 2. Load GT
    if args.gt_mesh_path:
        print(f"Loading Ground Truth Mesh: {args.gt_mesh_path}")
        try:
            gt_mesh = o3d.io.read_triangle_mesh(args.gt_mesh_path)
            if not gt_mesh.is_empty():
                gt_mesh.compute_vertex_normals()
                gt_mesh.paint_uniform_color([1.0, 0.2, 0.2]) # Paint GT mesh red
                geometries.append(gt_mesh)
            else:
                print(f"Warning: GT mesh {args.gt_mesh_path} is empty or failed to load.")
        except Exception as e:
            print(f"Warning: Could not load GT mesh: {e}")
    else:
        print(f"Loading Ground Truth JSON: {args.gt_path}")
        try:
            with open(args.gt_path, 'r') as f:
                data = json.load(f)

            # Dictionary to keep track of colors per tree instance
            tree_colors = {}
            
            print(f"Generating cylinder meshes for {len(data)} GT segments...")
            for cyl_data in data:
                tree_id = cyl_data["tree_instance_id"]
                if tree_id not in tree_colors:
                    # Generate a distinct bright color using golden ratio for hue distribution
                    h = (tree_id * 0.618033988749895) % 1.0
                    r, g, b = colorsys.hls_to_rgb(h, 0.6, 0.9) # Bright pastel colors
                    tree_colors[tree_id] = [r, g, b]
                    
                start = np.array(cyl_data["start"])
                end = np.array(cyl_data["end"])
                radius = cyl_data["radius"]
                
                cyl_mesh = create_cylinder_mesh(start, end, radius, tree_colors[tree_id])
                if cyl_mesh is not None:
                    geometries.append(cyl_mesh)
        except FileNotFoundError:
            print(f"Warning: Could not find GT JSON {args.gt_path}")

    if not geometries:
        print("No geometries to display. Exiting.")
        return

    # 3. Visualize with Toggle
    print("Launching Open3D visualizer. Use the mouse to rotate/pan/zoom.")
    print("Tip: Press 'P' in the visualizer window to toggle the Point Cloud ON/OFF.")
    print("Tip: Press '+' or '-' to change point size.")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PyTLidar Visualizer (Press 'P' to toggle Point Cloud)", width=1280, height=720)
    
    is_pcd_visible = True
    # We assume the first geometry is the Point Cloud
    pcd_geom = geometries[0] if isinstance(geometries[0], o3d.geometry.PointCloud) else None

    def toggle_pcd(vis_arg):
        nonlocal is_pcd_visible
        if pcd_geom is None: return False
        
        if is_pcd_visible:
            vis_arg.remove_geometry(pcd_geom, reset_bounding_box=False)
            is_pcd_visible = False
            print("Point cloud hidden.")
        else:
            vis_arg.add_geometry(pcd_geom, reset_bounding_box=False)
            is_pcd_visible = True
            print("Point cloud visible.")
        return False
        
    vis.register_key_callback(ord("P"), toggle_pcd)
    vis.register_key_callback(ord("p"), toggle_pcd)
    
    for geom in geometries:
        vis.add_geometry(geom)
        
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
