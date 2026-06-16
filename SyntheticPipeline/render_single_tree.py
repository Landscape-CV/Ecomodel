import open3d as o3d
import numpy as np
import json
import trimesh
import os

def render_single_tree(mesh_path, gt_json, out_img):
    print("Loading tree mesh...")
    mesh = trimesh.load(mesh_path)
    
    # Convert trimesh to Open3D point cloud by sampling points
    pts = trimesh.sample.sample_surface(mesh, 100000)[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.2, 0.2, 0.2]) # Dark grey
    
    geometries = [pcd]
    
    print("Loading GT cylinders...")
    with open(gt_json, 'r') as f:
        data = json.load(f)
        
    for cyl_data in data:
        start = np.array(cyl_data["start"])
        end = np.array(cyl_data["end"])
        radius = cyl_data["radius"]
        
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-4: continue
        
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=12)
        cyl.translate(np.array([0, 0, length / 2]))
        
        axis = vec / length
        z_axis = np.array([0, 0, 1])
        
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
        elif c == -1.0:
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0])), center=(0,0,0))
            
        cyl.translate(start)
        cyl.paint_uniform_color([1.0, 0.0, 0.0]) # Red cylinders
        
        geometries.append(cyl)
        
    print("Rendering...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    for geom in geometries:
        vis.add_geometry(geom)
        
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.array([1, 1, 1])
    
    # Auto-fit camera
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_img)
    vis.destroy_window()
    print(f"Saved render to {out_img}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--out_img", type=str, required=True)
    args = parser.parse_args()
    
    render_single_tree(args.mesh_path, args.gt_json, args.out_img)
