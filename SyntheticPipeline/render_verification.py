import open3d as o3d
import numpy as np
import laspy
import json
import colorsys
import os

def create_cylinder_mesh(start, end, radius, color):
    vec = end - start
    length = np.linalg.norm(vec)
    if length < 1e-4: return None
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
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)
    return cyl

def render_scene(scene_name):
    geometries = []
    
    with open(f"SyntheticPipeline/output/scenes/{scene_name}_gt.json", 'r') as f:
        data = json.load(f)
    tree_colors = {}
    for cyl_data in data:
        tree_id = cyl_data["tree_instance_id"]
        if tree_id not in tree_colors:
            h = (tree_id * 0.618033988749895) % 1.0
            r, g, b = colorsys.hls_to_rgb(h, 0.6, 0.9)
            tree_colors[tree_id] = [r, g, b]
        start = np.array(cyl_data["start"])
        end = np.array(cyl_data["end"])
        radius = cyl_data["radius"]
        cyl_mesh = create_cylinder_mesh(start, end, radius, tree_colors[tree_id])
        if cyl_mesh is not None:
            geometries.append(cyl_mesh)

    # Point cloud
    las = laspy.read(f"SyntheticPipeline/output/pointclouds/{scene_name}_scan.laz")
    pts = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.uniform_down_sample(200) # Highly downsample for rendering
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    for geom in geometries:
        vis.add_geometry(geom)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Change view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, 0.5])
    ctr.set_lookat([0, 0, 10])
    ctr.set_up([-0.5, 0.5, 1.0])
    
    vis.poll_events()
    vis.update_renderer()
    
    art_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    img_path = os.path.join(art_dir, f"{scene_name}_render.png")
    vis.capture_screen_image(img_path)
    vis.destroy_window()
    print(f"Saved render to {img_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="demo_forest_reverted")
    args = parser.parse_args()
    render_scene(args.scene_name)
