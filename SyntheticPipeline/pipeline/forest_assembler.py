import json
import os
import random
import argparse
import math
import warnings
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import psutil
import sys

try:
    import trimesh
except ImportError:
    warnings.warn("trimesh is not installed. Please 'pip install trimesh' to run ForestAssembler.")

class ForestAssembler:
    """
    Assembles individual tree assets into a combined forest scene.
    It randomly places trees, applies wind tilt and scale variations,
    and accurately transforms ground truth skeleton parameters.
    """
    def __init__(self, asset_dir="SyntheticPipeline/output/assets", output_dir="SyntheticPipeline/output/scenes"):
        self.asset_dir = Path(asset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_tree_assets(self):
        assets = []
        for obj_path in self.asset_dir.glob("*.obj"):
            if "_noleaf" in obj_path.name:
                continue
            json_path = obj_path.with_suffix(".json")
            noleaf_path = obj_path.with_name(obj_path.stem + "_noleaf.obj")
            if json_path.exists():
                assets.append((obj_path, json_path, noleaf_path if noleaf_path.exists() else None))
        return assets
        
    def generate_scene(
        self, 
        scene_name: str = "scene_001", 
        area_size: Tuple[float, float] = (50.0, 50.0), 
        num_trees: int = 50, 
        wind_vector: Tuple[float, float, float] = (0.1, 0.0, 0.0)
    ) -> None:
        """
        Generates a forest scene mesh and global ground truth JSON.
        
        Args:
            scene_name (str): Name of the generated scene files.
            area_size (tuple): Width and height of the generation area.
            num_trees (int): Number of trees to place.
            wind_vector (tuple): 3D vector representing wind direction and magnitude.
        """
        assert area_size[0] > 0 and area_size[1] > 0, "Area size dimensions must be positive."
        assert num_trees > 0, "Number of trees must be greater than zero."

        assets = self._get_tree_assets()
        if not assets:
            warnings.warn("No tree assets found. Please run AssetManager first.")
            return
            
        print(f"Found {len(assets)} tree assets. Generating forest...")
        
        merged_mesh = trimesh.Trimesh()
        merged_noleaf_mesh = trimesh.Trimesh()
        global_gt = []
        
        # Ground plane: Create a dense bumpy grid to simulate grassy floor
        import scipy.ndimage
        resolution = 0.5  # half-meter vertices
        nx = int(area_size[0] / resolution)
        ny = int(area_size[1] / resolution)
        
        # We can just create vertices manually and build faces
        xs = np.linspace(-area_size[0]/2, area_size[0]/2, nx)
        ys = np.linspace(-area_size[1]/2, area_size[1]/2, ny)
        X, Y = np.meshgrid(xs, ys)
        
        # Perturb Z
        noise = np.random.normal(0, 0.5, X.shape)
        smoothed_noise = scipy.ndimage.gaussian_filter(noise, sigma=2.0)
        
        vertices = np.column_stack([X.ravel(), Y.ravel(), smoothed_noise.ravel()])
        
        faces = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                v0 = i * nx + j
                v1 = v0 + 1
                v2 = v0 + nx
                v3 = v0 + nx + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
                
        grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
        merged_mesh = trimesh.util.concatenate([merged_mesh, grid_mesh])

        for tree_id in range(num_trees):
            mem = psutil.virtual_memory()
            if mem.percent > 90.0:
                sys.stderr.write(f"\nCRITICAL MEMORY ERROR: System RAM usage reached {mem.percent}%. Aborting assembly to prevent OS freeze/swap thrashing!\n")
                sys.exit(1)
                
            obj_path, json_path, noleaf_path = random.choice(assets)
            
            # 1. Random Placement
            x = random.uniform(-area_size[0]/2, area_size[0]/2)
            y = random.uniform(-area_size[1]/2, area_size[1]/2)
            z = 0.0 # Attach firmly to the ground
            
            # Random yaw rotation
            yaw = random.uniform(0, 2 * np.pi)
            yaw_transform = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
            
            # Random scale for height variation
            scale = random.uniform(0.7, 1.5)
            scale_transform = trimesh.transformations.scale_matrix(scale)
            
            # Combine base transformations: Scale -> Yaw (Wind tilt removed, handled dynamically in LiDAR sim)
            rot_scale_transform = yaw_transform @ scale_transform
            
            # 2. Process Mesh
            tree_mesh = trimesh.load(str(obj_path))
            tree_mesh.apply_transform(rot_scale_transform)
            
            # Mathematically ground the tree: find the lowest vertex and offset Z so it touches 0
            min_z = tree_mesh.bounds[0][2]
            z_translation = -min_z
            
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = [x, y, z_translation]
            
            tree_mesh.apply_transform(translation_matrix)
            merged_mesh = trimesh.util.concatenate([merged_mesh, tree_mesh])
            
            if noleaf_path:
                noleaf_mesh = trimesh.load(str(noleaf_path))
                noleaf_mesh.apply_transform(rot_scale_transform)
                noleaf_mesh.apply_transform(translation_matrix)
                merged_noleaf_mesh = trimesh.util.concatenate([merged_noleaf_mesh, noleaf_mesh])
            
            # Full transform matrix for Ground Truth calculation
            transform = translation_matrix @ rot_scale_transform
            
            # 3. Process GT Skeleton
            with open(json_path, 'r') as f:
                skeleton = json.load(f)
                
            rot_matrix = transform[:3, :3]
            translation = transform[:3, 3]
            
            for cyl in skeleton:
                assert "start" in cyl and "end" in cyl and "axis" in cyl, "Invalid skeleton data format."
                
                # Transform Start
                start_orig = np.array(cyl["start"])
                start_new = rot_matrix @ start_orig + translation
                
                # Transform End
                end_orig = np.array(cyl["end"])
                end_new = rot_matrix @ end_orig + translation
                
                # Transform Axis (only rotation, need to normalize in case of scale)
                axis_orig = np.array(cyl["axis"])
                axis_new = rot_matrix @ axis_orig
                axis_new = axis_new / np.linalg.norm(axis_new)
                
                new_cyl = {
                    "tree_instance_id": tree_id,
                    "branch_id": cyl["branch_id"],
                    "start": start_new.tolist(),
                    "end": end_new.tolist(),
                    "radius": cyl["radius"] * scale,
                    "length": cyl["length"] * scale,
                    "axis": axis_new.tolist()
                }
                global_gt.append(new_cyl)
                
        # Export Scene as PLY (OBJ causes Blender to auto-rotate it 90 degrees on import)
        out_ply = self.output_dir / f"{scene_name}.ply"
        out_noleaf_ply = self.output_dir / f"{scene_name}_gt_mesh.ply"
        out_json = self.output_dir / f"{scene_name}_gt.json"
        
        merged_mesh.export(str(out_ply))
        if len(merged_noleaf_mesh.faces) > 0:
            merged_noleaf_mesh.export(str(out_noleaf_ply))
        
        with open(out_json, 'w') as f:
            json.dump(global_gt, f, indent=4)
            
        print(f"Generated scene {scene_name} with {num_trees} trees.")
        print(f"Mesh: {out_ply}")
        print(f"GT  : {out_json}")

if __name__ == "__main__":
    import math # imported here since math wasn't at top
    parser = argparse.ArgumentParser(description="Assemble Forest Scene from Tree Assets")
    parser.add_argument("--scene_name", type=str, default="forest_001")
    parser.add_argument("--num_trees", type=int, default=10)
    parser.add_argument("--area_size", type=float, default=200.0)
    parser.add_argument("--wind_x", type=float, default=0.0)
    parser.add_argument("--wind_y", type=float, default=0.0)
    
    args = parser.parse_args()
    
    assembler = ForestAssembler()
    assembler.generate_scene(
        scene_name=args.scene_name,
        area_size=(args.area_size, args.area_size),
        num_trees=args.num_trees,
        wind_vector=(args.wind_x, args.wind_y, 0)
    )
