import json
import os
import random
import argparse
from pathlib import Path
import numpy as np
import psutil
import sys

try:
    import trimesh
except ImportError:
    print("Warning: trimesh is not installed. Please 'pip install trimesh' to run ForestAssembler.")

class ForestAssembler:
    def __init__(self, asset_dir="SyntheticPipeline/output/assets", output_dir="SyntheticPipeline/output/scenes"):
        self.asset_dir = Path(asset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_tree_assets(self):
        assets = []
        for obj_path in self.asset_dir.glob("*.obj"):
            json_path = obj_path.with_suffix(".json")
            if json_path.exists():
                assets.append((obj_path, json_path))
        return assets
        
    def generate_scene(self, scene_name="scene_001", area_size=(50, 50), num_trees=50, wind_vector=(0.1, 0, 0)):
        assets = self._get_tree_assets()
        if not assets:
            print("No tree assets found. Please run AssetManager first.")
            return
            
        print(f"Found {len(assets)} tree assets. Generating forest...")
        
        merged_mesh = trimesh.Trimesh()
        global_gt = []
        
        # Ground plane
        ground = trimesh.creation.box(extents=(area_size[0], area_size[1], 0.1))
        ground.apply_translation((0, 0, -0.05))
        merged_mesh = trimesh.util.concatenate([merged_mesh, ground])
        
        # Calculate wind rotation (tilt)
        # Convert wind vector to a rotation matrix
        # Wind pushes tree, so it rotates around the cross product of wind and UP vector
        wind_mag = np.linalg.norm(wind_vector)
        if wind_mag > 1e-4:
            wind_dir = np.array(wind_vector) / wind_mag
            up_vector = np.array([0, 0, 1])
            rot_axis = np.cross(up_vector, wind_dir)
            if np.linalg.norm(rot_axis) > 1e-4:
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                # Angle proportional to wind magnitude (simple approximation)
                wind_angle = min(wind_mag, math.radians(30)) 
                wind_transform = trimesh.transformations.rotation_matrix(wind_angle, rot_axis)
            else:
                wind_transform = np.eye(4)
        else:
            wind_transform = np.eye(4)

        for tree_id in range(num_trees):
            mem = psutil.virtual_memory()
            if mem.percent > 90.0:
                sys.stderr.write(f"\nCRITICAL MEMORY ERROR: System RAM usage reached {mem.percent}%. Aborting assembly to prevent OS freeze/swap thrashing!\n")
                sys.exit(1)
                
            obj_path, json_path = random.choice(assets)
            
            # 1. Random Placement
            x = random.uniform(-area_size[0]/2, area_size[0]/2)
            y = random.uniform(-area_size[1]/2, area_size[1]/2)
            z = random.uniform(-1.5, 1.5) # Add minor elevation variation
            
            # Random yaw rotation
            yaw = random.uniform(0, 2 * np.pi)
            yaw_transform = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
            
            # Combine transformations: Yaw -> Wind Tilt -> Translation
            transform = np.eye(4)
            transform[:3, 3] = [x, y, z]
            transform = transform @ wind_transform @ yaw_transform
            
            # 2. Process Mesh
            tree_mesh = trimesh.load(str(obj_path))
            tree_mesh.apply_transform(transform)
            merged_mesh = trimesh.util.concatenate([merged_mesh, tree_mesh])
            
            # 3. Process GT Skeleton
            with open(json_path, 'r') as f:
                skeleton = json.load(f)
                
            rot_matrix = transform[:3, :3]
            translation = transform[:3, 3]
            
            for cyl in skeleton:
                # Transform Start
                start_orig = np.array(cyl["start"])
                start_new = rot_matrix @ start_orig + translation
                
                # Transform End
                end_orig = np.array(cyl["end"])
                end_new = rot_matrix @ end_orig + translation
                
                # Transform Axis (only rotation)
                axis_orig = np.array(cyl["axis"])
                axis_new = rot_matrix @ axis_orig
                
                new_cyl = {
                    "tree_instance_id": tree_id,
                    "branch_id": cyl["branch_id"],
                    "start": start_new.tolist(),
                    "end": end_new.tolist(),
                    "radius": cyl["radius"],
                    "length": cyl["length"],
                    "axis": axis_new.tolist()
                }
                global_gt.append(new_cyl)
                
        # Export Scene as PLY (OBJ causes Blender to auto-rotate it 90 degrees on import)
        out_ply = self.output_dir / f"{scene_name}.ply"
        out_json = self.output_dir / f"{scene_name}_gt.json"
        
        merged_mesh.export(str(out_ply))
        
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
    parser.add_argument("--wind_x", type=float, default=0.1)
    parser.add_argument("--wind_y", type=float, default=0.1)
    
    args = parser.parse_args()
    
    assembler = ForestAssembler()
    assembler.generate_scene(
        scene_name=args.scene_name,
        area_size=(args.area_size, args.area_size),
        num_trees=args.num_trees,
        wind_vector=(args.wind_x, args.wind_y, 0)
    )
