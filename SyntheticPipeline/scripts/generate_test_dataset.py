import os
import sys
import glob
import json
import random
import shutil
from pathlib import Path

# Add the root of SyntheticPipeline to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.asset_manager import AssetManager
from pipeline.forest_assembler import ForestAssembler
from pipeline.simulator_open3d import Open3DSimulator
from pipeline.gt_parser import GTParser
from generators.arbaro_generator import ArbaroGenerator

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arbaro_trees_dir = os.path.join(base_dir, "lib", "arbaro", "trees")
    jar_path = os.path.join(base_dir, "lib", "arbaro", "arbaro_cmd.jar")
    
    # Temporary output directories
    assets_dir = os.path.join(base_dir, "output", "temp_assets")
    scenes_dir = os.path.join(base_dir, "output", "temp_scenes")
    pc_dir = os.path.join(base_dir, "output", "temp_pointclouds")
    
    # Final dataset directory
    dataset_dir = os.path.join(base_dir, "testdataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Noise parameters
    noise_params = {
        "resolution_theta_deg": 0.2,
        "resolution_phi_deg": 0.2,
        "distance_noise_std": 0.005,
        "wind_sway_std": 0.02,
        "beam_divergence_noise_std": 0.005,
        "voxel_downsample_size": 0.05,
        "wind_x": 0.0,
        "wind_y": 0.0
    }
    
    area_size = 20.0
    pool_size = 10
    batches = 1
    tiles_per_batch = 10
    
    xml_files = glob.glob(os.path.join(arbaro_trees_dir, "*.xml"))
    
    for xml_path in xml_files:
        veg_name = Path(xml_path).stem
        print(f"\n======================================")
        print(f"Processing vegetation: {veg_name}")
        print(f"======================================")
        
        # 1. Clear temp assets dir and recreate
        if os.path.exists(assets_dir):
            shutil.rmtree(assets_dir)
        os.makedirs(assets_dir, exist_ok=True)
        
        # 2. Generate tree pool
        generator = ArbaroGenerator(
            jar_path=jar_path,
            xml_template=xml_path,
            java_bin="java"
        )
        asset_manager = AssetManager(generator=generator, output_dir=assets_dir)
        print(f"Generating pool of {pool_size} {veg_name} trees...")
        asset_manager.generate_trees(num_trees=pool_size, num_workers=4)
        
        # 3. Generate Tiles
        total_tiles = batches * tiles_per_batch
        
        for batch_i in range(batches):
            for tile_j in range(tiles_per_batch):
                tile_id = batch_i * tiles_per_batch + tile_j
                print(f"  -> Generating tile {tile_id + 1}/{total_tiles} for {veg_name}")
                
                # Clear scenes and pc dir
                if os.path.exists(scenes_dir):
                    shutil.rmtree(scenes_dir)
                os.makedirs(scenes_dir, exist_ok=True)
                
                if os.path.exists(pc_dir):
                    shutil.rmtree(pc_dir)
                os.makedirs(pc_dir, exist_ok=True)
                
                num_trees = random.randint(2, 5)
                num_scans = random.randint(2, 3)
                scene_name = f"{veg_name}_scene"
                
                # Assemble scene
                assembler = ForestAssembler(asset_dir=assets_dir, output_dir=scenes_dir)
                assembler.generate_scene(
                    scene_name=scene_name,
                    area_size=(area_size, area_size),
                    num_trees=num_trees,
                    wind_vector=(0.0, 0.0, 0.0)
                )
                
                # Simulate LiDAR
                scene_mesh_path = os.path.join(scenes_dir, f"{scene_name}.ply")
                if not os.path.exists(scene_mesh_path):
                    print(f"     [Error] Scene mesh not found: {scene_mesh_path}")
                    continue
                
                positions = []
                for _ in range(num_scans):
                    x = random.uniform(-area_size/2, area_size/2)
                    y = random.uniform(-area_size/2, area_size/2)
                    positions.append([x, y, 1.5])
                
                sim = Open3DSimulator(output_dir=pc_dir)
                out_scan_name = f"{scene_name}_scan"
                sim.scan(scene_mesh_path, positions, noise_params, out_scan_name)
                
                # Generate GT txt
                parser = GTParser(output_dir=scenes_dir)
                scene_gt_json = os.path.join(scenes_dir, f"{scene_name}_gt.json")
                if os.path.exists(scene_gt_json):
                    parser.parse_to_txt(scene_gt_json, f"{scene_name}_gt")
                
                # Copy to final dataset dir
                final_laz = os.path.join(dataset_dir, f"{veg_name}_tile_{tile_id}_scan.laz")
                final_trunk = os.path.join(dataset_dir, f"{veg_name}_tile_{tile_id}_trunk.ply")
                final_gt_json = os.path.join(dataset_dir, f"{veg_name}_tile_{tile_id}_gt.json")
                final_gt_txt = os.path.join(dataset_dir, f"{veg_name}_tile_{tile_id}_gt.txt")
                
                src_laz = os.path.join(pc_dir, f"{out_scan_name}.laz")
                src_trunk = os.path.join(scenes_dir, f"{scene_name}_gt_mesh.ply")
                src_gt_json = os.path.join(scenes_dir, f"{scene_name}_gt.json")
                src_gt_txt = os.path.join(scenes_dir, f"{scene_name}_gt.txt")
                
                if os.path.exists(src_laz):
                    shutil.copy(src_laz, final_laz)
                if os.path.exists(src_trunk):
                    shutil.copy(src_trunk, final_trunk)
                if os.path.exists(src_gt_json):
                    shutil.copy(src_gt_json, final_gt_json)
                if os.path.exists(src_gt_txt):
                    shutil.copy(src_gt_txt, final_gt_txt)
                
                # Write meta
                meta_path = os.path.join(dataset_dir, f"{veg_name}_tile_{tile_id}_meta.json")
                meta = {
                    "vegetation": veg_name,
                    "tile_id": tile_id,
                    "num_trees": num_trees,
                    "num_scans": num_scans,
                    "area_size": area_size,
                    "scan_positions": positions,
                    "noise_params": noise_params
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
                    
    print("\nDataset generation completed successfully!")

if __name__ == "__main__":
    main()
