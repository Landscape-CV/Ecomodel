import os
import sys
import json
import argparse
from pathlib import Path

# Add the root of SyntheticPipeline to sys.path so we can import modules properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.asset_manager import AssetManager
from pipeline.forest_assembler import ForestAssembler
from pipeline.simulator_open3d import Open3DSimulator
from pipeline.gt_parser import GTParser
from generators.blender_generator import BlenderGenerator
# from generators.custom_generator import CustomGenerator  # Example for future generators

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        print("Please provide a valid configuration file using --config.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in config file {config_path}: {e}")
            sys.exit(1)

def run_pipeline(config):
    # 1. Setup Directories
    out_dirs = config.get("output_dirs", {})
    base_dir = out_dirs.get("base", "SyntheticPipeline/output")
    assets_dir = out_dirs.get("assets", f"{base_dir}/assets")
    scenes_dir = out_dirs.get("scenes", f"{base_dir}/scenes")
    pc_dir = out_dirs.get("pointclouds", f"{base_dir}/pointclouds")

    for d in [assets_dir, scenes_dir, pc_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    sim_config = config.get("simulation", {})
    gen_config = config.get("generator", {})
    num_trees = sim_config.get("num_trees", 5)

    print("=== 1. Generating Assets ===")
    gen_type = gen_config.get("type", "blender")
    
    if gen_type == "blender":
        blender_path = gen_config.get("blender_path")
        if not blender_path:
            print("Error: Generator type is 'blender' but 'blender_path' is missing in config.")
            print("Ensure your config defines generator -> blender_path.")
            sys.exit(1)
            
        generator = BlenderGenerator(
            blender_path=blender_path,
            blend_file=gen_config.get("blend_file")
        )
    elif gen_type == "arbaro":
        from generators.arbaro_generator import ArbaroGenerator
        generator = ArbaroGenerator(
            jar_path=gen_config.get("arbaro_jar_path", "SyntheticPipeline/lib/arbaro/arbaro_cmd.jar"),
            xml_template=gen_config.get("xml_template", "SyntheticPipeline/configs/arbaro.xml"),
            java_bin=gen_config.get("java_path", "java")
        )
    else:
        print(f"Error: Unsupported generator type '{gen_type}'. Available types: blender, arbaro.")
        sys.exit(1)

    asset_manager = AssetManager(generator=generator, output_dir=assets_dir)
    workers = gen_config.get("workers", 2)
    asset_manager.generate_trees(num_trees=num_trees, num_workers=workers)

    print("\n=== 2. Assembling Forest Scene ===")
    assembler = ForestAssembler(asset_dir=assets_dir, output_dir=scenes_dir)
    scene_name = sim_config.get("scene_name", "demo_forest")
    area_size = sim_config.get("area_size", 20.0)
    wind_x = sim_config.get("wind_x", 0.0)
    wind_y = sim_config.get("wind_y", 0.0)
    
    assembler.generate_scene(
        scene_name=scene_name,
        area_size=(area_size, area_size),
        num_trees=num_trees,
        wind_vector=(wind_x, wind_y, 0)
    )

    print("\n=== 3. Simulating LiDAR Scan ===")
    sim = Open3DSimulator(output_dir=pc_dir)
    scene_mesh_path = os.path.join(scenes_dir, f"{scene_name}.ply")
    
    if not os.path.exists(scene_mesh_path):
        print(f"Error: Scene mesh not found at {scene_mesh_path}. Assembly might have failed.")
        sys.exit(1)

    import random
    positions = []
    num_scans = sim_config.get("num_scans", 5)
    for _ in range(num_scans):
        x = random.uniform(-area_size/2, area_size/2)
        y = random.uniform(-area_size/2, area_size/2)
        positions.append([x, y, 1.5])

    noise_params = config.get("noise", {})
    sim.scan(scene_mesh_path, positions, noise_params, f"{scene_name}_scan")

    print("\n=== 4. Parsing Ground Truth ===")
    parser = GTParser(output_dir=scenes_dir)
    scene_gt_json = os.path.join(scenes_dir, f"{scene_name}_gt.json")
    
    if not os.path.exists(scene_gt_json):
        print(f"Error: GT JSON not found at {scene_gt_json}.")
        sys.exit(1)
        
    parser.parse_to_txt(scene_gt_json, f"{scene_name}_gt")

    print("\n=== Demo Run Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Lidar Pipeline Runner")
    parser.add_argument("--config", type=str, help="Path to a specific configuration JSON file.")
    parser.add_argument("--config-dir", type=str, help="Directory containing multiple configs. (Will run the first pipeline_config.json found if not specified otherwise)")
    
    args = parser.parse_args()

    config_path = None
    if args.config:
        config_path = args.config
    elif args.config_dir:
        config_path = os.path.join(args.config_dir, "pipeline_config.json")
    else:
        # Default behavior if left empty
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "pipeline_config.json")
        if os.path.exists(default_config):
            print(f"No config specified, defaulting to: {default_config}")
            config_path = default_config
        else:
            print("Error: No config provided and default configs/pipeline_config.json not found.")
            parser.print_help()
            sys.exit(1)

    config = load_config(config_path)
    run_pipeline(config)
