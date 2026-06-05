import os
import subprocess
import argparse
import multiprocessing
from pathlib import Path

def run_blender_gen(args):
    blender_exec, blend_file, script_path, seed, height, out_obj, out_json = args
    cmd = [
        blender_exec,
        "-b" # headless
    ]
    if blend_file:
        cmd.append(blend_file)
    cmd.extend([
        "-P", script_path,
        "--",
        "--seed", str(seed),
        "--height", str(height),
        "--out_obj", out_obj,
        "--out_json", out_json
    ])
    
    print(f"Generating Tree (Seed: {seed})...")
    # Redirect output to DEVNULL to avoid blender spam in console
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error generating tree {seed}: {result.stderr}")
    else:
        print(f"Finished Tree (Seed: {seed})")

class AssetManager:
    def __init__(self, output_dir="SyntheticPipeline/output/assets", blender_path="blender", blend_file=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.blender_path = blender_path
        self.blend_file = blend_file
        self.script_path = str(Path(__file__).parent / "blender_tree_gen.py")

    def generate_trees(self, num_trees=10, height_range=(8.0, 15.0), num_workers=4):
        tasks = []
        import random
        for i in range(num_trees):
            seed = random.randint(0, 999999)
            height = random.uniform(*height_range)
            out_obj = str(self.output_dir / f"tree_{i:04d}.obj")
            out_json = str(self.output_dir / f"tree_{i:04d}.json")
            
            # Skip if already generated
            if os.path.exists(out_obj) and os.path.exists(out_json):
                print(f"Tree {i:04d} already exists, skipping...")
                continue
                
            tasks.append((self.blender_path, self.blend_file, self.script_path, seed, height, out_obj, out_json))
            
        if not tasks:
            return

        print(f"Starting generation of {len(tasks)} trees using {num_workers} workers...")
        with multiprocessing.Pool(num_workers) as pool:
            pool.map(run_blender_gen, tasks)
        print("Asset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tree Assets using Blender")
    parser.add_argument("--num_trees", type=int, default=5, help="Number of trees to generate")
    parser.add_argument("--blender_path", type=str, default="/Applications/Blender.app/Contents/MacOS/Blender", help="Path to Blender executable")
    parser.add_argument("--blend_file", type=str, default=None, help="Path to .blend file (e.g., Geometry Nodes preset)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    manager = AssetManager(blender_path=args.blender_path, blend_file=args.blend_file)
    manager.generate_trees(num_trees=args.num_trees, num_workers=args.workers)
