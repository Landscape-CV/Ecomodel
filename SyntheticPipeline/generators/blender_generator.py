import os
import subprocess
import sys
import warnings
from pathlib import Path
from .base_generator import BaseTreeGenerator

class BlenderGenerator(BaseTreeGenerator):
    """
    Implementation of the tree generator using Blender.
    This wrapper calls the internal `blender_script.py` which runs within the Blender Python environment.
    """
    def __init__(self, blender_path: str, blend_file: str = None):
        if not blender_path or not os.path.exists(blender_path):
            warnings.warn(f"Blender executable not found at '{blender_path}'. Please check your configuration.")
            
        self.blender_path = blender_path
        self.blend_file = blend_file
        # Path to the script that Blender will execute internally
        self.script_path = str(Path(__file__).parent / "blender_script.py")
        
    def generate_tree(self, seed: int, height: float, out_obj: str, out_json: str, **kwargs):
        """
        Generates a tree using Blender.
        """
        cmd = [
            self.blender_path,
            "-b" # headless
        ]
        if self.blend_file and os.path.exists(self.blend_file):
            cmd.append(self.blend_file)
            
        cmd.extend([
            "-P", self.script_path,
            "--",
            "--seed", str(seed),
            "--height", str(height),
            "--out_obj", out_obj,
            "--out_json", out_json
        ])
        
        print(f"Generating Tree (Seed: {seed}) via Blender...")
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error generating tree {seed}: {result.stderr}")
            raise RuntimeError(f"Blender generation failed for seed {seed}")
        else:
            # Generate Ground Truth JSON by parsing the leafless mesh
            out_noleaf = out_obj.replace(".obj", "_noleaf.obj")
            parser_script = str(Path(__file__).parent / "mesh_to_gt_cylinders.py")
            
            # Since mesh_to_gt_cylinders uses standard python, run it with current executable
            if os.path.exists(out_noleaf):
                parse_cmd = [sys.executable, parser_script, "--mesh_path", out_noleaf, "--out_json", out_json]
                parse_result = subprocess.run(parse_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if parse_result.returncode != 0:
                    print(f"Error parsing GT for tree {seed}: {parse_result.stderr}")
            else:
                warnings.warn(f"Leafless mesh {out_noleaf} was not generated. GT JSON parsing skipped.")
                
            print(f"Finished Tree (Seed: {seed})")
