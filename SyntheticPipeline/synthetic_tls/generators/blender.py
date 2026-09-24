import os
import subprocess
import warnings
from pathlib import Path

from .base import BaseTreeGenerator
from .cylinders import extract_cylinders


class BlenderGenerator(BaseTreeGenerator):
    """
    Host-side tree generator that launches Blender headless with the blender_runtime entrypoint.
    """

    def __init__(self, blender_path: str, blend_file: str = None):
        if not blender_path or not os.path.exists(blender_path):
            warnings.warn(
                f"Blender executable not found at '{blender_path}'. Please check your configuration."
            )

        self.blender_path = blender_path
        self.blend_file = blend_file
        self.script_path = str(
            Path(__file__).parent / "blender_runtime" / "entrypoint.py"
        )

    def generate_tree(self, seed: int, height: float, out_obj: str, out_json: str, **kwargs):
        cmd = [
            self.blender_path,
            "-b",  # headless
        ]
        if self.blend_file and os.path.exists(self.blend_file):
            cmd.append(self.blend_file)

        cmd.extend([
            "-P", self.script_path,
            "--",
            "--seed", str(seed),
            "--height", str(height),
            "--out_obj", out_obj,
            "--out_json", out_json,
        ])

        print(f"Generating Tree (Seed: {seed}) via Blender...")

        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print(f"Error generating tree {seed}: {result.stderr}")
            raise RuntimeError(f"Blender generation failed for seed {seed}")

        out_noleaf = out_obj.replace(".obj", "_noleaf.obj")
        if os.path.exists(out_noleaf):
            try:
                extract_cylinders(out_noleaf, out_json)
            except Exception as exc:
                print(f"Error parsing GT for tree {seed}: {exc}")
        else:
            warnings.warn(
                f"Leafless mesh {out_noleaf} was not generated. GT JSON parsing skipped."
            )

        print(f"Finished Tree (Seed: {seed})")
