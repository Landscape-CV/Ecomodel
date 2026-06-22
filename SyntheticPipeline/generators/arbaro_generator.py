import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
from typing import Tuple
from .base_generator import BaseTreeGenerator

class ArbaroGenerator(BaseTreeGenerator):
    """
    Generator that interfaces with Arbaro Java CLI to produce tree models.
    """
    def __init__(self, jar_path: str, xml_template: str, java_bin: str = "java"):
        self.jar_path = jar_path
        self.xml_template = xml_template
        self.java_bin = java_bin
        
        if not os.path.exists(self.jar_path):
            raise FileNotFoundError(f"Arbaro jar not found at {self.jar_path}")
        if not os.path.exists(self.xml_template):
            raise FileNotFoundError(f"Arbaro XML template not found at {self.xml_template}")

    def _modify_xml(self, input_xml: str, output_xml: str, target_height: float, remove_leaves: bool):
        """
        Parses the base XML and creates a tailored copy with modified parameters.
        """
        tree = ET.parse(input_xml)
        root = tree.getroot()
        
        # Arbaro wraps everything in <species>
        species = root.find("species")
        if species is None:
            raise ValueError("Invalid Arbaro XML: missing <species> tag.")
            
        for param in species.findall("param"):
            name = param.get("name")
            if name == "Scale":
                # Assuming 'Scale' correlates roughly to height
                param.set("value", str(target_height))
            elif name == "Leaves" and remove_leaves:
                param.set("value", "0")
                
        tree.write(output_xml)

    def generate_tree(self, seed: int, height: float, out_obj: str, out_json: str, **kwargs):
        out_obj_path = Path(out_obj)
        noleaf_obj = out_obj_path.with_name(out_obj_path.stem + "_noleaf.obj")
        temp_xml_leaf = out_obj_path.with_name(out_obj_path.stem + "_temp_leaf.xml")
        temp_xml_noleaf = out_obj_path.with_name(out_obj_path.stem + "_temp_noleaf.xml")
        
        # 1. Generate full tree (Leaves ON)
        self._modify_xml(self.xml_template, str(temp_xml_leaf), height, remove_leaves=False)
        cmd_leaf = [
            self.java_bin, "-jar", self.jar_path,
            "-s", str(seed),
            "-f", "OBJ",
            "-o", out_obj,
            str(temp_xml_leaf)
        ]
        
        # 2. Generate leafless tree (Leaves OFF)
        self._modify_xml(self.xml_template, str(temp_xml_noleaf), height, remove_leaves=True)
        cmd_noleaf = [
            self.java_bin, "-jar", self.jar_path,
            "-s", str(seed),
            "-f", "OBJ",
            "-o", str(noleaf_obj),
            str(temp_xml_noleaf)
        ]
        
        # Execute Generation
        subprocess.run(cmd_leaf, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd_noleaf, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Post-process: Arbaro outputs Y-up. We need Z-up.
        import trimesh
        import numpy as np
        rot_x_90 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        
        for mesh_path in [out_obj, noleaf_obj]:
            mesh = trimesh.load(str(mesh_path), force='mesh')
            mesh.apply_transform(rot_x_90)
            mesh.export(str(mesh_path))
        
        # Clean up temp XMLs
        if temp_xml_leaf.exists(): os.remove(temp_xml_leaf)
        if temp_xml_noleaf.exists(): os.remove(temp_xml_noleaf)
        
        # 3. Extract GT Cylinders from the leafless OBJ
        extractor_script = Path(__file__).parent / "mesh_to_gt_cylinders.py"
        import sys
        cmd_gt = [
            sys.executable, str(extractor_script),
            "--mesh_path", str(noleaf_obj),
            "--out_json", out_json
        ]
        subprocess.run(cmd_gt, check=True, stdout=subprocess.DEVNULL)
