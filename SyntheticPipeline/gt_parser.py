import json
import numpy as np
import argparse
from pathlib import Path

class GTParser:
    def __init__(self, output_dir="SyntheticPipeline/output/gt"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_to_txt(self, json_path, output_filename="scene_gt_cylinders"):
        print(f"Parsing GT JSON: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if not data:
            print("Warning: JSON data is empty.")
            return None
            
        # Expected format: Cx9 array
        # [start_x, start_y, start_z, radius, axis_x, axis_y, axis_z, length, tree_instance_id]
        
        cylinders_array = []
        for cyl in data:
            row = [
                cyl["start"][0], cyl["start"][1], cyl["start"][2],
                cyl["radius"],
                cyl["axis"][0], cyl["axis"][1], cyl["axis"][2],
                cyl["length"],
                cyl["tree_instance_id"]
            ]
            cylinders_array.append(row)
            
        np_array = np.array(cylinders_array, dtype=np.float32)
        
        out_path = self.output_dir / f"{output_filename}.txt"
        np.savetxt(str(out_path), np_array)
        
        print(f"Saved {len(np_array)} GT cylinders to: {out_path}")
        return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to global GT JSON from ForestAssembler")
    parser.add_argument("--out_name", type=str, default="scene_gt_cylinders")
    args = parser.parse_args()
    
    parser_obj = GTParser()
    parser_obj.parse_to_txt(args.json_path, args.out_name)
