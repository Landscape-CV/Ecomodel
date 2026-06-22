import trimesh
import numpy as np
import json
import argparse
from pathlib import Path

def extract_cylinders(mesh_path, out_json):
    print(f"Loading mesh for GT extraction: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    # Split the mesh into its disconnected topological components
    # The procedural generator outputs disconnected faces for different branch segments!
    components = mesh.split(only_watertight=False)
    print(f"Found {len(components)} branch segments.")
    
    cylinders = []
    
    for i, comp in enumerate(components):
        if len(comp.vertices) < 4:
            continue
            
        obb = comp.bounding_box_oriented
        
        # OBB transform gives the center and axes
        transform = obb.primitive.transform
        extents = obb.primitive.extents
        
        # The primary axis is the one with the largest extent
        primary_axis_idx = np.argmax(extents)
        length = extents[primary_axis_idx]
        
        if length < 1e-3: continue
        
        axis = transform[:3, primary_axis_idx]
        center = transform[:3, 3]
        
        start = center - axis * (length / 2.0)
        end = center + axis * (length / 2.0)
        
        # Determine which end is 'start' (closer to root/ground)
        # Trees grow upwards, so 'start' should be the lower Z coordinate
        if start[2] > end[2]:
            start, end = end, start
            axis = -axis
            
        # Fix: mathematically robust surface-area derivation for true radius
        # Area = 2 * pi * r * h  =>  r = Area / (2 * pi * h)
        radius = comp.area / (2 * np.pi * length)
        
        # Limit anomalous radii for very flat/tiny fragments
        if radius > length * 2:
            radius = length * 0.1
            
        cylinders.append({
            "tree_instance_id": 0,
            "branch_id": i,
            "start": start.tolist(),
            "end": end.tolist(),
            "radius": float(radius),
            "length": float(length),
            "axis": axis.tolist()
        })
        
    with open(out_json, 'w') as f:
        json.dump(cylinders, f, indent=4)
        
    print(f"Successfully extracted {len(cylinders)} GT cylinders to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()
    
    extract_cylinders(args.mesh_path, args.out_json)
