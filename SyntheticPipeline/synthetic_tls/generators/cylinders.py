"""Fit OBB cylinders to disconnected leafless mesh components for GT skeletons."""

import argparse
import json

import numpy as np
import trimesh


def extract_cylinders(mesh_path, out_json):
    print(f"Loading mesh for GT extraction: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")

    # Split the mesh into its disconnected topological components.
    # Procedural generators often output disconnected faces per branch segment.
    components = mesh.split(only_watertight=False)
    print(f"Found {len(components)} branch segments.")

    cylinders = []

    for i, comp in enumerate(components):
        if len(comp.vertices) < 4:
            continue

        obb = comp.bounding_box_oriented

        transform = obb.primitive.transform
        extents = obb.primitive.extents

        primary_axis_idx = np.argmax(extents)
        length = extents[primary_axis_idx]

        if length < 1e-3:
            continue

        axis = transform[:3, primary_axis_idx]
        center = transform[:3, 3]

        start = center - axis * (length / 2.0)
        end = center + axis * (length / 2.0)

        # Prefer lower-Z end as start (trees grow upward).
        if start[2] > end[2]:
            start, end = end, start
            axis = -axis

        # Area = 2 * pi * r * h  =>  r = Area / (2 * pi * h)
        radius = comp.area / (2 * np.pi * length)

        if radius > length * 2:
            radius = length * 0.1

        cylinders.append({
            "tree_instance_id": 0,
            "branch_id": i,
            "start": start.tolist(),
            "end": end.tolist(),
            "radius": float(radius),
            "length": float(length),
            "axis": axis.tolist(),
        })

    with open(out_json, "w") as f:
        json.dump(cylinders, f, indent=4)

    print(f"Successfully extracted {len(cylinders)} GT cylinders to {out_json}")
    return cylinders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    extract_cylinders(args.mesh_path, args.out_json)
