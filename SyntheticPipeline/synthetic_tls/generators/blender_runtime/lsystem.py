"""Native recursive L-system tree generation inside Blender."""

import json
import random

import bpy
import mathutils

from common import clear_scene, create_cylinder


def generate_tree_native(seed, height, out_obj, out_json):
    """
    Procedurally generate a tree using a recursive branching algorithm (L-System style).

    Args:
        seed (int): Random seed for reproducible generation.
        height (float): Target height scale for the tree.
        out_obj (str): Filepath to save the exported .obj model.
        out_json (str): Filepath to save the exported skeleton data as .json.
    """
    clear_scene()
    random.seed(seed)
    skeleton_data = []
    cylinders = []
    branch_counter = 0

    def build_branch(start_pt, direction, length, radius, level):
        nonlocal branch_counter
        if level == 0 or length < 0.2:
            return

        end_pt = start_pt + direction * length
        cyl_obj = create_cylinder(start_pt, end_pt, radius)

        if cyl_obj:
            cylinders.append(cyl_obj)

        skeleton_data.append({
            "branch_id": branch_counter,
            "start": [start_pt.x, start_pt.y, start_pt.z],
            "end": [end_pt.x, end_pt.y, end_pt.z],
            "radius": radius,
            "length": length,
            "axis": [direction.x, direction.y, direction.z],
        })
        branch_counter += 1

        num_branches = random.randint(1, 3) if level > 1 else 0

        for _ in range(num_branches):
            angle_x, angle_y = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
            rot = mathutils.Euler((angle_x, angle_y, 0), "XYZ").to_matrix()
            new_dir = rot @ direction

            new_length = length * random.uniform(0.6, 0.8)
            new_radius = radius * random.uniform(0.5, 0.7)

            spawn_t = random.uniform(0.5, 1.0)
            spawn_pt = start_pt + direction * (length * spawn_t)

            build_branch(spawn_pt, new_dir, new_length, new_radius, level - 1)

    build_branch(
        mathutils.Vector((0, 0, 0)),
        mathutils.Vector((0, 0, 1)),
        height * 0.4,
        height * 0.05,
        4,
    )

    with open(out_json, "w") as f:
        json.dump(skeleton_data, f, indent=4)

    if cylinders:
        bpy.ops.object.select_all(action="DESELECT")
        for c in cylinders:
            c.select_set(True)
        bpy.context.view_layer.objects.active = cylinders[0]
        bpy.ops.object.join()
        bpy.ops.wm.obj_export(
            filepath=out_obj,
            export_selected_objects=True,
            export_materials=False,
            export_triangulated_mesh=True,
            forward_axis="Y",
            up_axis="Z",
        )
        print(f"Generated native L-System tree {out_obj}")
