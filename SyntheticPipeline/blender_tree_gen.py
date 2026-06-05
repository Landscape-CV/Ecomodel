import bpy
import json
import sys
import argparse
import random
import math
import mathutils

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_cylinder(start, end, radius):
    v = end - start
    length = v.length
    if length < 1e-4:
        return None
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length)
    cyl = bpy.context.active_object
    z_axis = mathutils.Vector((0, 0, 1))
    v_norm = v.normalized()
    quat = z_axis.rotation_difference(v_norm)
    cyl.rotation_mode = 'QUATERNION'
    cyl.rotation_quaternion = quat
    cyl.location = start + v / 2.0
    return cyl

def generate_tree_native(seed, height, out_obj, out_json):
    clear_scene()
    random.seed(seed)
    skeleton_data = []
    cylinders = []
    branch_counter = 0
    
    def build_branch(start_pt, direction, length, radius, level):
        nonlocal branch_counter
        if level == 0 or length < 0.2: return
        end_pt = start_pt + direction * length
        cyl_obj = create_cylinder(start_pt, end_pt, radius)
        if cyl_obj:
            cylinders.append(cyl_obj)
        skeleton_data.append({
            "branch_id": branch_counter, "start": [start_pt.x, start_pt.y, start_pt.z],
            "end": [end_pt.x, end_pt.y, end_pt.z], "radius": radius,
            "length": length, "axis": [direction.x, direction.y, direction.z]
        })
        branch_counter += 1
        num_branches = random.randint(1, 3) if level > 1 else 0
        for _ in range(num_branches):
            angle_x, angle_y = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
            rot = mathutils.Euler((angle_x, angle_y, 0), 'XYZ').to_matrix()
            new_dir = rot @ direction
            new_length, new_radius = length * random.uniform(0.6, 0.8), radius * random.uniform(0.5, 0.7)
            spawn_t = random.uniform(0.5, 1.0)
            spawn_pt = start_pt + direction * (length * spawn_t)
            build_branch(spawn_pt, new_dir, new_length, new_radius, level - 1)

    build_branch(mathutils.Vector((0, 0, 0)), mathutils.Vector((0, 0, 1)), height * 0.4, height * 0.05, 4)
    with open(out_json, 'w') as f:
        json.dump(skeleton_data, f, indent=4)
    if cylinders:
        bpy.ops.object.select_all(action='DESELECT')
        for c in cylinders: c.select_set(True)
        bpy.context.view_layer.objects.active = cylinders[0]
        bpy.ops.object.join()
        bpy.ops.wm.obj_export(filepath=out_obj, export_selected_objects=True, export_materials=False, export_triangulated_mesh=True)
        print(f"Generated native L-System tree {out_obj}")

def generate_from_geometry_nodes(seed, out_obj, out_json):
    # Find the target object
    obj = bpy.data.objects.get("Mangrove tree")
    if not obj:
        print("Error: Could not find 'Mangrove tree' object in the provided .blend file.")
        return
        
    random.seed(seed)
    
    # Randomize the Geometry Nodes modifier inputs to get unique trees
    # We mutate integer inputs as they usually represent Random Seeds
    for mod in obj.modifiers:
        if mod.type == 'NODES':
            for k in mod.keys():
                if type(mod[k]) == int and not k.endswith('_use_attribute') and k not in ['_RNA_UI', 'name', 'type', 'show_viewport', 'show_render', 'show_in_editmode', 'show_on_cage']:
                    mod[k] = random.randint(0, 999999)
                    print(f"Set {k} to {mod[k]} for randomization.")

    # Select only the tree object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Convert Geometry Nodes output (with all its instances) into a single raw MESH
    bpy.ops.object.convert(target='MESH')
    
    # NOW we can safely add the Decimate modifier to reduce poly count from 3GB down to ~15MB
    decimate_mod = obj.modifiers.new(name="Auto_Decimate", type='DECIMATE')
    decimate_mod.ratio = 0.005  # Reduce to 0.5% of original polygons (Mangroves have millions of leaves)
    
    # Apply the decimation immediately
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)
    
    # Export the compressed mesh
    bpy.ops.wm.obj_export(
        filepath=out_obj,
        export_selected_objects=True,
        export_materials=False,
        export_triangulated_mesh=True
    )
    print(f"Successfully generated Geometry Nodes tree {out_obj}")
    
    # Note: Extracting analytical skeletons from a closed Geometry Nodes setup 
    # requires the GN tree to explicitly output curves or attribute data.
    # For now, we write a dummy JSON or a warning for the train track.
    with open(out_json, 'w') as f:
        json.dump([], f, indent=4)
        print("Warning: GT skeleton extraction from arbitrary GN trees requires node-level curve outputs.")

if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=float, default=10.0)
    parser.add_argument("--out_obj", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args, _ = parser.parse_known_args(argv)
    
    # Check if we are running inside a loaded GN blend file
    if "Mangrove tree" in bpy.data.objects:
        print("Found 'Mangrove tree' object! Using Track A: Geometry Nodes generator.")
        generate_from_geometry_nodes(args.seed, args.out_obj, args.out_json)
    else:
        print("No GN file loaded. Falling back to native L-System generator.")
        generate_tree_native(args.seed, args.height, args.out_obj, args.out_json)
