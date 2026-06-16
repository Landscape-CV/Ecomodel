"""
This script generates 3D tree models using Blender's Python API (bpy).
It supports two methods of generation:
1. Native L-System: Procedurally generates a tree using recursive branching if no specific Blender file is loaded.
2. Geometry Nodes: Mutates properties of an existing 'Mangrove tree' Geometry Node setup in a provided .blend file to create variations, then decimates the mesh to optimize polygon count.

It outputs the resulting 3D model as an .obj file and, for the L-System, a .json file containing the tree's skeleton data.
"""
import bpy
import json
import sys
import argparse
import random
import math
import mathutils

def clear_scene():
    """
    Clears all existing objects from the current Blender scene.
    This ensures a clean slate before generating a new tree.
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_cylinder(start, end, radius):
    """
    Creates a cylinder mesh in Blender between two 3D points to represent a tree branch.
    
    Args:
        start (mathutils.Vector): The starting point of the cylinder.
        end (mathutils.Vector): The ending point of the cylinder.
        radius (float): The radius (thickness) of the cylinder.
        
    Returns:
        bpy.types.Object: The created cylinder object, or None if the length is too small.
    """
    v = end - start
    length = v.length
    if length < 1e-4:
        return None
    
    # Add a basic cylinder primitive with the required length and radius
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length)
    cyl = bpy.context.active_object
    
    # Calculate the rotation required to align the cylinder along the vector 'v'
    z_axis = mathutils.Vector((0, 0, 1))
    v_norm = v.normalized()
    quat = z_axis.rotation_difference(v_norm) # Rotation from Z-axis to the target direction
    
    # Apply rotation and position
    cyl.rotation_mode = 'QUATERNION'
    cyl.rotation_quaternion = quat
    # The default cylinder origin is at its center, so position it halfway between start and end
    cyl.location = start + v / 2.0
    
    return cyl

def generate_tree_native(seed, height, out_obj, out_json):
    """
    Procedurally generates a tree using a recursive branching algorithm (L-System style).
    
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
        """
        Recursive helper function to build branches.
        
        Args:
            start_pt (mathutils.Vector): Origin point of the current branch.
            direction (mathutils.Vector): Direction vector of the branch.
            length (float): Length of the branch.
            radius (float): Thickness of the branch.
            level (int): Recursion depth remaining. When 0, stop branching.
        """
        nonlocal branch_counter
        # Stop condition: recursion limit reached or branch is too short
        if level == 0 or length < 0.2: return
        
        end_pt = start_pt + direction * length
        cyl_obj = create_cylinder(start_pt, end_pt, radius)
        
        if cyl_obj:
            cylinders.append(cyl_obj)
            
        # Store metadata about the branch for ground truth skeleton extraction
        skeleton_data.append({
            "branch_id": branch_counter, "start": [start_pt.x, start_pt.y, start_pt.z],
            "end": [end_pt.x, end_pt.y, end_pt.z], "radius": radius,
            "length": length, "axis": [direction.x, direction.y, direction.z]
        })
        branch_counter += 1
        
        # Determine how many child branches will grow from this branch (0 to 3)
        num_branches = random.randint(1, 3) if level > 1 else 0
        
        for _ in range(num_branches):
            # Randomize rotation angles for the child branch
            angle_x, angle_y = random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)
            rot = mathutils.Euler((angle_x, angle_y, 0), 'XYZ').to_matrix()
            new_dir = rot @ direction # Apply rotation to current direction
            
            # Shrink length and radius for child branches
            new_length, new_radius = length * random.uniform(0.6, 0.8), radius * random.uniform(0.5, 0.7)
            
            # Decide where along the parent branch the child will spawn (top half)
            spawn_t = random.uniform(0.5, 1.0)
            spawn_pt = start_pt + direction * (length * spawn_t)
            
            # Recursively build the child branch
            build_branch(spawn_pt, new_dir, new_length, new_radius, level - 1)

    # Start the recursive generation with the trunk
    build_branch(mathutils.Vector((0, 0, 0)), mathutils.Vector((0, 0, 1)), height * 0.4, height * 0.05, 4)
    
    # Save the skeleton structure to JSON
    with open(out_json, 'w') as f:
        json.dump(skeleton_data, f, indent=4)
        
    # Join all separate cylinder objects into a single mesh and export
    if cylinders:
        bpy.ops.object.select_all(action='DESELECT')
        for c in cylinders: c.select_set(True)
        bpy.context.view_layer.objects.active = cylinders[0]
        bpy.ops.object.join() # Merge meshes
        bpy.ops.wm.obj_export(filepath=out_obj, export_selected_objects=True, export_materials=False, export_triangulated_mesh=True, forward_axis='Y', up_axis='Z')
        print(f"Generated native L-System tree {out_obj}")

def generate_from_geometry_nodes(seed, out_obj, out_json):
    """
    Generates a tree by modifying parameters of an existing 'Mangrove tree' object 
    built with Geometry Nodes, converting it to a mesh, and decimating it for optimization.
    
    Args:
        seed (int): Random seed offset to perturb geometry node 'seed' parameters.
        out_obj (str): Filepath to save the exported .obj model.
        out_json (str): Filepath to save skeleton data (currently outputs empty JSON for this mode).
    """
    import os
    import bmesh

    # Find the target object
    obj = bpy.data.objects.get("Mangrove tree")
    if not obj:
        print("Error: Could not find 'Mangrove tree' object in the provided .blend file.")
        return
        
    random.seed(seed)
    
    # --------------------------------------------------------------------------------
    # 1. PARAMETER FINE-TUNING & CONFIG FILE
    # Previously, randomizing ALL integer inputs from 0-999999 caused artifacts and 
    # leafless trees because it randomized structural values (branch counts, resolutions, etc).
    # We now dump all parameters to a config file so you can fine-tune them safely.
    # --------------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "mangrove_config.json")
    config = {
        "decimate_ratio": 0.5,
        "remove_underground": True,
        "randomize_seed_parameters": True,
        "seed_parameters": ["Input_20"], # The UI name for Input_20 is 'Seed'
        "parameters": {}
    }
    
    # Extract current parameters from modifier to create a template
    for mod in obj.modifiers:
        if mod.type == 'NODES':
            for k in mod.keys():
                if not k.startswith('_') and not k.endswith('_use_attribute') and k not in ['name', 'type', 'show_viewport', 'show_render', 'show_in_editmode', 'show_on_cage']:
                    val = mod[k]
                    # We only store standard types (ints, floats, booleans)
                    if isinstance(val, (int, float, bool)):
                        config["parameters"][k] = val
                        
    # Load user config if it exists, otherwise create it
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            # Update our default config with the user's saved config
            config.update(user_config)
            # Make sure parameters dict is updated properly
            if "parameters" in user_config:
                config["parameters"].update(user_config["parameters"])
        print(f"Loaded mangrove parameters from {config_file}")
    else:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created default config file at {config_file}. You can edit this to fine-tune the tree parameters.")

    # Apply parameters to the geometry node modifier
    for mod in obj.modifiers:
        if mod.type == 'NODES':
            for k, v in config["parameters"].items():
                if k in mod:
                    try:
                        # Force Proxy off so we get full geometry
                        if k == "Input_25":
                            mod[k] = False
                            continue
                            
                        # If the parameter is in our seed list, add our CLI seed to it
                        if config["randomize_seed_parameters"] and k in config.get("seed_parameters", []) and isinstance(v, int):
                            mod[k] = (v + seed) % 999999
                            print(f"Set {k} = {mod[k]} (Base {v} + Seed {seed})")
                        # Randomize spins (Input_19: Root spin, Input_22: Branching spin)
                        elif config["randomize_seed_parameters"] and k in ["Input_19", "Input_22"]:
                            mod[k] = v + random.uniform(0, 6.28)
                            print(f"Set {k} = {mod[k]} (Randomized Spin)")
                        else:
                            mod[k] = v
                            # print(f"Set {k} = {v}") # Muted to reduce spam
                    except Exception as e:
                        print(f"Could not set {k}: {e}")

    # Select only the tree object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # --------------------------------------------------------------------------------
    # 1.5 REALIZE INSTANCES 
    # Mangrove Gen uses ~81,000 instances for leaves. Convert to mesh drops instances.
    # We must explicitly add a Realize Instances modifier before converting!
    # --------------------------------------------------------------------------------
    realize_mod = obj.modifiers.new(name="Auto_Realize", type='NODES')
    group = bpy.data.node_groups.new(name="RealizeGroup", type='GeometryNodeTree')
    realize_mod.node_group = group
    group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    input_node = group.nodes.new('NodeGroupInput')
    output_node = group.nodes.new('NodeGroupOutput')
    realize_node = group.nodes.new('GeometryNodeRealizeInstances')
    group.links.new(input_node.outputs[0], realize_node.inputs[0])
    group.links.new(realize_node.outputs[0], output_node.inputs[0])
    
    # Convert Geometry Nodes output (with all its realized instances) into a single raw MESH
    bpy.ops.object.convert(target='MESH')
    
    # --------------------------------------------------------------------------------
    # 2. OPTIMIZATION: DECIMATION
    # --------------------------------------------------------------------------------
    decimate_mod = obj.modifiers.new(name="Auto_Decimate", type='DECIMATE')
    decimate_mod.ratio = config["decimate_ratio"]
    
    # Apply the decimation immediately
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)
    
    # --------------------------------------------------------------------------------
    # 3. MESH CLEANUP: REMOVE UNDERGROUND VERTICES
    # --------------------------------------------------------------------------------
    if config["remove_underground"]:
        # Use bmesh to accurately remove any vertices below Z=0
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        
        # Find all vertices below the ground plane
        verts_to_delete = [v for v in bm.verts if v.co.z < -0.01] # slightly below 0 to keep ground contact
        
        if verts_to_delete:
            bmesh.ops.delete(bm, geom=verts_to_delete, context='VERTS')
            print(f"Removed {len(verts_to_delete)} underground vertices.")
            
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Export the compressed mesh
    bpy.ops.wm.obj_export(
        filepath=out_obj,
        export_selected_objects=True,
        export_materials=False,
        export_triangulated_mesh=True,
        forward_axis='Y',
        up_axis='Z'
    )
    print(f"Successfully generated Geometry Nodes tree {out_obj}")
    
    # Note: Extracting analytical skeletons from a closed Geometry Nodes setup 
    # requires the GN tree to explicitly output curves or attribute data.
    with open(out_json, 'w') as f:
        json.dump([], f, indent=4)
        print("Warning: GT skeleton extraction from arbitrary GN trees requires node-level curve outputs.")

if __name__ == "__main__":
    # Handle arguments passed via Blender command line (arguments after '--')
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
    
    # Check if we are running inside a loaded GN blend file that contains a "Mangrove tree"
    if "Mangrove tree" in bpy.data.objects:
        print("Found 'Mangrove tree' object! Using Track A: Geometry Nodes generator.")
        generate_from_geometry_nodes(args.seed, args.out_obj, args.out_json)
    else:
        print("No GN file loaded. Falling back to native L-System generator.")
        generate_tree_native(args.seed, args.height, args.out_obj, args.out_json)
