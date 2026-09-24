"""Mangrove Geometry Nodes tree generation inside Blender."""

import json
import random
from pathlib import Path

import bmesh
import bpy


def _mangrove_config_path() -> Path:
    # blender_runtime/ -> generators/ -> synthetic_tls/ -> SyntheticPipeline/
    pipeline_root = Path(__file__).resolve().parents[3]
    return pipeline_root / "configs" / "mangrove_config.json"


def generate_from_geometry_nodes(seed, out_obj, out_json):
    """
    Generate a tree by mutating a loaded 'Mangrove tree' Geometry Nodes object,
    converting to mesh, decimating, and exporting leaf-on + leafless OBJs.
    """
    obj = bpy.data.objects.get("Mangrove tree")
    if not obj:
        print("Error: Could not find 'Mangrove tree' object in the provided .blend file.")
        return

    random.seed(seed)

    config_file = _mangrove_config_path()
    config = {
        "decimate_ratio": 0.5,
        "remove_underground": True,
        "randomize_seed_parameters": True,
        "seed_parameters": ["Input_20"],  # UI name for Input_20 is 'Seed'
        "parameters": {},
    }

    for mod in obj.modifiers:
        if mod.type == "NODES":
            for k in mod.keys():
                if (
                    not k.startswith("_")
                    and not k.endswith("_use_attribute")
                    and k not in [
                        "name",
                        "type",
                        "show_viewport",
                        "show_render",
                        "show_in_editmode",
                        "show_on_cage",
                    ]
                ):
                    val = mod[k]
                    if isinstance(val, (int, float, bool)):
                        config["parameters"][k] = val

    if config_file.exists():
        with open(config_file, "r") as f:
            user_config = json.load(f)
            config.update(user_config)
            if "parameters" in user_config:
                config["parameters"].update(user_config["parameters"])
        print(f"Loaded mangrove parameters from {config_file}")
    else:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        print(
            f"Created default config file at {config_file}. "
            "You can edit this to fine-tune the tree parameters."
        )

    for mod in obj.modifiers:
        if mod.type == "NODES":
            for k, v in config["parameters"].items():
                if k in mod:
                    try:
                        if k == "Input_25":
                            mod[k] = False
                            continue

                        if (
                            config["randomize_seed_parameters"]
                            and k in config.get("seed_parameters", [])
                            and isinstance(v, int)
                        ):
                            mod[k] = (v + seed) % 999999
                            print(f"Set {k} = {mod[k]} (Base {v} + Seed {seed})")
                        elif config["randomize_seed_parameters"] and k in ["Input_19", "Input_22"]:
                            mod[k] = v + random.uniform(0, 6.28)
                            print(f"Set {k} = {mod[k]} (Randomized Spin)")
                        else:
                            mod[k] = v
                    except Exception as e:
                        print(f"Could not set {k}: {e}")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Realize instances before convert (Mangrove uses many leaf instances).
    realize_mod = obj.modifiers.new(name="Auto_Realize", type="NODES")
    group = bpy.data.node_groups.new(name="RealizeGroup", type="GeometryNodeTree")
    realize_mod.node_group = group
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")
    realize_node = group.nodes.new("GeometryNodeRealizeInstances")
    group.links.new(input_node.outputs[0], realize_node.inputs[0])
    group.links.new(realize_node.outputs[0], output_node.inputs[0])

    bpy.ops.object.convert(target="MESH")

    decimate_mod = obj.modifiers.new(name="Auto_Decimate", type="DECIMATE")
    decimate_mod.ratio = config["decimate_ratio"]
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    if config["remove_underground"]:
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(obj.data)
        verts_to_delete = [v for v in bm.verts if v.co.z < -0.01]
        if verts_to_delete:
            bmesh.ops.delete(bm, geom=verts_to_delete, context="VERTS")
            print(f"Removed {len(verts_to_delete)} underground vertices.")
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.wm.obj_export(
        filepath=out_obj,
        export_selected_objects=True,
        export_materials=False,
        export_triangulated_mesh=True,
        forward_axis="Y",
        up_axis="Z",
    )
    print(f"Successfully generated Geometry Nodes tree {out_obj}")

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    # Material 0 is Bark. Delete faces that are not Material 0.
    faces_to_delete = [f for f in bm.faces if f.material_index != 0]
    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
        print(f"Removed {len(faces_to_delete)} leaf/twig faces for GT mesh.")
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")

    out_obj_noleaf = out_obj.replace(".obj", "_noleaf.obj")
    bpy.ops.wm.obj_export(
        filepath=out_obj_noleaf,
        export_selected_objects=True,
        export_materials=False,
        export_triangulated_mesh=True,
        forward_axis="Y",
        up_axis="Z",
    )
    print(f"Successfully generated Leafless GT tree {out_obj_noleaf}")

    # Skeleton is filled later by host-side mesh cylinder extraction.
    with open(out_json, "w") as f:
        json.dump([], f, indent=4)
