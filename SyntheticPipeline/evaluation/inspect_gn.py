import bpy

def inspect_node_group(group_name="Mangrove Gen"):
    group = bpy.data.node_groups.get(group_name)
    if not group:
        print(f"Could not find node group {group_name}")
        return
        
    print(f"\n--- Nodes in {group_name} ---")
    for node in group.nodes:
        print(f"Node: {node.name} (Type: {node.type})")
        
    print(f"\n--- Outputs in {group_name} ---")
    for out in group.outputs:
        print(f"Output: {out.name} (Type: {out.type})")

inspect_node_group("Mangrove Gen")
