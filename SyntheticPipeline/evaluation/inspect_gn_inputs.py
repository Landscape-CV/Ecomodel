import bpy

print("\n--- BLENDER INSPECTION START ---")
obj = bpy.data.objects.get("Mangrove tree")
if obj:
    for mod in obj.modifiers:
        if mod.type == 'NODES':
            ng = mod.node_group
            print(f"Node Group: {ng.name}")
            # In Blender 4.0+, interface is used
            if hasattr(ng, "interface"):
                for item in ng.interface.items_tree:
                    if item.item_type == 'SOCKET':
                        print(f"Input: {item.identifier} -> {item.name} (Type: {item.socket_type})")
            else:
                for inp in ng.inputs:
                    print(f"Input: {inp.identifier} -> {inp.name} (Type: {inp.type})")
print("--- BLENDER INSPECTION END ---\n")
