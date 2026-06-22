import bpy

obj = bpy.data.objects.get("Mangrove tree")
if obj:
    for mod in obj.modifiers:
        if mod.type == 'NODES':
            node_group = mod.node_group
            if node_group:
                print("--- GEOMETRY NODES INPUTS ---")
                for item in node_group.interface.items_tree:
                    if item.item_type == 'SOCKET':
                        print(f"{item.identifier}: {item.name} ({item.socket_type})")
                print("-----------------------------")
