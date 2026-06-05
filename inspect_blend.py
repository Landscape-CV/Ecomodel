import bpy

print("\n--- BLENDER INSPECTION START ---")
print("Objects in scene:")
for obj in bpy.context.scene.objects:
    print(f"- {obj.name} (Type: {obj.type})")
    for mod in obj.modifiers:
        print(f"  * Modifier: {mod.name} (Type: {mod.type})")
        if mod.type == 'NODES':
            print(f"    Node Group: {mod.node_group.name if mod.node_group else 'None'}")
            print(f"    Inputs:")
            for k in mod.keys():
                if k not in ['_RNA_UI', 'name', 'type', 'show_viewport', 'show_render', 'show_in_editmode', 'show_on_cage']:
                    print(f"      - {k} : {mod[k]}")
print("--- BLENDER INSPECTION END ---\n")
