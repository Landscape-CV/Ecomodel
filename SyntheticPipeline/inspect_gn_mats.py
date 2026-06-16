import bpy

print("\n--- BLENDER MATERIALS START ---")
obj = bpy.data.objects.get("Mangrove tree")
if obj:
    for i, slot in enumerate(obj.material_slots):
        print(f"Material {i}: {slot.name}")
print("--- BLENDER MATERIALS END ---\n")
