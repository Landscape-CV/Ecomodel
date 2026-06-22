import trimesh
import numpy as np

mesh = trimesh.load("SyntheticPipeline/output/assets/tree_0000_noleaf.obj")
print(f"Loaded mesh with {len(mesh.faces)} faces")
components = mesh.split(only_watertight=False)
print(f"Split into {len(components)} components")

for i, comp in enumerate(components[:5]):
    extents = comp.extents
    print(f"Component {i}: faces={len(comp.faces)}, extents={extents}")
