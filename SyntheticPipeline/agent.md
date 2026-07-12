# SyntheticPipeline - Agent Knowledge Base

This document serves as a knowledge base containing key learnings, architectural decisions, and physics-based logic implemented during the development of the Synthetic Pipeline.

## 1. Blender Geometry & Memory Constraints
- **Procedural Generation Memory:** Generating complex procedural trees (like Mangroves) via Blender's Python API requires enormous RAM during mesh realization. 
- **Decimate Strategy:** A `decimate_ratio` of `0.2` to `0.3` was found to be the sweet spot for a 32GB RAM machine. This retains enough geometric detail (4x more than standard 0.05) while preventing Out-Of-Memory (OOM) crashes.
- **Fail-Safe:** Added a `psutil` memory guard in the Python scripts that monitors system RAM and fails gracefully if usage exceeds 90%, preventing catastrophic OS-level freezing or infinite page-file thrashing.

## 2. Spatial Orientation & Coordinate Frames
- **Blender Import Rotation:** Exporting assembled scenes as `.obj` caused Blender to automatically rotate the entire forest 90 degrees sideways (Y-up vs Z-up conversion issue). 
- **Solution:** Switched the final forest mesh export format in `forest_assembler.py` from `.obj` to `.ply`. `.ply` naturally retains its absolute coordinates without auto-rotation on import.

## 3. Forest Assembly Physics
- **Grounding Trees:** Simply placing a tree at `Z = 0.0` causes it to float or sink depending on where Blender set its internal origin point. 
  - *Fix:* Always mathematically compute the lowest vertex of the bounding box (`tree_mesh.bounds[0][2]`) and translate the mesh by `-min_z` so the absolute bottom root always perfectly touches the ground plane.
- **Randomization:** 
  - To prevent identical tree cloning, a 3D scaling matrix (`trimesh.transformations.scale_matrix`) scales trees dynamically between 0.7x and 1.5x.
  - *Gotcha:* When scaling the mesh, the Ground Truth (GT) skeleton must also be scaled! The `radius` and `length` of the skeleton cylinders are manually multiplied by the scale factor, and the directional `axis` vector is explicitly re-normalized after applying the transformation matrix.
- **Wind Tilted Forests:** Be careful with the `wind_x` and `wind_y` parameters. Applying a universal wind vector calculates a rotation matrix that tilts the entire forest sideways. Keep default wind at `0.0` for perfectly vertical growth.

## 4. LiDAR Simulation Physics (Open3D)
- **Crop Circle Artifacts:** Standard Open3D mathematical spherical rays (`theta`, `phi`) are too perfect. Hitting a perfectly flat ground plane results in unnatural, concentric "crop circle" scanning artifacts.
  - *Fix:* Injected tiny random Gaussian jitter (`np.random.uniform`) into the angular grid generation to organically break the concentric rings and simulate real-world LiDAR motor jitter.
- **Physical LiDAR Intensity:**
  - Standard simulated scans only output XYZ. Real LiDAR captures intensity based on surface reflectance.
  - *Lambertian Reflectance Model:* Implemented an intensity proxy using the absolute dot product of the outgoing ray direction and the mesh primitive normal (`abs(dot(ray, normal))`). 
  - *Result:* Flat, thick wood trunks that take "direct hits" reflect bright light (High Intensity). Chaotic, thin canopy leaves that take "glancing hits" reflect scattered light (Low Intensity).
  - *Distance Decay:* Added an inverse-square law (`1/R^2`) distance decay to further approximate realistic signal loss.
- **Data Export (.laz):** Instead of using a `.ply` RGB color hack or stripping data with `.xyz`, the simulator now uses `laspy` to natively export a compressed `.laz` (LASer) file. The 0-1 intensity is scaled back up to a 16-bit integer (0-65535) and natively embedded into the `las.intensity` dimension, matching true professional LiDAR datasets.
- **Scan Distribution:** A realistic survey requires overlapping viewpoints to prevent occlusion. Instead of clustering scanners near the origin `[0,0]`, `simulator_open3d.py` dynamically computes random `[X, Y, 1.5]` tripod coordinates scattered uniformly across the entire bounding box area.

## 5. Large-Scale Dataset Generation Strategy
- **Asset Pooling Optimization:** When generating thousands of tiles (e.g. 125 tiles per vegetation type), running the external procedural generator (like Java Arbaro) for every single tree in every tile is extremely slow (O(Tiles * Trees) subprocess calls).
- **Solution:** Generate a large pool of tree variants upfront (e.g., 50 variations of a given species) just once per vegetation type. The `ForestAssembler` can then construct endless unique scene combinations by randomly sampling from this pre-generated pool, applying random coordinate placements, random Z-axis yaw rotations, and random scale factors.

## 6. Benchmarking & Downscaling
- **Memory Footprint:** Running benchmarks on massive point clouds (e.g. 0.05-degree rays and 0.01m voxels generating 100MB+ per tile) easily causes 32GB RAM machines to OOM during KDTree construction and scanline processing.
- **Downscale Strategy:** For development and regular benchmarking, `resolution_theta_deg` and `resolution_phi_deg` are increased to `0.2`, and `voxel_downsample_size` is increased to `0.05` (5cm). This shrinks the point clouds by over ~16x and allows fast processing (seconds per tile instead of minutes/hours) while maintaining structural tree integrity for algorithms.
- **Batching:** `generate_test_dataset.py` restricts tree pools and tile batches to ~10-20 to ensure generation takes minutes rather than days.
