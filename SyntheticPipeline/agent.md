# SyntheticPipeline — Agent Knowledge Base

Key learnings, architectural decisions, and physics logic from developing the synthetic LiDAR pipeline. Use this alongside `README.md` for operational context.

## 1. Blender Geometry & Memory Constraints

- **Procedural generation memory:** Complex procedural trees (e.g. Mangrove via Geometry Nodes) can OOM during mesh realization in Blender's Python API.
- **Decimate strategy:** `decimate_ratio` of `0.2`–`0.3` works well on 32 GB RAM — enough detail for TLS simulation without crashing.
- **Fail-safe:** `forest_assembler.py` uses a `psutil` memory guard that aborts gracefully above ~90% RAM usage.

## 2. Spatial Orientation & Coordinate Frames

- **Blender scene export:** Assembling scenes as `.obj` caused Blender to rotate the forest 90° (Y-up vs Z-up). Scene export in `forest_assembler.py` uses `.ply` to preserve absolute coordinates.
- **Arbaro export:** Arbaro OBJ output is Y-up. `ArbaroGenerator` applies a +90° X rotation via `trimesh` before downstream use so all assets are Z-up consistent with the rest of the pipeline.

## 3. Forest Assembly Physics

- **Grounding trees:** Placing at `Z = 0` floats or sinks trees depending on mesh origin. Always translate by `-min_z` from the bounding box so roots touch the ground plane.
- **Randomization:**
  - Scale trees between 0.7× and 1.5× with `trimesh.transformations.scale_matrix`.
  - **Gotcha:** GT skeleton cylinders must be scaled too — multiply `radius` and `length`, re-normalize the `axis` vector after the transform matrix.
- **Wind tilt:** `wind_x` / `wind_y` in assembly apply a rotation that tilts the whole forest. Keep defaults at `0.0` for vertical trees unless wind is intentional.

## 4. LiDAR Simulation Physics (Open3D)

- **Crop circle artifacts:** Perfect spherical ray grids on flat ground produce concentric ring artifacts. Gaussian jitter on `theta` and `phi` in `_create_spherical_rays` breaks the symmetry.
- **Intensity model:**
  - Lambertian proxy: `abs(dot(ray_direction, surface_normal))`
  - Distance decay: inverse-square `1/R²`
  - Stored as 16-bit `las.intensity` (0–65535) in `.laz` exports via `laspy`
- **Wind sway during scan:** Per-scan-position wind is perturbed by `wind_sway_std`; displacement scales with `(z / max_height)²` so bases stay fixed.
- **Scan placement:** Scanner positions are random `[X, Y, 1.5]` spread across the scene bounding box, not clustered at the origin.

## 5. Ground Truth Pipeline

- **Per-tree GT:** Both generators produce a leafless `_noleaf.obj`. `mesh_to_gt_cylinders.py` fits cylinders to branch geometry and writes per-tree JSON.
- **Scene GT:** `ForestAssembler` merges transformed per-tree cylinders into `{scene}_gt.json`. `GTParser` flattens to `{scene}_gt.txt` (9 columns per cylinder).
- **Trunk mesh:** `{scene}_gt_mesh.ply` (leafless geometry only) is copied as `{species}_tile_{id}_trunk.ply` for labeling.
- **Point labels:** `label_gt_points.py` uses Open3D `RaycastingScene.compute_distance` against the trunk mesh. Points within `dist_thresh` (default 5 cm) are wood (1), else leaf (0).

## 6. Large-Scale Dataset Generation Strategy

- **Asset pooling:** Running Arbaro/Blender per tree per tile is O(tiles × trees). Instead, generate a pool (e.g. 5–50 variants) once per species, then sample from it for each tile.
- **`generate_test_dataset.py` flow:**
  1. For each `lib/arbaro/trees/*.xml` species
  2. Build pool in `output/temp_assets/`
  3. For each tile: assemble 1 tree → simulate 2–3 scans → export GT → copy to `testdataset/single/`
  4. Write `{species}_tile_{id}_meta.json` with scan positions and noise params
- **Downscale defaults** (in script): `resolution_theta/phi_deg = 0.2`, `voxel_downsample_size = 0.05`, `pool_size = 5`, `tiles_per_batch = 3`. Tweak upward only when RAM allows.

## 7. Benchmarking & Separation Evaluation

- **Preprocessing parity:** `benchmark_separation.py` uses `EcomodelLite` (normalize → CSF ground removal → intensity filter) to match `pipeline_lite.py`.
- **Single-tree leafy tiles:** `SegmenterScanline` is designed for wood-heavy skeletons and drops or mis-segments leafy canopies. For benchmark tiles with one tree, instance IDs are set to all-zero (single instance) after preprocessing instead of running scanline segmentation.
- **Algorithms evaluated:** `SegmentRGI`, `GBSeparation`, `SmartQSM` (optional; needs `thirdparty/SmartQSM` config paths).
- **GT partitioning:** Cylinders with `radius >= trunk_radius_threshold` (default 5 cm) define trunk; smaller radii define canopy. Surface points are sampled from cylinders for KD-tree distance queries.
- **Metrics:** Voxel-level precision, recall, F1, and IoU at `--voxel_size` (default 10 cm), computed separately for trunk and canopy regions.
- **Caching:** `{tile}_instances.npz` caches preprocessed points so algorithm re-runs skip EcomodelLite.
- **Outputs:** Append-safe CSV at `--out_csv`, per-tile logs in `output/logs/`, optional `--visualize` PLY layers and `--plot` summary bar chart. `plot_species_performance.py` groups trunk F1 by species from a results CSV.

## 8. Generator-Specific Notes

### Blender

- Runs headless: `blender -b [blend_file] -P blender_script.py -- --seed ... --height ...`
- Mangrove parameters controlled via `configs/mangrove_config.json` (copy from `.example.json`)
- `inspect_gn.py` at repo root of SyntheticPipeline lists Geometry Nodes socket identifiers when debugging parameter names

### Arbaro

- Requires Java and `lib/arbaro/arbaro_cmd.jar` (install via `scripts/setup_arbaro.py`)
- Two subprocess calls per tree: leaves on (visual) + leaves off (GT skeleton)
- `Scale` XML param is overwritten per tree to match sampled height from `AssetManager`
- Species XML files live in `lib/arbaro/trees/` and are gitignored along with the JAR

## 9. Operational Gotchas

| Issue | Mitigation |
|-------|------------|
| OOM during assembly or benchmark KD-tree | Lower ray resolution, increase voxel downsample, reduce `pool_size` / `tiles_per_batch` |
| Empty asset dir in assembler | Generators must output paired `.obj` + `.json`; assembler skips `*_noleaf.obj` when listing assets |
| SmartQSM benchmark failures | Verify `--sq_dir`, `--sq_py`, `--sq_cfg`; check `output/logs/{tile}.log` |
| Missing `lib/arbaro` | Run `python scripts/setup_arbaro.py` and add species XMLs to `lib/arbaro/trees/` |
| Config not found | Copy `configs/*.example.*` to non-example names; machine-specific paths are gitignored |

## 10. File Layout Quick Reference

```
SyntheticPipeline/
├── run_simulation.py          # Demo: config-driven full pipeline
├── scripts/
│   ├── generate_test_dataset.py
│   ├── label_gt_points.py
│   ├── benchmark_separation.py
│   └── plot_species_performance.py
├── pipeline/                  # asset_manager, forest_assembler, simulator, gt_parser
├── generators/                # blender, arbaro, mesh_to_gt_cylinders
├── evaluation/                # QSM evaluator, visualizers, GN inspectors
├── configs/                   # *.example.json / *.example.xml templates
├── lib/arbaro/                # JAR + trees/*.xml (gitignored, local setup)
├── output/                    # Generated artifacts (gitignored)
└── testdataset/single/        # Benchmark tiles (gitignored)
```
