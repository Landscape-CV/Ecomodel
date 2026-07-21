# Synthetic LiDAR Pipeline

This submodule generates synthetic forest scenes, simulates Terrestrial Laser Scanner (TLS) point clouds, and exports structural ground truth (GT) cylinder skeletons. It is used both for interactive demo runs and for building benchmark datasets that evaluate wood/leaf separation algorithms in the parent PyTLidar project.

## Pipeline Overview

```
Tree Generator (Blender or Arbaro)
        ↓
AssetManager  →  pool of .obj meshes + per-tree GT JSON
        ↓
ForestAssembler  →  scene .ply + scene GT JSON (+ trunk mesh)
        ↓
Open3DSimulator  →  merged .laz scan with intensity
        ↓
GTParser  →  scene-level GT cylinders (.txt)
```

Optional downstream steps (benchmark workflow):

```
label_gt_points.py  →  per-point wood/leaf labels (.npy)
benchmark_separation.py  →  metrics CSV + visualizations
plot_species_performance.py  →  per-species comparison charts
```

## Features

- **Generator agnostic:** `BlenderGenerator` (Geometry Nodes / Mangrove preset) and `ArbaroGenerator` (Java Weber–Penn procedural trees) share a common `BaseTreeGenerator` interface.
- **Asset pooling:** Generate a tree variant pool once, then assemble many unique scenes by sampling, rotating, scaling, and placing trees.
- **Forest assembly:** Grounds trees on `Z = 0`, applies random scale (0.7–1.5×) and yaw, optionally tilts for wind, and propagates GT skeleton transforms.
- **LiDAR simulation:** Open3D raycasting with angular jitter, Lambertian intensity, distance decay, distance noise, beam-divergence proxy, wind sway, and voxel downsampling.
- **Ground truth:** Per-tree cylinder skeletons (from leafless meshes) are merged into scene-level JSON/TXT; trunk-only meshes support point labeling.
- **Benchmarking:** Evaluates `SegmentRGI`, `GBSeparation`, and `SmartQSM` against synthetic GT using the same `EcomodelLite` preprocessing as production.

## Directory Structure

| Path | Purpose |
|------|---------|
| `run_simulation.py` | End-to-end demo orchestrator (assets → scene → scan → GT) |
| `configs/` | Example configs; copy `.example` files to machine-local JSON/XML |
| `pipeline/` | Core engines |
| `generators/` | Blender and Arbaro tree generators + `mesh_to_gt_cylinders.py` |
| `scripts/` | Dataset generation, labeling, benchmarking, and plotting |
| `evaluation/` | Visualization, QSM evaluation, and Blender GN inspection utilities |
| `tests/` | `pytest` suite for assembler and generator interfaces |
| `lib/` | Downloaded Arbaro JAR and species XML templates (gitignored) |
| `output/` | Generated assets, scenes, point clouds, benchmark results (gitignored) |
| `testdataset/` | Generated benchmark tiles (gitignored) |
| `inspect_gn.py` | Small Blender helper to list Geometry Nodes inputs on the Mangrove tree |

### `pipeline/`

- `asset_manager.py` — Parallel tree generation via any `BaseTreeGenerator`
- `forest_assembler.py` — Scene mesh (`.ply`) + global GT JSON + trunk mesh (`.ply`)
- `simulator_open3d.py` — TLS raycaster; exports `.laz` with 16-bit intensity
- `simulator_base.py` — Shared simulator interface
- `gt_parser.py` — Converts scene GT JSON to cylinder `.txt` for evaluation

### `scripts/`

- `setup_arbaro.py` — Downloads and extracts Arbaro 1.9.8 into `lib/arbaro/`
- `generate_test_dataset.py` — Builds per-species single-tree benchmark tiles
- `label_gt_points.py` — Labels scan points as wood (1) or leaf (0) from trunk mesh distance
- `benchmark_separation.py` — Runs separation algorithms and writes metrics CSV
- `benchmark_qsm.py` — Compares TreeQSM and SmartQSM under leaf-on, RGI, and oracle inputs
- `plot_species_performance.py` — Bar charts of trunk metrics grouped by species
- `plot_benchmark_qsm.py` — Plots successful QSM benchmark rows

### `evaluation/`

- `evaluator.py` — `QSMEvaluator` for matching predicted vs GT cylinder instances
- `visualize_pipeline.py` — Overlay LAZ points with GT cylinder meshes
- `render_single_tree.py`, `render_verification.py` — Render checks for generated assets
- `inspect_gn*.py` — Blender Geometry Nodes debugging helpers

## Setup

### 1. Python dependencies

From the parent PyTLidar environment:

- `open3d`, `trimesh`, `laspy[lazrs]`, `psutil`, `pytest`
- Benchmarking also needs `pandas`, `scipy`, `matplotlib`, `seaborn`

### 2. Pipeline config

```bash
cp configs/pipeline_config.example.json configs/pipeline_config.json
```

Edit paths such as `blender_path` and `blend_file`. For Mangrove trees, also copy:

```bash
cp configs/mangrove_config.example.json configs/mangrove_config.json
```

### 3. Arbaro (for dataset generation)

Arbaro is not committed to git. Install it once:

```bash
python scripts/setup_arbaro.py
```

Place species XML templates in `lib/arbaro/trees/*.xml`. `generate_test_dataset.py` iterates over every XML in that folder.

For `run_simulation.py` with Arbaro, copy and customize:

```bash
cp configs/arbaro.example.xml configs/arbaro.xml
```

Then set `"generator": { "type": "arbaro", ... }` in `pipeline_config.json`.

## Quickstart: Demo Run

Runs the full pipeline with settings from `configs/pipeline_config.json`:

```bash
python run_simulation.py
python run_simulation.py --config configs/my_custom_config.json
```

Outputs land under `output/assets`, `output/scenes`, and `output/pointclouds` by default.

## Quickstart: Benchmark Dataset

### 1. Generate tiles

Reads every `lib/arbaro/trees/*.xml`, builds a small tree pool per species, assembles single-tree scenes, simulates 2–3 TLS scans, and writes to `testdataset/single/`:

```bash
python scripts/generate_test_dataset.py
```

Default generation uses downscaled scan settings to keep memory and runtime manageable:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `pool_size` | 5 | Tree variants per species |
| `tiles_per_batch` | 3 | Scenes per species |
| `area_size` | 8 m | Plot extent |
| `resolution_theta/phi_deg` | 0.2° | ~16× fewer rays than production 0.05° |
| `voxel_downsample_size` | 0.05 m | 5 cm voxels |

### 2. Label point clouds

Maps each LAZ point to wood or leaf by distance to the trunk GT mesh (`*_trunk.ply`):

```bash
python scripts/label_gt_points.py --dataset_dir testdataset/single
```

### 3. Run the benchmark

Applies `EcomodelLite` ground removal and intensity filtering, then scores trunk/canopy voxel metrics:

```bash
python scripts/benchmark_separation.py \
  --dataset_dir testdataset/single \
  --out_csv output/benchmark_results_single.csv \
  --num_workers 3
```

### 4. Plot species breakdown (optional)

```bash
python scripts/plot_species_performance.py
```

Or pass `--plot` to `benchmark_separation.py` for an algorithm-level summary chart.

## Dataset File Convention

Each tile in `testdataset/single/` uses the prefix `{species}_tile_{id}`:

| File | Description |
|------|-------------|
| `*_scan.laz` | Simulated TLS point cloud with intensity |
| `*_trunk.ply` | Leafless trunk/branch mesh for labeling |
| `*_gt.json` / `*_gt.txt` | Scene cylinder skeleton (10 columns per row) |
| `*_meta.json` | Scan positions, noise params, tile metadata |
| `*_labels.npy` | Binary wood/leaf labels (created by `label_gt_points.py`) |

Current GT cylinder format: `[start_x, start_y, start_z, radius, axis_x, axis_y, axis_z, length, tree_instance_id, branch_id]`. The benchmark also accepts legacy 9-column files. `branch_id` is a mesh-component identifier, not topological branch order.

## Generator Configuration

### Blender / Mangrove (`mangrove_config.json`)

Key parameters when using `BlenderGenerator`:

- `"decimate_ratio"` (e.g. `0.3`) — Polygon reduction; `0.2–0.3` is a practical range on 32 GB RAM
- `"remove_underground"` — Strips vertices below `Z = 0`
- `"randomize_seed_parameters"` / `"seed_parameters"` — Per-tree Geometry Nodes variation
- `"parameters"` — Direct mapping of Blender Geometry Nodes inputs (`Input_2` … `Input_25`)

See `configs/mangrove_config.example.json` for the full Mangrove input reference.

### Arbaro (`arbaro.xml` or per-species XML in `lib/arbaro/trees/`)

```json
"generator": {
  "type": "arbaro",
  "arbaro_jar_path": "SyntheticPipeline/lib/arbaro/arbaro_cmd.jar",
  "xml_template": "SyntheticPipeline/configs/arbaro.xml",
  "java_path": "java",
  "workers": 2
}
```

The pipeline injects a random seed, sets `Scale` from the target height, and runs Arbaro twice per tree: once with leaves (visual mesh) and once with `Leaves = 0` (skeleton mesh for GT extraction). Meshes are rotated from Arbaro's Y-up output to Z-up.

Notable Arbaro XML parameters: `Scale`, `ScaleV`, `BaseSize`, `Ratio`, `RatioPower`, `Flare`, `Leaves`, and per-level branch settings (`0Branches`, `0DownAngle`, `1Length`, etc.). See `configs/arbaro.example.xml` and the Weber & Penn tree paper for full definitions.

## Benchmarking Reference

`benchmark_separation.py` evaluates wood/leaf separation against GT skeletons sampled from cylinder surfaces. Metrics are computed in voxel space separately for **trunk** (radius ≥ threshold) and **canopy** regions.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithms` | all three | `SegmentRGI`, `GBSeparation`, `SmartQSM` |
| `--trunk_radius_threshold` | `0.05` | Cylinder radius (m) separating trunk from canopy |
| `--voxel_size` | `0.1` | Voxel size (m) for precision/recall/F1/IoU |
| `--num_workers` | `cpu_count - 1` | Parallel tile processing |
| `--sample_n` | all tiles | Random subset for quick runs |
| `--visualize` | off | Export colored GT vs prediction `.ply` layers |
| `--plot` | off | Write `output/visualizations/benchmark_comparison.png` |
| `--sq_dir`, `--sq_py`, `--sq_cfg` | SmartQSM paths | Required for SmartQSM evaluation |

Preprocessing mirrors `pipeline_lite.py`: normalize → CSF ground removal → intensity filter. For single-tree leafy tiles, instance segmentation is bypassed (all points assigned to one tree) because `SegmenterScanline` is tuned for wood-only skeletons.

Per-tile logs are written to `output/logs/`. Instance caches (`*_instances.npz`) speed up re-runs.

## QSM Reconstruction Benchmark

`benchmark_qsm.py` measures downstream reconstruction quality, separately from wood/leaf classification quality. Every synthetic single-tree tile uses the same production preprocessing (normalization, CSF ground removal, and intensity filtering), followed by:

- `leaf_on` — no leaf separation
- `rgi` — production `EcomodelLite` SegmentRGI parameters
- `oracle_wood` — synthetic GT labels; an upper bound, not a deployable method
- `gbseparation` — optional geometry-only separator (`--include-gbseparation`)

TreeQSM uses the same cover-set → segmentation → cylinder path as `gui/pipeline_lite.py`. SmartQSM uses its recommended `LEAFON` config for leaf-on points and `LEAFOFF` config for separated/oracle points.

For a conservative run below 16 GB:

```bash
python scripts/benchmark_qsm.py \
  --dataset-dir testdataset/single \
  --sample-n 3 \
  --qsm-voxel-size 0.08 \
  --max-points 400000 \
  --memory-limit-gb 15 \
  --treeqsm-timeout 900 \
  --smartqsm-timeout 900
```

### Metric definition (primary): high-res → low-poly abstraction

The primary question is: **given a high-resolution wood model, how close is the QSM to an optimal low-poly cylinder abstraction?**

All conditions (`leaf_on`, `rgi`, `oracle_wood`, …) are scored the same way. The condition only changes the **input cloud** used to build the QSM; the reference wood target is always the leafless high-res model.

**Reference targets**

| Symbol | Source | Role |
|--------|--------|------|
| Dense wood \(W\) | `*_trunk.ply` surface samples (fallback: labeled LAZ wood points) | True high-res wood surface |
| Coarse wood \(W_c\) | \(W\) voxel-downsampled at `--abstraction-target-voxel` (default **0.08 m**) | Proxy for structure a low-poly model should keep |
| Predicted cylinders \(C\) | TreeQSM / SmartQSM output, starts translated back to world coordinates | Low-poly abstraction |
| Cylinder surface samples \(S\) | Lateral surfaces of \(C\), area-weighted (`--surface-samples`) | Discrete QSM surface |

**Distance**

For a point \(p\) and cylinder set \(C\), \(d(p, C)\) is the unsigned distance to the nearest finite cylinder **surface** (axis segment clipped to cylinder length; radial distance \(|r_{\mathrm{point}} - r_{\mathrm{cyl}}|\)).

**Primary scores** (tolerance \(\tau =\) `--distance-tolerance`, default **0.05 m**)

| Column | Definition | Interpretation |
|--------|------------|----------------|
| `Whole_Precision` | \(\lvert\{ s \in S : d(s, W) \le \tau \}\rvert / \lvert S\rvert\) | Fraction of QSM surface that stays near true wood (**does not invent** geometry) |
| `Whole_Recall` | \(\lvert\{ w \in W_c : d(w, C) \le \tau \}\rvert / \lvert W_c\rvert\) | Fraction of the **coarse** wood support covered by cylinders |
| `Whole_F1` | Harmonic mean of precision and recall | Overall abstraction fidelity |
| `Whole_IoU` | \(F1 / (2 - F1)\) | F1 rewritten as an IoU-like score |
| `Whole_MeanDist_m` | \(\mathrm{mean}_{w \in W_c} d(w, C)\) | Average wood→model distance on the low-poly support |
| `Whole_MedianDist_m` | median of those distances | Robust “how far from optimal fit” |
| `Whole_P90Dist_m` | 90th percentile of those distances | Tail / missed branches |

`MetricTarget` in the CSV records which wood sources were used (e.g. `hires_wood_abstraction:trunk_mesh+recall:trunk_mesh_voxel_0.08m`).

**Why asymmetric targets?** Precision uses dense \(W\) so floating or oversized cylinders are penalized against the real surface. Recall and distances use coarsened \(W_c\) so hair-thin twigs that no reasonable low-poly model would keep do not dominate completeness. Setting `--abstraction-target-voxel 0` makes recall use dense \(W\) as well.

**Secondary scores (optional):** `CylGT_*` plus radius-split `Trunk_*` / `Branch_*` compare predicted cylinders to weak mesh-OBB cylinder GT from `*_gt.txt`. These are diagnostics only—not the wood-only abstraction score. Disable with `--no-keep-cyl-gt-metrics`.

Defaults are sequential: do not run multiple full TreeQSM/SmartQSM jobs concurrently on a 16 GB machine. Each worker is thread-pinned, TreeQSM plots and distance reports are disabled, and memory/time limit failures are recorded as statuses rather than zero scores.

Runs resume at `(file, condition, algorithm, config)` granularity. Successful cylinder models are cached under `cylinders/` for re-scoring.

Useful options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--preprocess-voxel-size` | `0` | Optional pre-RGI downsample; changes density-sensitive RGI behavior |
| `--qsm-voxel-size` | `0.08` | Input downsampling before QSM |
| `--max-points` | `400000` | Deterministic cap per condition |
| `--num-workers` | `1` | Tile workers; explicitly limited to 1 or 2 |
| `--distance-tolerance` | `0.05` | Match tolerance (m) for wood target ↔ cylinder abstraction |
| `--abstraction-target-voxel` | `0.08` | Coarsen wood target to this resolution (proxy for optimal low-poly); `0` = full high-res |
| `--metric-voxel-size` | `0.10` | Legacy voxel helper / secondary CylGT path |
| `--trunk-radius-threshold` | `0.05` | Radius split for optional CylGT trunk/branch summaries |
| `--keep-cyl-gt-metrics` / `--no-keep-cyl-gt-metrics` | on | Secondary mesh-OBB cylinder-GT scores |
| `--sq-leafon-cfg` / `--sq-leafoff-cfg` | spconv LEAFON / space-colonization | Condition-specific SmartQSM settings; override for CPU/GPU availability |
| `--export-adqsm` | off | Export condition-specific XYZ + manifest for manual AdQSM runs |

AdQSM's available test build is GUI-only and cannot be included in an unattended timed benchmark. `--export-adqsm` creates reproducible inputs and parameter metadata; its outputs must be run and imported manually.

Plot completed rows:

```bash
python scripts/plot_benchmark_qsm.py
```

## Testing

```bash
pytest tests/
```

## See Also

- `agent.md` — Design decisions, physics details, and operational gotchas gathered during development
