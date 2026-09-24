# SyntheticPipeline CLI Cookbook

Run all commands from `SyntheticPipeline/` unless noted. Use `py -3.11` (or your env with `open3d` / `trimesh` / `laspy`) if the default Python lacks deps.

```bash
cd SyntheticPipeline
```

---

## 0. One-time setup

```bash
# Arbaro JAR + species XMLs under lib/arbaro/
python scripts/setup_arbaro.py

# Machine-local configs from examples
cp configs/pipeline_config.example.json configs/pipeline_config.json
cp configs/mangrove_config.example.json configs/mangrove_config.json   # Blender / Mangrove only
```

Edit `pipeline_config.json` for Blender paths or switch `"generator": { "type": "arbaro", ... }`.

---

## 1. End-to-end TLS tile (main entry)

Thin CLI → `synthetic_tls.orchestrate.run_tile`.

```bash
# Default: configs/pipeline_config.json
python run_simulation.py

# Explicit config
python run_simulation.py --config configs/demo_short.json
python run_simulation.py --config-dir configs
```

**Config blocks**

| Block | Role |
|-------|------|
| `simulation` | `scene_name`, `num_trees`, `area_size`, `num_scans`, wind |
| `generator` | `type`: `blender` \| `arbaro`, paths, `workers` |
| `noise` | TLS resolution, distance / beam / wind sway noise |
| `output_dirs` | `assets`, `scenes`, `pointclouds` |

**Short smoke example** (`configs/demo_short.json`): 1 Arbaro aspen, 8 m plot, 1 coarse scan → `output/demo_short/`.

```bash
python run_simulation.py --config configs/demo_short.json
```

---

## 2. Stage CLIs (optional)

Usually you do not need these; `run_tile` wires them. Useful for debugging one stage.

### GT cylinders from leafless mesh

```bash
python -m synthetic_tls.generators.cylinders \
  --mesh_path output/demo_short/assets/tree_0000_noleaf.obj \
  --out_json output/demo_short/assets/tree_0000.json
```

### Assemble forest from existing assets

Requires assets already under the default/asset dir the assembler uses (or set paths in code/config). CLI:

```bash
python -m synthetic_tls.assemble.forest_assembler \
  --scene_name my_scene \
  --num_trees 5 \
  --area_size 20 \
  --species quaking_aspen \
  --min_spacing 2.0 \
  --placement_mode spread
```

### Scene GT JSON → evaluation `.txt`

```bash
python -m synthetic_tls.gt.parser \
  --json_path output/demo_short/scenes/demo_short_gt.json \
  --out_name demo_short_gt
```

### Open3D TLS scan only

```bash
python -m synthetic_tls.simulate.open3d \
  --mesh_path output/demo_short/scenes/demo_short.ply \
  --out_name demo_scan \
  --num_scans 1 \
  --area_size 8 \
  --face_tree_ids output/demo_short/scenes/demo_short_face_tree_ids.npy
```

### Blender runtime (host normally uses `BlenderGenerator`)

```bash
blender -b [optional.blend] -P synthetic_tls/generators/blender_runtime/entrypoint.py -- \
  --seed 0 --height 10 --out_obj tree.obj --out_json tree.json
```

- With a loaded Mangrove `.blend` containing object `Mangrove tree` → Geometry Nodes path (`mangrove.py`), config at `configs/mangrove_config.json`.
- Otherwise → native L-system (`lsystem.py`).

---

## 3. Build datasets

### Single-tree wood/leaf tiles

```bash
python scripts/generate_test_dataset.py
# → testdataset/single/{species}_tile_*_{scan,gt,trunk,meta}.*
```

Defaults (in script): small pool, 0.2° rays, 5 cm voxels — keep RAM down.

### Multi-tree instance tiles

```bash
cp configs/instance_benchmark.example.json configs/instance_benchmark.json
# or smoke:
cp configs/instance_benchmark_smoke.example.json configs/instance_benchmark_smoke.json

python scripts/generate_instance_benchmark.py
python scripts/generate_instance_benchmark.py \
  --config configs/instance_benchmark_smoke.example.json \
  --skip_asset_gen \
  --seed 42
# → testdataset/instance/ (or config dataset_dir)
```

### Label wood/leaf on scans

Distance to trunk mesh (`*_trunk.ply`):

```bash
python scripts/label_gt_points.py \
  --dataset_dir testdataset/single \
  --dist_thresh 0.05
```

---

## 4. Wood / leaf separation benchmark

```bash
python scripts/benchmark_separation.py \
  --dataset_dir testdataset/single \
  --out_csv output/benchmark_results_single.csv \
  --algorithms SegmentRGI,GBSeparation \
  --num_workers 3 \
  --sample_n 10 \
  --voxel_size 0.1 \
  --trunk_radius_threshold 0.05 \
  --visualize \
  --plot
```

Optional SmartQSM: include in `--algorithms` and set `--sq_dir` / `--sq_py` / `--sq_cfg`.

### Plots

```bash
python scripts/plot_species_performance.py \
  --csv output/benchmark_results_single.csv \
  --out output/visualizations/species_f1.png

python scripts/plot_species_pr.py \
  --csv output/benchmark_results_single.csv \
  --out output/visualizations/species_pr.png
```

### Real wood/leaf corpora

```bash
python scripts/prepare_lewos_woodleaf.py
python scripts/prepare_heidelberg_woodleaf.py

python scripts/benchmark_wood_leaf.py
python scripts/benchmark_wood_leaf.py --lewos_only
python scripts/benchmark_wood_leaf.py --heidelberg_only --skip_gb
```

---

## 5. Instance segmentation benchmark

```bash
python scripts/benchmark_instance_segmentation.py \
  --dataset_dir testdataset/instance \
  --algorithms scanline,treelearn,pointsam,snap \
  --prompt_mode both \
  --stratify 16 \
  --seed 0 \
  --out_csv output/benchmark_instance.csv \
  --pointsam_ckpt ../thirdparty/checkpoints/point_sam/model.safetensors \
  --snap_ckpt ../thirdparty/checkpoints/snap/SNAP_C.pth
```

Useful flags: `--leaf_removal`, `--sample_n`, `--iou_thresh`, `--no_treelearn_gpu`, `--no_pointsam_gpu`, `--no_snap_gpu`.

### Plots

```bash
python scripts/plot_benchmark_instance.py \
  --csv output/benchmark_instance.csv \
  --out output/visualizations/benchmark_instance.png

python scripts/plot_leaf_removal_ablation.py
python scripts/plot_method_comparison_val16.py
```

### Real TLS/MLS instance tiles

```bash
python scripts/prepare_real_instance_tiles.py --help
```

### Instance annotator demos

```bash
python -m instance_annotator --tile <prefix>
python -m instance_annotator --only manual --skip_treelearn
# --only: all | treelearn | manual | correct
```

---

## 6. QSM reconstruction benchmark

```bash
python scripts/benchmark_qsm.py \
  --dataset-dir testdataset/single \
  --out-dir output/qsm_benchmark \
  --algorithms TreeQSM,SmartQSM \
  --conditions leaf_on,rgi,oracle_wood \
  --sample-n 5 \
  --num-workers 1
```

Optional backends: AdTree / aRchi flags (`--adtree-exe`, `--archi-rscript`, …). See `python scripts/benchmark_qsm.py -h`.

```bash
python scripts/plot_benchmark_qsm.py
python scripts/quick_test_smartqsm.py path/to/tile_scan.laz
```

---

## 7. Visualize a tile

```bash
python evaluation/visualize_pipeline.py \
  --laz_path output/demo_short/pointclouds/demo_short_scan.laz \
  --gt_path output/demo_short/scenes/demo_short_gt.json \
  --downsample 20

# Or overlay trunk mesh instead of cylinder JSON:
python evaluation/visualize_pipeline.py \
  --laz_path output/demo_short/pointclouds/demo_short_scan.laz \
  --gt_mesh_path output/demo_short/scenes/demo_short_gt_mesh.ply
```

---

## Cheat sheet

| Goal | Command |
|------|---------|
| Full synthetic tile | `python run_simulation.py --config …` |
| Single-tree dataset | `python scripts/generate_test_dataset.py` |
| Instance tiles | `python scripts/generate_instance_benchmark.py` |
| Wood/leaf labels | `python scripts/label_gt_points.py` |
| Separation metrics | `python scripts/benchmark_separation.py` |
| Instance metrics | `python scripts/benchmark_instance_segmentation.py` |
| QSM metrics | `python scripts/benchmark_qsm.py` |
| Inspect a scan | `python evaluation/visualize_pipeline.py` |
| Package import API | `from synthetic_tls import run_tile, create_generator` |

---

## Package layout (for orientation)

```
synthetic_tls/
  orchestrate.py / factory.py / constants.py
  assemble/     AssetManager, ForestAssembler
  simulate/     Open3DSimulator
  gt/           GTParser
  generators/   Blender + Arbaro hosts, cylinders
  generators/blender_runtime/   Blender -P entrypoint (Mangrove + L-system)
```

Instance ID conventions: trees `≥ 0`, ground `-1`, clutter `-2`.
