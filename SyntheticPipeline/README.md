# Synthetic LiDAR Pipeline

This submodule manages the generation, assembly, and LiDAR simulation of synthetic forest environments. It's designed to abstract the 3D generation process, allow for custom environments, and output highly accurate TLS (Terrestrial Laser Scanning) point clouds and corresponding structural ground truth (GT) cylinders.

## Features

- **Generator Agnostic:** Supports generating 3D models from any system (currently implements Blender Geometry Nodes/L-Systems).
- **Automated Scene Assembly:** Seamlessly places individual trees into a simulated forest area, accounting for random rotations, scales, wind tilt, and proper ground projection.
- **LiDAR Simulation:** Utilizes `Open3D` raycasting to simulate a Terrestrial Laser Scanner (TLS). Supports adding spatial jitter (beam divergence proxy), distance noise, and wind sway.
- **Ground Truth Export:** Traces full analytical tree skeleton parameters back into the scene to output high-fidelity structural data.

## Directory Structure

- `configs/`: Contains machine-specific `pipeline_config.json` and tree generation settings (like `mangrove_config.json`). Use the provided `.example.json` files to set up your environment.
- `pipeline/`: The core orchestration engines:
  - `asset_manager.py`: Manages bulk generation of trees.
  - `forest_assembler.py`: Assembles individual trees into the forest scene.
  - `simulator_open3d.py`: The TLS LiDAR Raycaster.
  - `gt_parser.py`: Formats the parsed GT into evaluating formats.
- `generators/`: Implementations of tree algorithms (e.g., Blender).
- `evaluation/`: Scripts to visualize, inspect, and evaluate generated meshes and point clouds against ground truth.
- `tests/`: A `pytest` suite for automated CI/CD checks.
- `output/`: (Generated) The destination for all intermediate assets, scenes, and laz point clouds.

## Mangrove Configuration (`mangrove_config.json`)

When using the `BlenderGenerator` with the Mangrove preset, you can fine-tune the tree generation properties via `configs/mangrove_config.json`. This file prevents randomizing structural values inappropriately and allows precise control over the generated assets.

Key parameters include:
- `"decimate_ratio"`: (Float, e.g., 0.3) Reduces the final polygon count of the generated tree mesh by this ratio, heavily optimizing downstream assembly and simulation.
- `"remove_underground"`: (Boolean) If true, a script cleans up any mesh vertices that fall below the ground plane (`Z < 0`), ensuring flush placement on terrain.
- `"randomize_seed_parameters"`: (Boolean) If true, enables varying specific Geometry Nodes parameters per-tree to create realistic forest diversity.
- `"seed_parameters"`: (List) Specifies which Geometry Nodes inputs (e.g., `"Input_20"`) receive the random seed offset.
- `"parameters"`: (Object) A direct mapping of Blender Geometry Nodes modifier inputs. Here is the complete reference of what each parameter controls for the Mangrove tree:
  - `Input_2`: Base level
  - `Input_3`: Root Spread
  - `Input_4`: Root lift
  - `Input_5`: Root Angle
  - `Input_6`: Stem tip radius
  - `Input_7`: Stem base radius
  - `Input_8`: Stem height
  - `Input_9`: Root Base radius
  - `Input_10`: Root Tip Radius
  - `Input_11`: Base Radius
  - `Input_12`: Root depth
  - `Input_14`: Root Secondary Radius
  - `Input_15`: Resolution Multiplier
  - `Input_16`: root amount
  - `Input_17`: Root trim Elevation
  - `Input_18`: Root Rise
  - `Input_19`: Root spin
  - `Input_20`: Seed
  - `Input_21`: Branching start
  - `Input_22`: Branching spin
  - `Input_23`: Branching bend
  - `Input_24`: Branching vertical spread
  - `Input_25`: Enable Proxy (Usually overridden internally to False)

## Arbaro Configuration (`arbaro_template.xml`)

Arbaro is a Java-based procedural tree generator that uses an XML parameter system to define species and growth rules based on the Weber/Penn algorithm. You can switch to Arbaro by updating your `pipeline_config.json`:
```json
"generator": {
    "type": "arbaro",
    "arbaro_jar_path": "SyntheticPipeline/lib/arbaro/arbaro_cmd.jar",
    "xml_template": "SyntheticPipeline/configs/arbaro.xml",
    "java_path": "java",
    "workers": 2
}
```

The pipeline automatically handles injecting the random seed, dynamically adjusting the `Scale` parameter to match the height distribution in your config, and running Arbaro twice (once to generate the visual leaf-on mesh, and once with `Leaves` set to 0 to generate a leafless skeleton for GT cylinder extraction).

### Comprehensive Arbaro Options
Inside your XML template (e.g., `configs/arbaro.example.xml` or `configs/arbaro.xml`), you have full control over the structural parameters. Key parameters to experiment with include:

**General Tree Geometry:**
- `Scale`: The global scale multiplier (the pipeline actively manipulates this per-tree).
- `ScaleV`: Variance of the scale (randomness).
- `BaseSize`: Fractional height of the main trunk before the first branches appear.
- `Ratio`: How quickly branch thickness decreases.
- `RatioPower`: The tapering curve of branches.
- `Flare`: The expansion at the very base of the root.

**Leaves:**
- `Leaves`: The number of leaves generated on the highest level branches.
- `LeafShape`: Index of leaf shape (0=ovate, 1=triangle, etc. depending on Arbaro).
- `LeafScale` / `LeafScaleX`: Physical dimensions of individual leaves.
- `LeafBend`: How much leaves droop under gravity.

**Branch Levels (0 = Trunk, 1 = Main branches, 2 = Twigs, etc.):**
*Each level has its own configuration prefix, e.g., `0DownAngle`, `1DownAngle`.*
- `[level]Branches`: How many branches spawn from the parent level.
- `[level]DownAngle` / `[level]DownAngleV`: The angle relative to the parent branch.
- `[level]Rotate` / `[level]RotateV`: The helical rotation around the parent branch.
- `[level]Length` / `[level]LengthV`: Branch length relative to the parent.
- `[level]Curve` / `[level]CurveV` / `[level]CurveBack`: Gravity and phototropism bending forces.

For full mathematical definitions of these properties, refer to Jason Weber & Joseph Penn: "Creation and Rendering of Realistic Trees".

## Quickstart

1. **Setup configs:** Copy `configs/pipeline_config.example.json` to `configs/pipeline_config.json`. Update paths like `"blender_path"` to match your local installation.
2. **Run the simulation:** Run the main orchestrator script:
   ```bash
   python run_simulation.py
   ```
   *Note: If no arguments are passed, it defaults to `configs/pipeline_config.json`.*

3. **Advanced Runs:** You can run specific configurations by pointing to them:
   ```bash
   python run_simulation.py --config configs/my_custom_config.json
   ```

## Quickstart: Generating & Benchmarking a Dataset

If you are new to the project and want to quickly generate synthetic point clouds and benchmark the segmentation models against them, follow these 3 steps:

**1. Generate the Dataset**
This script reads tree types from `lib/arbaro/trees`, builds random forests, and simulates LiDAR scans, saving them to `testdataset/`.
```bash
python scripts/generate_test_dataset.py
```
*(Note: To prevent out-of-memory errors and keep generation fast, the default parameters in this script are downscaled for lower ray density and fewer trees per tile).*

**2. Label the Point Clouds (Ground Truth)**
The segmentation models need to know which points are wood and which are leaves. This script computes distances to the GT trunk meshes to create binary labels (`*_labels.npy`).
```bash
python scripts/label_gt_points.py
```

**3. Run the Benchmark**
Now evaluate the `SegmentRGI` model against your newly generated dataset.
```bash
python scripts/benchmark_separation.py --dataset_dir testdataset/ --out_csv output/benchmark_results.csv --num_workers 3
```

## Benchmarking Arguments

The pipeline includes a script (`scripts/benchmark_separation.py`) to evaluate tree segmentation models (e.g., `SegmentRGI`) against the synthetic ground truth datasets.

To run the benchmark across your generated dataset:
```bash
python scripts/benchmark_separation.py --dataset_dir testdataset/ --out_csv output/benchmark_results.csv
```

**Key Arguments:**
- `--num_workers <int>`: Leverages Python's `ProcessPoolExecutor` to run multiple tiles in parallel, speeding up evaluation significantly. Defaults to `cpu_count() - 1`.
- `--sample_n <int>`: Randomly samples a subset of tiles (e.g., `--sample_n 5`) to run a quick test instead of benchmarking the entire dataset.
- `--visualize`: Instead of popping up a UI window that blocks parallel workers, this flag generates colored side-by-side `.ply` point clouds showing the Ground Truth (left) vs Prediction (right). These are saved directly to `output/visualizations/` for easy review in software like CloudCompare.

## Requirements

Ensure the parent environment dependencies are installed, particularly:
- `open3d`
- `trimesh`
- `laspy[lazrs]`
- `pytest`

## Testing

Run tests by executing:
```bash
pytest tests/
```
