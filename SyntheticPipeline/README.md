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
