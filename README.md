# Ecomodel

Ecomodel turns terrestrial LiDAR scans of forest plots into per-tree structural models.
It takes raw `.las` / `.laz` tiles, removes the ground, separates individual trees, then
fits a Quantitative Structure Model (QSM). A QSM is a cylinder skeleton of every trunk and
branch. You can view it, measure it, and run spatial queries against it in the GUI.

QSM fitting is powered by [TreeQSM](#treeqsm-engine), available here as a standalone Python
implementation.

> **Status:** active development. Expect rough edges.

![QSM skeleton fitted to a scanned tree, shown over the source point cloud](Docs/figs/gui_qsm_skeleton.png)

---

## Install

Requires **Python 3.8 to 3.11**. Python 3.12 is not supported because Open3D has no build
for it. We recommend 3.11.

```bash
conda create -n ecomodel python=3.11
conda activate ecomodel

git clone https://github.com/Landscape-CV/Ecomodel.git
cd Ecomodel
pip install -r requirements.txt
```

## Quick start

```bash
python run_gui.py
```

Point **LAS/LAZ Folder** at a directory of tiles, pick a pipeline mode, then press **Start**.

> The pipeline processes **every** `.las` / `.laz` file directly inside that folder.
> Subfolders are ignored. Keep one dataset per folder.

![Pipeline page: parameters on the left, run controls and log on the right](Docs/figs/gui_pipeline.png)

## The pipeline

There are two modes. **Lite** is the CPU pipeline and the one to start with. **Full** is the
larger GPU-accelerated pipeline for big scans.

Lite runs six steps per tile:

| # | Step | Notes |
|---|------|-------|
| 1 | Load & normalise | |
| 2 | Ground removal | Cloth Simulation Filter (CSF) |
| 3 | Intensity filter | drops low-return points |
| 4 | Leaf removal | region-growing (RGI) |
| 5 | Instance segmentation | splits the tile into individual trees |
| 6 | QSM | fits cylinders per tree |

Each stage has its own parameter group in the left-hand panel. Groups are collapsed by
default and use the values shown until you enable them.

## Results

Each run writes a timestamped folder under your results directory. Open it from the
**Results** page to view:

- **Point Cloud**: the processed cloud, coloured by height or intensity
- **Segments**: coloured by tree, for checking segmentation
- **Cylinders / Skeleton**: the fitted QSM
- **Tree Metrics**: per-tree volume, height and branch statistics

**Query** answers spatial questions against a finished run, such as the cylinder volume
within a radius of a point.

![Results page showing the processed point cloud coloured by height](Docs/figs/gui_point_cloud.png)

![Tree Metrics page: diameter, length and branch angle distributions with a summary table](Docs/figs/gui_tree_metrics.png)

## TreeQSM engine

The QSM step is a Python port of TreeQSM. It can be used on its own without the GUI:

```bash
python -m PyTLidar.treeqsm file.las --normalize        # single file
python -m PyTLidar.treeqsm_batch folder --normalize    # a folder, in parallel
```

Run with `--help` for the full option list, including patch diameters, optimum-model
selection, output directory and core count. See [Docs](Docs) for the module API and the
metrics definitions.

## Contributing

Bugs and suggestions go in **Issues**. Please check for an existing report first.

For fixes and small features, open a pull request and link it to an issue where you can.
For anything larger, talk to the dev team first so we can plan it together.


# Troubleshooting & Machine-Specific Configurations

## SmartQSM on NVIDIA RTX 5070 Ti (Blackwell Architecture)
When running the experimental pipeline or benchmarking against **SmartQSM** on next-gen hardware like the RTX 5070 Ti, several machine-specific modifications were required to ensure compatibility and concurrent execution:

### 1. PyTorch Nightly & CUDA 12.8
The Blackwell architecture (RTX 5000 series) is not fully supported by older CUDA runtime binaries bundled with stable PyTorch releases. To resolve CUDA architecture mismatch errors, we explicitly installed the **PyTorch Nightly (dev) build** compiled against **CUDA 12.8**:
```bash
# Example nightly installation (ensure you match the cu128 index)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
*(Verified running on `2.12.0.dev+cu128`)*

### 2. SmartQSM Python Syntax Fixes
Depending on the local Python version, the original `SmartQSM` repository may throw `SyntaxError` due to nested f-strings with identical quote types. 
- **Files patched:** `thirdparty/SmartQSM/entrypoints/smartqsm.py` and `thirdparty/SmartQSM/entrypoints/_updater.py`
- **Fix:** Refactored inline `.join()` calls out of f-strings. (e.g., extracting `skipped_str = '\n'.join(skipped_files)` before printing).

### 3. Disabling SmartQSM FileLocks for Multi-Processing
By default, the `smartqsm.py` entrypoint enforces a strict `FileLock` (`lock.acquire()`), which aggressively prevents multiple instances from running concurrently.
- Because our pipeline benchmarks multiple point cloud tiles in parallel using `ProcessPoolExecutor`, this lock caused instant `Timeout` failures.
- **Fix:** We manually commented out the `FileLock` acquisition and release blocks in `thirdparty/SmartQSM/entrypoints/smartqsm.py` to allow parallel inference across different sub-processes.

### 4. Spconv Support for CUDA 12.8 (Issue #775)
The `spconv` library (used for sparse convolutions in our deep learning models) lacks official prebuilt stable wheels for CUDA 12.8/13.0 (see [spconv/issues/775](https://github.com/traveller59/spconv/issues/775)).
- If you use the standard `spconv-cu121` or `spconv-cu118` on a Blackwell GPU, it will fail at runtime with NVCC/NVRTC architecture compilation errors.
- **Fix:** You must uninstall older versions and explicitly install the experimental `spconv-cu128` (e.g. `2.4.1`), or build it from source:
```bash
pip uninstall spconv-cu121
pip install spconv-cu128==2.4.1
```

### 5. AdQSM Integration
[AdQSM](https://github.com/GuangpengFan/AdQSM) was integrated into the `thirdparty/AdQSM` directory for structural benchmarking purposes.
- **Note on Batch Processing:** The standard release of AdQSM (V1.7.5 test version) does **not** provide a native Command-Line Interface (CLI) or batch processing API. It is strictly a GUI application.
- **Configuration:** We cloned the repository and kept the pre-built Windows executables (`AdQSM-V1.7.5..exe`). To process point clouds using AdQSM, users must manually load their `.xyz` files into the GUI, adjust the Height Segmentation (HS) and Cloud Parameter (CP) parameters (default CP=0.003), and click "Reconstruct". 
- **Output extraction:** Results are manually saved to `treesparams.csv` and `branchinfo.txt` in the installation path. Due to its manual nature, AdQSM is excluded from our automated multi-tile `benchmark_qsm.py` pipeline, but it is supported as a comparative tool for single-tree manual analysis.

### Contributing Fixes

You may create a fork of our repository to submit a pull request. Your request will be reviewed and if approved will be incorporated. For best chances at approval, attach to an existing issue or create your own to resolve. 

### Contributing New Features

If you have a simple feature to add, you may follow the same procedure as contributing a fix. However, if you have a larger feature, collaboration with the broader team may be warranted. If you feel this is the case, please reach out to someone on the dev team and a plan can be developed for your collaboration and certain permissions may be granted. 

## License

PyTLidar is published under the GPL 3.0 License
