# TLS Instance Annotator (Web)

Inspect and correct tree **instance** labels on terrestrial / MLS point clouds, then export corrected ground truth in the Ecomodel tile layout.

It is a local web app: a FastAPI backend holds the full-resolution cloud and labels, and a three.js frontend in the browser renders a voxel level-of-detail (up to ~2M points by default) on the GPU. Edits are sent as selected display points and expanded exactly to full resolution on the server, so nothing is lost to downsampling.

## Install

Core deps are already in the Ecomodel / PyTLidar env (`numpy`, `scipy`, `laspy`, segmenters). Add the web server:

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
& $py -m pip install fastapi "uvicorn[standard]" python-multipart
```

No Node.js or build step: three.js is vendored under `web/vendor/`.

## Start

```powershell
cd D:\Projects\Ecomodel\SyntheticPipeline
D:\Projects\PyTLidar\.venv\Scripts\python.exe -m instance_annotator
```

The browser opens at `http://127.0.0.1:8765`. Useful flags:

| Flag | Meaning |
|------|---------|
| `--load testdataset/real_instance/l1w_t00_03` | Open a tile prefix (or LAZ/LAS/PLY) on start |
| `--blank` | Ignore `*_instances.npy` and start with all points non-tree |
| `--max_display 3000000` | Display LOD cap (edits stay full-res) |
| `--port 8766` / `--no_browser` | Server options |

## Workflow

1. **Open** (`O`): type a tile prefix or path, or browse folders (tiles show a *labels* badge when `*_instances.npy` exists). Choose **Use existing labels** or **Start blank**. Recent paths are remembered. Upload also works for small files.
2. **Review trees** in the **Trees** tab: sort by ID / points / height / unreviewed, filter by ID, step with `[` and `]` (camera frames each tree), press `K` to mark reviewed. The progress bar tracks review coverage; reviewed IDs are written to `*_meta.json`.
3. **Select**
   - Navigate mode (`V`): click a point to select its whole tree; Shift adds, Ctrl subtracts. Double-click isolates and frames a tree.
   - **Lasso** (`L`), **Box** (`B`), **Sphere brush** (`R`, radius in the top bar, `-` / `=` to resize). In tool modes, right-drag orbits and middle-drag pans.
   - Modifiers: Shift = add, Ctrl = subtract, **Alt = only the focused tree** (or the tree under the cursor).
   - **Grow** (`G`): connected region growing from the selection within the brush radius, same label only.
   - **Height slab** in the View panel hides everything outside a Z range, which makes stems easy to lasso under dense canopy. Hide non-tree (`H`) and Isolate (`I`) help too.
4. **Edit** (always available for the current selection):
   - **New tree** (`N`), **Assign** (`A`, to any ID or -1), **Merge** (`M`, merges whole trees touched by the selection into a target), **Non-tree** (`Del`).
   - **Undo / Redo** (`Ctrl+Z` / `Ctrl+Y`) are diff-based, so they are cheap even on 20M-point clouds.
5. **Save** (`Ctrl+S` or the **Export** tab). The tile name follows the loaded cloud; existing files trigger an overwrite prompt. Writes:
   - `{name}_scan.laz`, `{name}_instances.npy` (`int32`, trees `>=0`, non-tree `-1`), `{name}_meta.json`, `{name}_preview.ply`

That triplet is loadable by `scripts/benchmark_instance_segmentation.py`.

**Safety:** a dot next to the file name marks unsaved changes; the tab warns before closing, and opening another cloud or running segmentation asks first. Labels are **autosaved** every 10 edits to `output/annotator_workdir/<name>.autosave.npy`; on the next open you are offered to restore it. Merging into non-tree always asks for confirmation.

## Keyboard shortcuts

Press `?` in the app for the full list.

| Key | Action |
|-----|--------|
| `V` / `L` / `B` / `R` | Navigate / Lasso / Box / Brush |
| `N` / `A` / `M` / `Del` | New tree / Assign / Merge / Non-tree |
| `G` / `T` / `Esc` | Grow / select focused tree / clear selection |
| `[` / `]` / `K` | Previous / next tree / toggle reviewed |
| `I` / `H` / `C` / `F` | Isolate / hide non-tree / toggle wood-leaf colors / frame |
| `Ctrl+Z` / `Ctrl+Y` / `Ctrl+S` | Undo / redo / save |

## Wood / leaf reference (Wood / Leaf tab)

Runs in the background and colors points brown (wood) / green (leaf); toggle each class in the View panel. It does not change instance labels.

Defaults (Stem-grow `verticality_min=0.8`, `height_percentile=15`, `grow_radius=0.2`) were selected by a parameter sweep on the open [LeWoS LabelledPC](https://zenodo.org/records/4946676) wood/leaf GT (61 tropical TLS trees). Full logs and plots: `SyntheticPipeline/output/wood_leaf_benchmark/` (`trials.csv`, `summary.json`, comparison PNGs). Intensity methods are N/A on LeWoS (no intensity channel); they were evaluated on Heidelberg TLS LAZ.

| Method | Notes |
|--------|--------|
| Stem-grow | Seed low-Z vertical points, grow by radius. **Best LeWoS holdout macro-F1 about 0.81**. |
| Eigenfeatures | Wood = high linearity + verticality + low curvature (kNN PCA). Close second (about 0.80). |
| Intensity percentile / threshold / Otsu | Need real intensity. |
| RGI | Region growing on a subsample. Weak on LeWoS GT (about 0.24). |
| GBSeparation | Graph + root-path wood. Subsamples large tiles. |

## Segmentation (Segment tab)

Runs a benchmark instance method in the background and replaces all labels (undoable).

| Method | Notes |
|--------|--------|
| `treelearn` | Needs TreeLearn weights + CUDA for speed. Config: `TreeLearn/configs/pipeline/ecomodel.yaml`. |
| `pointsam` | Needs `thirdparty/checkpoints/point_sam/model.safetensors`. Auto prompts only. |
| `treex` | Stock `TreeXPresetTLS` when "TreeX stock TLS settings" is checked. |
| `tls2trees` | In-process port; RGI semantic unless leaf removal already ran. |
| `scanline` | Classic Ecomodel stem graph; often finds few stems on 0.1 m voxels. |

## Label convention

| Label | Meaning |
|------:|---------|
| `-1` | Non-tree / ground / unlabeled |
| `>=0` | Tree instance ID |

## Headless demos

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
cd D:\Projects\Ecomodel\SyntheticPipeline
& $py -m instance_annotator.cli_demo --tile testdataset/real_instance/l1w_t00_03
& $py -m instance_annotator.cli_demo --skip_treelearn --only manual
```

Artifacts land in `output/annotator_demos/` (tile triplets plus colored `_preview.ply`).

## Package layout

```
instance_annotator/
  __main__.py     # python -m instance_annotator -> web server
  server/         # FastAPI app (main.py), API routes, session state + jobs
  web/            # index.html, style.css, js/ (viewer, tools, panels), vendor/three.js
  lod.py          # Voxel display LOD with exact full-res mapping
  labels.py       # Diff-based undoable label editor
  io.py           # LAZ/PLY + tile I/O
  segment.py      # Benchmark method wrapper
  wood_leaf.py    # Wood/leaf classifiers
  viz.py          # Colors, colored PLY, radius pick
  cli_demo.py     # Scripted demos
  test_core.py    # Core + API tests (pytest)
```

## Tips

- **Large clouds:** open by disk path, not upload. A 20M-point cloud takes a few seconds to build the LOD; lower **Max display points** in the Open dialog if the browser is slow.
- Relative paths resolve under `SyntheticPipeline/`.
- Only one browser tab should edit at a time (the server holds a single session).
