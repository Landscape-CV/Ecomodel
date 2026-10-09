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
| `pointsam` | Auto prompts only. A **Point-SAM weights** dropdown (Segment tab and island segmentation) lists the stock `thirdparty/checkpoints/point_sam/model.safetensors` and every fine-tuned `SyntheticPipeline/pointsam_checkpoints/*/{best,last}.safetensors`; `--pointsam_ckpt` on the server sets the preselected default. |
| `treex` | Stock `TreeXPresetTLS` when "TreeX stock TLS settings" is checked. |
| `tls2trees` | In-process port; RGI semantic unless leaf removal already ran. |
| `scanline` | Classic Ecomodel stem graph; often finds few stems on 0.1 m voxels. |

## Whole-island projects (streaming)

For scans that are too big to load in one go (for example the 18-tile, 975M-point island in `D:\pointclouds\processed2025scans`), build an **island project** once. The browser then streams an octree of all tiles, Potree-style, and you load full-resolution **edit regions** only where you need to work.

### Build and open

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
cd D:\Projects\Ecomodel\SyntheticPipeline
& $py -m instance_annotator.project build D:\pointclouds\processed2025scans --out D:\pointclouds\island_project
& $py -m instance_annotator --project D:\pointclouds\island_project
```

You can also open a project from the **Open** dialog: project folders show an *island project* badge. The build merges all tiles into one project-local frame, using the absolute coordinates in the LAZ headers; the CRS is kept. Every point is stored exactly once, even where tiles overlap. A rebuild only redoes tiles whose source changed (`--force` redoes all of them).

### Island tab

- **Streaming:** the point budget slider defaults to 8M and goes up to 20M. **Adaptive point size** fills gaps at coarse levels. The stream stats line shows how many points and nodes are visible.
- **Minimap:** shows the tiles, tree bases, the open region and the camera footprint. Click it to move the camera.
- **Colors:** Instance IDs shade non-tree points by height. **Height** colors every point by elevation.
- **Picking:** click a point to focus its tree, which also flies the camera there. Double-click isolates the tree. The Trees tab lists island-wide trees.
- **Focused tree (whole island):** **Merge into…** and **Non-tree** act on every tile the tree touches without loading it. Both are undoable with `Ctrl+Z` / `Ctrl+Y`.
- **Edit region:** use **Draw box** (drag on the ground) or **Around focused tree** (with a margin in metres), then **Open region**. That box loads at full resolution into the normal editor, so Lasso, Box, Brush, New tree, Grow and so on all work. The rest of the island stays visible but dimmed. Every edit is **written straight into the project**, and new trees get island-unique IDs. There is no separate save step. **Close region** returns to streaming.
- **Segment whole island:** runs a benchmark method over **buffered tiles**: each tile plus a 10 m buffer.
  - Each tree belongs to the tile that contains its stem base.
  - Duplicates found in overlapping buffers are merged by voxel IoU.
  - Borderline pairs are queued for review.
  - Per-tile results are cached in `seg_cache/`, and labels are backed up to `labels_backup/` before they are overwritten.
  - Each tile is thinned to one seed point per voxel before segmenting. The default voxel is 5 cm, but **TreeX** uses 3 cm with up to 40M seeds per tile, because its stock TLS stem search finds nothing at 5 cm. If a tile's seeds end up coarser than 4 cm, TreeX switches to its relaxed parameters for sparse clouds.
  - If no tile yields a tree, the job fails and existing labels are left untouched.
- **Border merges:** **Find candidates** scores trees that touch across tile edges. Each candidate has **Fly** (look at the pair), **Merge** and **Reject** buttons. Decisions persist in `stitch_candidates.json`.
- **Export** tab (island mode) has three outputs:
  - Per-tile triplets in the standard layout (`<tile>_scan.laz` is a lossless copy of the source, plus `_instances.npy` with **island-wide IDs** and `_meta.json`). These load in single-file mode and in the benchmarks.
  - `trees.csv`.
  - An optional per-tree LAZ for the focused tree or for all trees.

### CLI

| Command | Purpose |
|---------|---------|
| `project build <src> --out <proj> [--workers N] [--force]` | Build the octree and label store |
| `project import <proj> <labels_dir> [--keep_ids]` | Import existing per-tile `*_instances.npy` (IDs are offset per tile unless `--keep_ids`) |
| `project segment <proj> --method treelearn [--buffer 10] [--tiles a,b] [--reuse_cache]` | Buffered island segmentation |
| `project stitch <proj> [--apply 0.5]` | Find border candidates, optionally auto-apply those with score ≥ the threshold |
| `project export <proj> --out <dir> [--tiles a,b] [--trees 3,7 \| --all-trees] [--no_tiles]` | Per-tile triplets, `trees.csv`, per-tree LAZ |
| `project info <proj>` | Summary |

All commands are `python -m instance_annotator.project …`.

### Measured on the real island (18 LAZ tiles, 974,765,161 points, 8.0 GB source)

| Step | Result |
|------|--------|
| Build (one-off) | 707 s. The project is 17.3 GB (octree tiles 13.6 GB, labels 3.6 GB). |
| Open in the browser | Preview in 0.8 s; the overview settles in about 3 s |
| Rendering | 7.5M visible points at about 5 ms per frame (about 180 fps) |
| Open an edit region | 5 to 6 s for 15 to 22M full-resolution points |
| Pick a tree | About 10 ms |

Whole-island segmentation time is dominated by the chosen method, since it runs once per buffered tile. Use `--tiles` to try one or two tiles first, and `--reuse_cache` to resume.

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
  server/         # FastAPI app (main.py), API routes, session + island project session, jobs
  project/        # Island projects: build (octree), store, segment_island, stitch, stats, export, CLI
  web/            # index.html, style.css, js/ (viewer, octree, island, tools, panels), vendor/three.js
  lod.py          # Voxel display LOD with exact full-res mapping
  labels.py       # Diff-based undoable label editor
  io.py           # LAZ/PLY + tile I/O
  segment.py      # Benchmark method wrapper
  wood_leaf.py    # Wood/leaf classifiers
  viz.py          # Colors, colored PLY, radius pick
  cli_demo.py     # Scripted demos
  test_core.py    # Core + API tests (pytest)
  test_project.py # Island project tests on a synthetic 2x2 tile set (pytest)
```

## Tips

- **Large clouds:** open by disk path, not upload. A 20M-point cloud takes a few seconds to build the LOD; lower **Max display points** in the Open dialog if the browser is slow.
- Relative paths resolve under `SyntheticPipeline/`.
- Only one browser tab should edit at a time (the server holds a single session).
