# TLS Instance Annotator (Browser)

Inspect and correct tree **instance** labels on terrestrial / MLS point clouds.

Real-TLS benchmarks showed only TreeLearn produces meaningful instances among the five methods; this tool lets you run any of them, review the result in 3D, fix obvious errors, and export corrected ground truth in the Ecomodel tile layout.

## Install

Core deps are already in the Ecomodel / PyTLidar env (`numpy`, `laspy`, `plotly`, segmenters). Add Streamlit for the UI:

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
& $py -m pip install streamlit
```

## Quick start (UI)

Run these **one line at a time** in PowerShell (do not paste the whole block as a single line):

```powershell
cd D:\Projects\Ecomodel\SyntheticPipeline
D:\Projects\PyTLidar\.venv\Scripts\python.exe -m streamlit run instance_annotator/app.py
```

Or from any directory:

```powershell
D:\Projects\PyTLidar\.venv\Scripts\python.exe -m streamlit run D:\Projects\Ecomodel\SyntheticPipeline\instance_annotator\app.py
```

Browser opens locally (default `http://localhost:8501`).

### Workflow

1. **Load**
   - Paste a **tile prefix** such as `testdataset/real_instance/l1w_t00_03` (loads `*_scan.laz` and optional `*_instances.npy`), **or** a raw `.laz` / `.ply`, **or** upload a file.
   - Start mode:
     - **Run segmentation** — blank labels, then run a method.
     - **Manual tagging only** — start all `-1`, paint trees yourself.
     - **Load existing GT** — edit an existing `*_instances.npy`.
2. **Wood / Leaf (reference, optional)** — separate wood vs leaves as a visual aid (does **not** change instance labels):
   - Methods: **Intensity percentile** (default), **Intensity threshold**, **Otsu**, **Eigenfeatures**, **Stem-grow**, **RGI**, **GBSeparation**.
   - **Run wood/leaf**, then toggle **Show wood** / **Show leaves**, and **Color by** Instance IDs or Wood / Leaf (brown = wood, green = leaf, gray = unknown).
   - On large plots prefer **Eigenfeatures** / **Stem-grow** / intensity methods; RGI is often blotchy after downsampling.
3. **Segment** (optional) — choose `scanline` / `treelearn` / `treex` / `tls2trees` / `pointsam`, optional leaf-removal preprocess, then **Run method**. TreeX uses **stock TLS** by default (good for real plots).
4. **Examine** — Plotly 3D view (downsampled for speed). Instance list on the right; click an ID to highlight / select that tree.
5. **Edit**
   - **Brush select:** click points in the plot (Streamlit ≥1.35) or enter a **seed point index** from the hover tooltip and **Add brush from seed**.
   - **Reassign** selection → another ID (or `-1`).
   - **Merge** source tree → target.
   - **Paint new / split** — selection becomes a new tree ID.
   - **Mark non-tree** — selection → `-1`.
   - **Undo / Redo** / compact IDs.
6. **Export** — writes:
   - `{name}_scan.laz`
   - `{name}_instances.npy` (`int32`, trees `>=0`, non-tree `-1`)
   - `{name}_meta.json`
   - `{name}_preview.ply` (colored)

That triplet is loadable by `scripts/benchmark_instance_segmentation.py`.

## Wood / leaf methods

| Method | Notes |
|--------|--------|
| Intensity percentile | Wood = intensity ≥ P-th percentile (default P=40). Fast exploration. |
| Intensity threshold | Wood = intensity ≥ absolute T (default = cloud median). |
| Otsu | Auto intensity threshold from histogram. Fast on full cloud. |
| Eigenfeatures | Wood = high linearity + verticality + low curvature (kNN PCA). Voxel-subsample on large clouds. |
| Stem-grow | Seed low-Z vertical points, grow by radius. Strong trunk reference. |
| RGI | Region-growing; voxel+intensity subsample then NN-paint. Often weak on multi-tree plots. |
| GBSeparation | Graph + root-path wood. Subsample on large tiles; prefer Eigen/Stem-grow/Intensity. |

Reference overlay only. Segment’s **Leaf removal (RGI)** checkbox still strips leaves before an instance method run.

## Label convention

| Label | Meaning |
|------:|---------|
| `-1` | Non-tree / ground / unlabeled |
| `>=0` | Tree instance ID |

## Method notes

| Method | Notes |
|--------|--------|
| `treelearn` | Needs TreeLearn weights + CUDA for speed. Config: `TreeLearn/configs/pipeline/ecomodel.yaml`. |
| `pointsam` | Needs `thirdparty/checkpoints/point_sam/model.safetensors`. Auto prompts only in the UI. |
| `treex` | Stock `TreeXPresetTLS` when “TreeX stock TLS” is checked. |
| `tls2trees` | In-process port; RGI semantic unless leaf-removal already ran. |
| `scanline` | Classic Ecomodel stem graph; often finds few stems on 0.1 m voxels. |

## Headless demos

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
cd D:\Projects\Ecomodel\SyntheticPipeline
& $py -m instance_annotator.cli_demo --tile testdataset/real_instance/l1w_t00_03
```

Artifacts land in `output/annotator_demos/`:

| Demo | Output |
|------|--------|
| 1 TreeLearn | `demo_treelearn_*` + HTML |
| 2 Manual paint | `demo_manual_*` + HTML |
| 3 Correct | merge + nontree on TreeLearn → `demo_corrected_*` + HTML |

Skip GPU TreeLearn:

```powershell
& $py -m instance_annotator.cli_demo --skip_treelearn --only manual
```

## Package layout

```
instance_annotator/
  app.py          # Streamlit UI
  cli_demo.py     # Scripted demos
  io.py           # LAZ/PLY + tile I/O
  labels.py       # Undoable editor
  segment.py      # Benchmark method wrapper
  viz.py          # Plotly + PLY colors
  README.md       # This guide
```

## Tips

- **Restart Streamlit** after pulling these fixes (`Ctrl+C`, then run again) so the browser gets the new app code.
- **Large files:** use a disk **path / tile prefix**, not the upload widget. Upload limit is 8 GB via [`.streamlit/config.toml`](../.streamlit/config.toml).
- Display default is **150k** points (sidebar can raise to 2M). Edits always hit the full cloud.
- **Brush select:** choose that tool, click in the plot (or seed index). Magenta = selection. Use **Add to selection** off to replace.
- If the plot feels stuck after Clear, click a different point once (Plotly may keep the old click until you pick again).
- Relative paths resolve under `SyntheticPipeline/` even if you started Streamlit elsewhere.
