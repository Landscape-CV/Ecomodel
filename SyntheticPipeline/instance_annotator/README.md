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
2. **Segment** (optional) — choose `scanline` / `treelearn` / `treex` / `tls2trees` / `pointsam`, optional leaf-removal, then **Run method**. TreeX uses **stock TLS** by default (good for real plots).
3. **Examine** — Plotly 3D view (downsampled for speed). Instance list on the right; click an ID to highlight / select that tree.
4. **Edit**
   - **Brush select:** click points in the plot (Streamlit ≥1.35) or enter a **seed point index** from the hover tooltip and **Add brush from seed**.
   - **Reassign** selection → another ID (or `-1`).
   - **Merge** source tree → target.
   - **Paint new / split** — selection becomes a new tree ID.
   - **Mark non-tree** — selection → `-1`.
   - **Undo / Redo** / compact IDs.
5. **Export** — writes:
   - `{name}_scan.laz`
   - `{name}_instances.npy` (`int32`, trees `>=0`, non-tree `-1`)
   - `{name}_meta.json`
   - `{name}_preview.ply` (colored)

That triplet is loadable by `scripts/benchmark_instance_segmentation.py`.

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
