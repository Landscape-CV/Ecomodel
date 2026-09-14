# pointtree (treeX) — Ecomodel install notes

Upstream: [ai4trees/pointtree](https://github.com/ai4trees/pointtree) (`pip install pointtree`).

## Ecomodel wiring

- Segmenter: `ecomodel_segmenters.SegmenterTreeX`
- Algorithm key: `treex`
- Starts from `TreeXPresetTLS`; default `adapt_synthetic=True` relaxes stem-search /
  circle-fit thresholds for sparser synthetic Open3D TLS (stock TLS often finds 0 stems).

## Proven stack (this machine)

| Piece | Value |
|-------|--------|
| Python | `D:\Projects\PyTLidar\.venv` (3.11) |
| Package | `pointtree==1.0.1` |
| Notes | `torch-scatter` may warn CUDA 12.8 vs 13.3; TreeX is algorithmic (CPU OK) |

## Install

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
& $py -m pip install pointtree
# If torch-scatter/cluster needed for other pointtree ops, prefer matching wheels;
# TreeX TLS path does not require a CUDA build of those extensions.
```

## Smoke

```powershell
& $py -c "from pointtree.instance_segmentation import TreeXAlgorithm, TreeXPresetTLS; print(TreeXPresetTLS())"
```
