# Point-SAM Fine-Tune (Synthetic Instance Data)

Domain-adapt the HuggingFace Point-SAM checkpoint on `testdataset/instance/`, then compare against vanilla weights on a held-out val set.

## Dataset verdict

| Item | Value |
|------|-------|
| Source | `SyntheticPipeline/testdataset/instance/` |
| Tiles | 64 (4 densities × 2 compositions × 8 replicates) |
| Instances with points | ~843 |
| Labels | per-point `*_instances.npy` (`>=0` tree, `-1` ground) |
| Split | **48 train / 16 val**, seed=0, **2 tiles per** density×composition |

**Conclusion:** Enough for **supervised fine-tuning / domain adaptation** of pretrained Point-SAM. **Not** enough to train from scratch. Ground must be filtered; voxel size **0.05 m** matches inference.

## Layout

```
SyntheticPipeline/
  pointsam_voxelized/     # prepared crops + split.json + meta.json
  pointsam_training/      # this code
  pointsam_checkpoints/   # decoder_ft/ + full_ft/ safetensors
  output/pointsam_ft_compare/  # vanilla vs FT report
```

## 1. Voxelize + split

```bash
cd SyntheticPipeline
python pointsam_training/prepare_voxel_dataset.py
# smoke:
python pointsam_training/prepare_voxel_dataset.py --smoke 2
```

Writes `pointsam_voxelized/split.json`, `train/*.npz`, `val/*.npz`, `meta.json`.

## 2. Train

Primary (freeze encoder):

```bash
cd SyntheticPipeline/pointsam_training
python train.py --config configs/forest_decoder_ft.yaml
```

Secondary (light full FT):

```bash
python train.py --config configs/forest_full_ft.yaml
```

Weights → `../pointsam_checkpoints/{decoder_ft,full_ft}/best.safetensors`.

## 3. Compare vanilla vs FT

```bash
cd SyntheticPipeline
python pointsam_training/compare_vanilla_vs_ft.py
```

Uses val tiles from `pointsam_voxelized/split.json`, prompt modes auto+oracle, writes:

- `output/pointsam_ft_compare/compare_raw.csv`
- `output/pointsam_ft_compare/compare_summary.csv`
- `output/pointsam_ft_compare/compare_report.md`
- `output/pointsam_ft_compare/next_steps.md`  ← SNAP / TreeLearn gate

## Decision rule (SNAP / TreeLearn)

See `next_steps.md` after compare:

- **FT helps on dense/extreme (ΔF1 ≥ ~0.03)** → keep iterating Point-SAM or try TreeLearn domain adapt.
- **Modest overall gain** → try SNAP Outdoor next on the same voxelized GT.
- **Little/no gain** → prefer SNAP Outdoor fine-tune, then TreeLearn.

## Verify FT checkpoints

```bash
python pointsam_training/verify_ft_checkpoints.py
# → pointsam_checkpoints/verify_report.json
```

Expected:
- `decoder_ft`: encoder **unchanged**, ~135 non-encoder tensors moved (real FT, not a failed save of vanilla).
- `full_ft`: encoder **and** decoder tensors changed.

## VRAM guardrails

Training on RTX 5070 Ti (~17GB) can look like “GPU locked at max” even when training is healthy — Point-SAM ViT-L activations are large. `train.py` now:

- Caps process memory fraction (`cuda_mem_fraction: 0.92`)
- Soft-trims cache at `vram_soft_frac`, **aborts** at `vram_hard_frac`
- Logs `vram% / alloc / peak` each `log_every` steps
- For `decoder_ft`, runs `pc_encoder` under `torch.no_grad()` + detach (much lower peak)

Smoke tests must use a separate out dir so they do not overwrite real weights:

```bash
python train.py --config configs/forest_decoder_ft.yaml --max_steps 4 --out_dir ../pointsam_checkpoints/_smoke
```

## Requirements

Same env as Point-SAM inference (`thirdparty/Point-SAM/INSTALL_ECOMODEL.md`): use **`D:\Projects\PyTLidar\.venv`** (Python 3.11), CUDA, `torkit3d`, `hydra-core`, `safetensors`, `timm`, `laspy`, `pyyaml`, `pandas`.

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
cd D:\Projects\Ecomodel\SyntheticPipeline
& $py pointsam_training/prepare_voxel_dataset.py
cd pointsam_training
& $py -u train.py --config configs/forest_decoder_ft.yaml
cd ..
& $py -u pointsam_training/compare_vanilla_vs_ft.py
```

## Results (val-16, seed=0, leaf-on)

Mask-IoU on crop val subset during training: decoder_ft **0.712**, full_ft **0.708**.

Instance-seg F1 / PQ on held-out tiles (`output/pointsam_ft_compare/`):

| model | auto F1 (all) | auto F1 (dense+extreme) | oracle F1 (all) |
|-------|--------------:|------------------------:|----------------:|
| vanilla | 0.232 | 0.037 | 0.230 |
| decoder_ft | 0.212 | **0.081** | **0.296** |
| full_ft | 0.233 | 0.055 | 0.238 |

## Next steps (SNAP / TreeLearn gate)

Decoder FT **helps dense/extreme** (ΔF1 +0.044) and improves oracle prompts, but slightly regresses overall auto F1 (easy strata). Prefer keeping `pointsam_checkpoints/decoder_ft/best.safetensors` for hard scenes.

**Recommendation:** Iterate Point-SAM (more crops / longer decoder FT) **or** try **TreeLearn domain adaptation** next (already weak on dense/extreme in the README smoke). Defer SNAP Outdoor fine-tune until after a TreeLearn adapt pass, unless TreeLearn remains flat.
