# TreeLearn Fine-Tune (Synthetic Instance Data)

Domain-adapt TreeLearn (heads-only) on expanded synthetic forests, then compare vs vanilla on the **frozen val-16** from the Point-SAM split.

## Data

| Set | Source | Role |
|-----|--------|------|
| Train base | `testdataset/instance/` ∩ split.train | 48 tiles |
| Train expand | `testdataset/instance_expand/` | +64 (seed=1000) |
| Val | split.val | 16 frozen |

### 1. Generate expand tiles

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
cd D:\Projects\Ecomodel\SyntheticPipeline
& $py -u scripts/generate_instance_benchmark.py `
  --config configs/instance_benchmark_ft_expand.json `
  --seed 1000 --skip_asset_gen
```

### 2. Prepare TreeLearn forests + crops

```powershell
& $py treelearn_training/prepare_treelearn_data.py --min_expand 64 --run_gen
```

### 3. Train (heads-only)

```powershell
cd D:\Projects\Ecomodel\TreeLearn
# via wrapper (exports best.pth + freeze check):
& $py ..\SyntheticPipeline\treelearn_training\run_train.py
# smoke:
& $py ..\SyntheticPipeline\treelearn_training\run_train.py --smoke
```

Weights → `SyntheticPipeline/treelearn_checkpoints/best.pth`  
Inference config → `TreeLearn/configs/pipeline/ecomodel_ft.yaml`

### 4. Compare on val-16

```powershell
cd D:\Projects\Ecomodel\SyntheticPipeline
& $py treelearn_training/compare_vanilla_vs_ft.py
```

## Kill criterion

On val-16 TreeLearn F1:

- **Hard success:** dense+extreme ΔF1 ≥ +0.10
- **Partial success:** overall ΔF1 ≥ +0.05 without dense regression (dense may still be ~0)
- **Weak / regression:** stop; do **not** FT SNAP the same way; keep vanilla weights on regression

## Results (this run)

| model | F1 all | F1 dense+extreme | F1 sparse | F1 moderate |
|-------|-------:|-----------------:|----------:|------------:|
| vanilla | 0.144 | 0.000 | 0.450 | 0.125 |
| FT heads | **0.215** | 0.000 | **0.601** | **0.260** |

**Partial success:** overall ΔF1 +0.071; dense/extreme still 0. See `output/treelearn_ft_compare/next_steps.md`.
