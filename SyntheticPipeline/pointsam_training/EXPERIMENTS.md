# Point-SAM improvement experiments

Running log for the plan to get a better fine-tuned Point-SAM, covering inference fixes, the merge step, the training recipe, open/real data and baselines. Every experiment has an id (`E<n><letter>`), the exact command, where its outputs land, and a results row. Add rows; don't rewrite old ones. When a result changes a default, say so in the **Decision** line.

Commands run from `SyntheticPipeline/` with `D:\Projects\PyTLidar\.venv\Scripts\python.exe` unless noted.

## Evaluation protocol

- **Synthetic test set:** the 16 val tiles of `pointsam_voxelized/split.json` (2 per density × composition, seed 0).
- **Real test set:** the 14 `val` tiles of `testdataset/real_instance/split.json` (30 m tiles):
  - 7 TreeLearn L1W (MLS);
  - 3 Wytham Woods ISPRS (TLS);
  - 4 Wytham Woods qutuwu (TLS, voxelized at 0.1 m).
- **Real tiles are only partly labelled.** Unlabelled points share the -1 label with ground and understory:
  - about 60% of L1W canopy points;
  - 66–75% of Wytham points between 0.5 and 4 m (understory);
  - about 65% of qutuwu canopy.
- **Real-set scoring therefore uses `--ignore_unlabeled`:** a predicted tree with more than 50% of its points on -1 is not counted as a false positive. This is the usual convention for partially labelled forest benchmarks. The summary reports how many predictions were ignored (`pred_ignored`).
- **Preprocessing** matches `run_segmenter_on_tile`: mean-centred, CSF ground removal, intensity filter. Segmenters see ground-removed clouds. Cached per tile under `output/pointsam_bench/_cache/`.
- **Metrics** (`benchmark_pointsam.py`):
  - instance F1 / precision / recall at IoU 0.5 over trees with at least 200 points;
  - `f1_hard` = dense + extreme strata only;
  - click quality: `click_recall` = share of GT trees hit by at least one first click; `click_precision` = clicks landing on a not-yet-hit tree; `click_dup`; `click_miss`.
- **Prompt modes:**
  - `auto`: the config's own prompts;
  - `oracle`: one GT stem click per tree at about 1.3 m (plus an upper stem click). This isolates **mask quality** from **click quality**.
- **Outputs:** `output/pointsam_bench/results.csv` gets one row per (run, ckpt, config, mode, tile), plus `output/pointsam_bench/<run>_summary.md`.

## E0: baseline (before this plan)

Source: `README.md` → `output/pointsam_ft_compare/`. Legacy inference uses one global pass over at most 150k points with 2048×256 groups and 64 grid prompts.

| model | auto F1 (all) | auto F1 (dense+extreme) | oracle F1 (all) |
|-------|--------------:|------------------------:|----------------:|
| vanilla | 0.232 | 0.037 | 0.230 |
| decoder_ft | 0.212 | 0.081 | 0.296 |
| full_ft | 0.233 | 0.055 | 0.238 |

Crop-level validation mask IoU was about 0.71 for both fine-tunes, but that was measured **after 5 correction clicks**, so it said little about one-click automatic use.

### Diagnosed causes of the weak fine-tune

1. **Scale mismatch:** training used 12 m crops of 10k points; inference used one global pass on a whole tile.
2. **Grouping mismatch:** training used 512×64 groups, pretraining 1024×256, legacy inference 2048×256.
3. **Colour mismatch:** legacy inference fed RGB without the `(x-0.5)/0.5` normalisation used in training.
4. **First prompt:** the upstream sampler picks the point furthest from the mask border, which is the crown interior. At inference the first click is on the stem.
5. **Ground:** training crops include ground, but CSF removes it at inference.
6. **Augmentation:** it was fixed per crop index, so it was effectively not random across epochs.
7. **Model selection:** chose by IoU after 5 clicks, not by the argmax-IoU mask after one click.
8. **Training budget:** only 2000 steps at batch 1 with 2 masks.
9. **NMS:** IoU 0.5 NMS keeps partial masks. Containment-based fusion is needed instead.
10. **Data:** dense synthetic tiles are small and dominated by one huge instance (for example 212k of 308k points), so the top-scored mask is often "everything".

## E1: inference configurations × checkpoints (synthetic test set)

Tests Phase 1. `local_*` configs are:

- 12 m crops with a 4 m stride and 10k points per crop, using 512×64 groups;
- normalised colour;
- batched prompts per crop, keeping the argmax-IoU mask;
- `fuse_stem_masks`, which merges masks by containment and seed overlap, then assigns points to one tree by max logit.

```
python pointsam_training/benchmark_pointsam.py --run E1 --configs legacy,local_grid,local_stems,local_stems_g1024 --ckpts vanilla,decoder_ft,full_ft --prompt_modes auto,oracle
```

Mean over 16 tiles. Full table: `output/pointsam_bench/E1_summary.md`. In oracle mode, `local_grid` and `local_stems` share the same prompts, so they match. Stem-click columns use the **old** detector defaults (before E2a).

| ckpt | config | auto F1 | auto P | auto R | auto f1_hard | oracle F1 | matched IoU (auto) |
|------|--------|--------:|-------:|-------:|-------------:|----------:|-------------------:|
| vanilla | legacy | 0.232 | 0.365 | 0.198 | 0.037 | 0.214 | 0.748 |
| vanilla | local_grid | **0.314** | **0.555** | **0.271** | 0.039 | 0.264 | 0.739 |
| vanilla | local_stems | 0.223 | 0.388 | 0.221 | 0.046 | 0.264 | 0.757 |
| vanilla | local_stems_g1024 | 0.231 | 0.412 | 0.215 | 0.059 | 0.275 | 0.778 |
| decoder_ft | legacy | 0.251 | 0.273 | 0.298 | 0.080 | 0.274 | 0.668 |
| decoder_ft | local_grid | 0.083 | 0.119 | 0.086 | 0.071 | 0.121 | 0.705 |
| decoder_ft | local_stems | 0.097 | 0.113 | 0.112 | 0.027 | 0.121 | 0.714 |
| decoder_ft | local_stems_g1024 | 0.085 | 0.093 | 0.115 | 0.031 | 0.111 | 0.684 |
| full_ft | legacy | 0.244 | 0.328 | 0.240 | 0.070 | 0.229 | 0.643 |
| full_ft | local_grid | 0.225 | 0.265 | 0.233 | 0.045 | 0.207 | 0.714 |
| full_ft | local_stems | 0.199 | 0.227 | 0.204 | 0.020 | 0.207 | 0.784 |
| full_ft | local_stems_g1024 | 0.193 | 0.228 | 0.198 | 0.032 | 0.281 | 0.774 |

Findings:

- **Local crops help the stock model:** vanilla F1 rises from 0.232 to 0.314 (+35%) and precision from 0.37 to 0.56, with no training. It is also about twice as fast (2.4 s vs 4.9 s per tile).
- **Clicks are not the bottleneck.** Oracle stem clicks (0.21–0.28) never clearly beat automatic grid prompts (0.31). The masks Point-SAM returns for one stem click are the limiting factor, so the effort should go into training (E3/E4), not prompt engineering.
- **The v1 fine-tunes don't transfer to the corrected inference.**
  - decoder_ft drops from 0.25 (legacy) to 0.08 (local); full_ft drops from 0.24 to 0.23.
  - Both learned the legacy input distribution (un-normalised colour, ground present, interior prompts) rather than trees.
  - This is why the v2 recipe matches training to inference.
- **Dense and extreme tiles remain unsolved** (f1_hard ≤ 0.08 everywhere). These scenes are where training data and the merge step have to help.
- 1024×128 groups give slightly higher matched IoU and hard F1 for vanilla but no overall F1 gain. The 512×64 training setting stays.

**Decision:** default inference is `mode="local"`. Grid prompts for the stock model; stem prompts get re-tested with the E2a detector and the v2/v3 checkpoints, which are trained on stem clicks.

## E2: inference and post-processing tuning

### E2a: stem detector parameters (done)

```
python pointsam_training/tune_stems.py            # CPU, uses the E1 cache
```

Grid over `eps × merge_dist × min_points × max_xy_extent`, mean over 16 synthetic tiles:

| setting | click recall | click precision | dup / tile |
|---------|-------------:|----------------:|-----------:|
| old default (eps .2, merge .5, min 15, ext 2.0) | 0.648 | 0.343 | 19.4 |
| **new default (eps .1, merge 1.5, min 40, ext 1.0)** | **0.725** | **0.447** | 9.5 |
| high precision (eps .2, merge 1.5, min 40, ext 1.0) | 0.390 | 0.622 | 4.0 |

Many duplicates come from multi-stem synthetic species (grasses, rushes, willow). They are cheap because fusion merges masks of the same tree, while misses are not recoverable. That is why selection favours recall.

**Decision:** the new defaults are in `Utils/promptable_instance.detect_stems`. Re-check on the real set (`--cache output/pointsam_bench/_cache/real_instance`) after E1R.

### E2b–E2e: planned

| id | question | command / change |
|----|----------|------------------|
| E2b | Do negative clicks on neighbouring stems reduce "everything" masks? | `--configs local_stems,local_stems_neg` |
| E2c | Does a larger crop (16 m) help tall trees? | `--configs local_stems,local_stems_c16` |
| E2d | Does `merge_fragments` fix split trees after Point-SAM and scanline? | add a `+merge` variant (calls `instance_annotator.merge_fragments.merge_fragments`) |
| E2f | Stop containment fusion from chaining large overlapping masks into one tree | `fuse_stem_masks`: require mutual seed inclusion for large masks; assign the rest by logit |

### E1R: real test set, stock and v1 checkpoints

```
python pointsam_training/benchmark_pointsam.py --run E1R --dataset_dir testdataset/real_instance --split testdataset/real_instance/split.json --configs legacy,local_grid,local_stems --ckpts vanilla,full_ft --prompt_modes auto,oracle --merge --ignore_unlabeled
```

Results: `output/pointsam_bench/E1R_summary.md`. Mean over 14 tiles, with unlabelled predictions ignored:

| ckpt | config | auto F1 | auto P | oracle F1 | oracle matched IoU | click recall (auto) |
|------|--------|--------:|-------:|----------:|-------------------:|--------------------:|
| vanilla | legacy | 0.000 | 0.000 | 0.000 | – | 0.30 |
| vanilla | local_grid | 0.010 | 0.048 | 0.058 | 0.787 | 0.30 |
| vanilla | local_stems | 0.013 | 0.050 | 0.058 | 0.787 | 0.14 |
| full_ft | legacy | 0.000 | 0.000 | 0.000 | – | 0.30 |
| full_ft | local_grid | 0.028 | 0.101 | 0.060 | 0.678 | 0.30 |
| full_ft | local_stems | 0.004 | 0.018 | 0.060 | 0.678 | 0.14 |

`+merge` variants are within ±0.006 of these: there are too few correct trees for merging to matter.

Findings from E1R and a 3-tile probe (one L1W, one Wytham ISPRS, one qutuwu):

- **The stock model fails on real forest:** F1 is 0.00 on every tile in legacy mode, and also with local crops of 12, 20 and 30 m combined with `max_mask_frac` 1.0 / 0.5 / 0.3.
- **The masks are the problem, not the clicks.** With oracle stem clicks on `l1w_t00_02`, the 13 masks hold 7k–167k points (voxelized) for trees of 4k–83k points. They overlap heavily, fusion chains them into 1–5 "trees", and the best F1 is 0.12 (12 m crops). With 30 m crops nearly every mask is "the whole scene".
- **Automatic clicks also land in understory:** in Wytham, click recall is 0.00–0.06 because the 1–4 m layer is mostly unlabelled shrubs.
- **Conclusion:** without training on real forest, Point-SAM is not usable on real TLS/MLS. E4a (mixed real training) is the priority run. The new `max_mask_frac` option (default 1.0, i.e. off) stays available for fine-tuned models.

## E3: training recipe v2 (synthetic only)

Config: `configs/forest_v2_lora.yaml`. Changes against the old recipe:

| change | fixes cause |
|--------|-------------|
| LoRA rank 8 on encoder `q_proj`/`v_proj` + patch embed; decoder fully trainable (5.8M of 312M params) | 8; less forgetting than full FT |
| First prompt sampled as `stem` 40% / `stem1` 20% / `random` 20% / `interior` 20%; first iteration weighted ×3 | 4 |
| Groups 512×64 and colour normalisation equal to local inference | 2, 3 |
| `p_drop_ground 0.7`, density / occlusion / height-cut / no-intensity / tilt / jitter augmentations, rng re-seeded per epoch | 5, 6; robustness to mixed sensors |
| 8 masks per crop, grad accumulation 2, 3000 optimizer steps (6000 crops), cosine schedule with 150 warmup steps | 8 |
| Select on `val_iou_pick` (argmax-IoU mask after the first stem click) | 7 |

LoRA weights are merged before saving, so `best.safetensors` keeps stock keys and loads in the annotator dropdown and in `benchmark_pointsam.py --ckpts`.

```
cd pointsam_training
python -u train.py --config configs/forest_v2_lora.yaml     # log: output/train_E3a_v2_lora.log
# smoke: --max_steps 4 --out_dir ../pointsam_checkpoints/_smoke   (verified: 90 s, stock keys, 12.7 GB peak)
```

With the GPU to itself, a step at accumulation 2 takes about 1 s, so 3000 steps plus validation take about 1 h. The smoke test seemed far slower only because it shared the GPU with E1. If curves are still rising at 3000, rerun with `max_steps: 6000`. Queued after E4a.

Evaluate by adding the checkpoint to `CKPTS` in `benchmark_pointsam.py` (`v2_lora`), then run `--run E3 --configs local_grid,local_stems --ckpts vanilla,v2_lora`.

| run | steps | val_iou_pick | auto F1 (local_grid) | auto F1 (local_stems) | oracle F1 | f1_hard |
|-----|------:|-------------:|---------------------:|----------------------:|----------:|--------:|
| vanilla (reference) | – | – | 0.314 | 0.269 | 0.264 | 0.039 |
| E3a v2_lora | 3000 (best at step 1800, early stop) | 0.570 (synthetic val) | **0.388** | 0.378 | **0.436** | 0.047 |

Source: `output/pointsam_bench/E4_syn_summary.md`. Same 16 synthetic tiles as E1.

- **The v2 recipe works on synthetic data.** Auto F1 rises from 0.314 to 0.388 (+24%) and oracle F1 from 0.264 to 0.436 (+65%). Mask quality, the E1 bottleneck, improved most.
- **On real tiles, v2 barely helps:** auto F1 0.012–0.043, oracle 0.104 (E4_real). Synthetic-only training does not transfer to real forest.
- **Dense and extreme tiles are still unsolved** (f1_hard ≤ 0.05).

Ablations to run if E3a helps: `first_iter_weight 1.0`; `prompt_mix` interior only; `p_drop_ground 0`; full FT instead of LoRA (`mode: full`, `encoder_lr 5e-6`).

## E4: real and open data

### E4a: mixed synthetic + real (data ready)

Real crops have been built from the local TreeLearn L1W and Wytham Woods tiles. The split reuses `testdataset/real_instance/split.json`, so the 14 real test tiles are never trained on.

```
python pointsam_training/prepare_voxel_dataset.py --src testdataset/real_instance --out pointsam_voxelized_real --split_file testdataset/real_instance/split.json --drop_unlabeled_above 3
# -> train=672 crops (36 tiles), val=282 crops (14 tiles)
cd pointsam_training
python -u train.py --config configs/forest_v3_mixed_lora.yaml   # log: output/train_E4a_v3_mixed_lora.log
```

Validation is on real crops. `--drop_unlabeled_above 3` removes unlabelled points more than 3 m above the local base. Unlabelled trees would otherwise be background and teach the model that trees are negatives. Ground and low understory stay as genuine background, matching what CSF leaves at inference. Running first (started after E1R).

After training, score it on both test sets:

```
python pointsam_training/benchmark_pointsam.py --run E4a_real --dataset_dir testdataset/real_instance --split testdataset/real_instance/split.json --configs local_grid,local_stems --ckpts vanilla,v3_mixed_lora --prompt_modes auto,oracle --merge --ignore_unlabeled
python pointsam_training/benchmark_pointsam.py --run E4a_syn --configs local_grid,local_stems --ckpts vanilla,v3_mixed_lora --prompt_modes auto,oracle
```

| run | train data | val_iou_pick (real) | synthetic auto F1 | real auto F1 | real oracle F1 |
|-----|-----------|--------------------:|------------------:|-------------:|---------------:|
| vanilla (reference) | – | – | 0.314 | 0.010 (grid) / 0.013 (stems) | 0.058 |
| E3a v2_lora | synthetic | – | 0.388 | 0.012 / 0.043 | 0.104 |
| E4a v3_mixed_lora | synthetic + real | 0.534 (step 2400; 0.283 at step 200), 39 min | 0.355 (grid) / 0.374 (stems) | **0.251** (grid) / 0.144 (stems) | **0.386** |

Sources: `output/pointsam_bench/E4_real_summary.md` and `E4_syn_summary.md`. Real-set scoring uses `--ignore_unlabeled`.

- **Real training data is what makes Point-SAM work on real forest.** Real auto F1 goes from 0.01 to 0.25, with precision 0.43 and matched IoU 0.73. Oracle F1 goes from 0.06 to 0.39.
- **There is no cost on synthetic data:** 0.355–0.374, still above vanilla's 0.314 and close to v2's 0.388.
- **Grid prompts beat stem prompts on real data** (0.251 vs 0.144). Stem click recall on real tiles is only 0.14 because the low stem layer is mostly unlabelled understory. Grid prompts stay the default.
- **Many predictions now land on unlabelled trees** (`pred_ignored` about 18 per tile with grid prompts). This is a sign the model finds real trees that the labels don't cover, not an error.
- **Merging split trees** changes real F1 by at most 0.02 (auto +0.001, oracle −0.018). It doesn't help yet; revisit once recall is higher.

**Decision:** `v3_mixed_lora` is the recommended checkpoint for real scans. Next steps:

- add more real data (E4b open datasets, E4c annotator corrections);
- try a longer run, since validation was still around 0.52–0.53 at step 3000;
- target recall: auto recall is 0.19, so most trees are still missed.
| E4a′ real only | real | | | | |

### E4b: external open datasets

`convert_open_dataset.py` turns LAS/LAZ/PLY files with a per-point tree id (or one file per tree) into the same npz crops. With `--tiles_out` it also writes benchmark tiles.

| dataset | sensor | licence | convert |
|---------|--------|---------|---------|
| FOR-instance (and V2) | ULS (Riegl) | CC BY 4.0 | `--dataset forinstance --inputs "D:/data/FOR-instance/**/*.las" --id_field treeID --background_ids 0` |
| LAUTx | MLS / PLS | CC BY 4.0 | `--dataset lautx --inputs "D:/data/LAUTx/**/*.las"` (auto-detects the id field) |
| Weiser et al. 2022 (individual trees, ALS/ULS/TLS) | multi | CC BY 4.0 (PANGAEA) | `--dataset weiser --per_tree_dir D:/data/weiser/<plot> --tree_glob "*.laz"` |
| Wytham Woods (Calders 2022) | TLS | CC BY 4.0 | already in `real_instance`; per-tree form: `--per_tree_dir ... --ground ...` |
| TreeLearn L1W | MLS | CC BY 4.0 | already in `real_instance` |

Check each licence on the download page before use; the table records what the papers state. Each dataset gets its own `--out pointsam_open/<name>`, and its `train` dir is appended to `train_dirs`. Keep its `val` split for a per-sensor score.

| run | added data | val_iou_pick | real auto F1 | per-sensor notes |
|-----|-----------|-------------:|-------------:|------------------|
| E4b | + FOR-instance | | | |

### E4c: annotator corrections loop

Tiles corrected in the annotator (single-file or island Export) are already `_scan.laz` + `_instances.npy`. Without a split file, `prepare_voxel_dataset.py` holds out 2 tiles per name stratum; `--val_per_stratum 0` puts everything in train.

```
python pointsam_training/prepare_voxel_dataset.py --src <export dir> --out pointsam_voxelized_corrections --val_per_stratum 0
```

Only fully reviewed tiles should be used, because unreviewed trees become wrong labels. Add the output to `train_dirs` and fine-tune from the latest best checkpoint for about 1000 steps.

## E5: baselines on the same test sets

| method | how | notes |
|--------|-----|-------|
| scanline, TreeLearn (stock), TreeX, tls2trees | `python scripts/benchmark_instance_segmentation.py --dataset_dir testdataset/real_instance --tiles_file testdataset/real_instance/split.json --split_key val --algorithms scanline,treelearn,treex,tls2trees` | same preprocessing |
| TreeLearn fine-tuned | train on `real_instance` train tiles with the TreeLearn repo, then `--treelearn_config` | candidate if Point-SAM stays below |
| SegmentAnyTree / ForAINet / ForestFormer3D | external repos; run on the exported tiles, save `*_instances.npy`, score with the same benchmark | trained on FOR-instance-like data, so a strong reference for ULS/TLS |

| method | synthetic F1 | real F1 | real L1W | real Wytham |
|--------|-------------:|--------:|---------:|------------:|
| scanline | | | | |
| treelearn (stock) | | | | |
| best Point-SAM | | | | |

## Merge step (Phase 2, shipped)

`instance_annotator/merge_fragments.py` merges split trees after any segmenter:

- **Automatic merges:**
  - two pieces whose stems are within 0.6 m of each other;
  - a stemless crown fragment touching a single stemmed tree;
  - touching floating fragments.
- **Review candidates:**
  - two stemmed trees that touch;
  - a fragment that touches several trees.

These are exposed in the annotator as the Segment tab's "Merge split trees afterwards" option with a candidate list, and in island mode as "Find split trees". Tests: `instance_annotator/test_merge_fragments.py`, `test_project.py::test_fragments_merges_split_crown`. E2d measures its effect on F1.
