"""
Point-SAM benchmark: separate click quality from mask quality, per inference config and checkpoint.

Each tile is preprocessed once (EcomodelLite normalize + CSF ground removal + intensity filter,
the same steps as scripts/benchmark_instance_segmentation.py) and cached under
``output/pointsam_bench/_cache``. Then for every (checkpoint, config, prompt mode):

  - prompt mode "auto": clicks from the config's prompt source; click quality is scored
    against GT (stem recall / precision / duplicates).
  - prompt mode "oracle": one GT stem-base click per tree, so F1 measures mask quality only.

Results append to ``output/pointsam_bench/results.csv`` (one row per tile) and a per-run
summary is printed and written to ``<run>_summary.md``. Record noteworthy runs in
pointsam_training/EXPERIMENTS.md.

Usage (from SyntheticPipeline/):
  python pointsam_training/benchmark_pointsam.py --run E1 --configs legacy,local_grid,local_stems \
      --ckpts vanilla,decoder_ft --prompt_modes auto,oracle
  # Real (annotator-exported) test tiles in the same <name>_scan.laz/_instances.npy format:
  python pointsam_training/benchmark_pointsam.py --run E1r --tiles_dir D:/pointclouds/realtest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_SP = _HERE.parent
_ROOT = _SP.parent
for p in (str(_ROOT), str(_SP), str(_SP / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.instance_metrics import match_instances, pairwise_overlap  # noqa: E402
from promptable_instance import (  # noqa: E402
    auto_stem_grid_prompts,
    oracle_stem_prompts,
    stem_click_prompts,
    voxel_downsample_with_index,
)

CKPTS = {
    "vanilla": _ROOT / "thirdparty" / "checkpoints" / "point_sam" / "model.safetensors",
    "decoder_ft": _SP / "pointsam_checkpoints" / "decoder_ft" / "best.safetensors",
    "full_ft": _SP / "pointsam_checkpoints" / "full_ft" / "best.safetensors",
    "v2_lora": _SP / "pointsam_checkpoints" / "v2_lora" / "best.safetensors",
    "v3_mixed_lora": _SP / "pointsam_checkpoints" / "v3_mixed_lora" / "best.safetensors",
}

# Inference configurations (SegmenterPointSAM kwargs + prompt source for "auto").
CONFIGS: Dict[str, dict] = {
    "legacy": {"mode": "global", "prompts": "grid64"},
    "local_grid": {"mode": "local", "prompts": "grid"},
    "local_stems": {"mode": "local", "prompts": "stems"},
    "local_stems_neg": {"mode": "local", "prompts": "stems", "neg_neighbors": True},
    "local_stems_g1024": {"mode": "local", "prompts": "stems", "crop_group_number": 1024, "crop_group_size": 128},
    "local_stems_c16": {"mode": "local", "prompts": "stems", "crop_size": 16.0},
}


def tile_prefixes(args) -> List[Path]:
    if args.tiles_dir:
        d = Path(args.tiles_dir)
        return sorted(Path(str(f)[: -len("_scan.laz")]) for f in d.glob("*_scan.laz")
                      if Path(str(f)[: -len("_scan.laz")] + "_instances.npy").exists())
    split = json.loads(Path(args.split).read_text())
    names = split[args.split_key][: args.max_tiles] if args.max_tiles else split[args.split_key]
    return [Path(args.dataset_dir) / n for n in names]


def preprocess(prefix: Path, cache_dir: Path) -> dict:
    """Ground-removed working cloud (N,4), GT labels on it, and tile meta. Cached."""
    cache = cache_dir / f"{prefix.name}.npz"
    meta_path = Path(str(prefix) + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if cache.exists():
        z = np.load(cache)
        return {"work": z["work"], "gt": z["gt"], "meta": meta}
    import laspy
    from ecomodel_lite import EcomodelLite

    las = laspy.read(str(prefix) + "_scan.laz")
    inten = np.asarray(las.intensity, dtype=np.float64)
    inten = inten / 65535.0 if inten.max() > 1.5 else inten
    pts = np.vstack([las.x, las.y, las.z, inten]).T
    gt_full = np.load(str(prefix) + "_instances.npy").astype(np.int32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = EcomodelLite(segmenter_type="scanline", results_folder=str(cache_dir / "_tmp"))
    pts = np.hstack([pts, np.arange(len(pts), dtype=float).reshape(-1, 1)])
    pts = model.normalize_point_cloud(pts)
    pts = model.remove_ground(pts)
    pts = model.filter_intensity(pts, model.intensity_threshold)
    work = pts[:, :4].astype(np.float32)
    gt = gt_full[pts[:, 4].astype(np.int64)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, work=work, gt=gt)
    return {"work": work, "gt": gt, "meta": meta}


def auto_prompts(work: np.ndarray, source: str) -> List[np.ndarray]:
    xyz = work[:, :3].astype(np.float64)
    xyz_ds, _ = voxel_downsample_with_index(xyz, 0.05)
    if source == "grid64":
        return [p.reshape(1, 3) for p in auto_stem_grid_prompts(xyz_ds)]
    if source == "grid":
        return [p.reshape(1, 3) for p in auto_stem_grid_prompts(xyz_ds, max_prompts=100000)]
    if source == "stems":
        return stem_click_prompts(xyz_ds)
    raise ValueError(source)


def click_quality(work: np.ndarray, gt: np.ndarray, prompts: List[np.ndarray], min_tree_pts: int = 200) -> dict:
    """Stem recall: GT trees hit by >=1 first click; precision: clicks on a not-yet-hit tree."""
    from scipy.spatial import cKDTree

    ids, cnt = np.unique(gt[gt >= 0], return_counts=True)
    trees = set(ids[cnt >= min_tree_pts].tolist())
    if not prompts:
        return {"n_clicks": 0, "click_recall": 0.0, "click_precision": 0.0, "click_dup": 0, "click_miss": 0}
    d, nn = cKDTree(work[:, :3]).query(np.stack([p[0] for p in prompts]), k=1)
    hit_lab = np.where(d < 0.3, gt[nn], -1)
    seen, good, dup, miss = set(), 0, 0, 0
    for t in hit_lab.tolist():
        if t < 0:
            miss += 1
        elif t in seen:
            dup += 1
        else:
            seen.add(t)
            good += 1
    return {
        "n_clicks": len(prompts),
        "click_recall": len(seen & trees) / max(1, len(trees)),
        "click_precision": good / max(1, len(prompts)),
        "click_dup": dup,
        "click_miss": miss,
    }


def score(gt: np.ndarray, pred: np.ndarray, iou_thresh: float, ignore_unlabeled: bool) -> dict:
    """Instance F1 etc. With ignore_unlabeled, predictions mostly on unlabelled points are not scored."""
    pred = np.asarray(pred, dtype=np.int32)
    out = {}
    if ignore_unlabeled:
        pred = pred.copy()
        ids, inv = np.unique(pred, return_inverse=True)
        unl = np.bincount(inv, weights=(gt < 0).astype(np.float64)) / np.bincount(inv)
        drop = (ids >= 0) & (unl > 0.5)
        pred[drop[inv]] = -1
        out["pred_ignored"] = int(drop.sum())
    iou, _, _, inter, gt_counts, pred_counts = pairwise_overlap(gt, pred)
    out.update(match_instances(
        iou, iou_thresh=iou_thresh, inter=inter, gt_counts=gt_counts, pred_counts=pred_counts,
    ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Experiment id, e.g. E1 (see EXPERIMENTS.md)")
    ap.add_argument("--configs", default="legacy,local_stems")
    ap.add_argument("--ckpts", default="vanilla,decoder_ft",
                    help=f"Names {list(CKPTS)} or paths to .safetensors")
    ap.add_argument("--prompt_modes", default="auto,oracle")
    ap.add_argument("--dataset_dir", default=str(_SP / "testdataset" / "instance"))
    ap.add_argument("--split", default=str(_SP / "pointsam_voxelized" / "split.json"))
    ap.add_argument("--split_key", default="val")
    ap.add_argument("--tiles_dir", default="", help="Folder of <name>_scan.laz + _instances.npy (real test set)")
    ap.add_argument("--max_tiles", type=int, default=0)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--out_dir", default=str(_SP / "output" / "pointsam_bench"))
    ap.add_argument("--ignore_unlabeled", action="store_true",
                    help="Partially labelled data: skip predicted trees with >50%% of points on unlabelled (-1) GT")
    ap.add_argument("--merge", action="store_true",
                    help="Also score every prediction after merge_fragments as '<config>+merge'")
    args = ap.parse_args()

    from ecomodel_promptable_segmenters import SegmenterPointSAM

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tiles_dir:
        cache_name = Path(args.tiles_dir).name
    elif Path(args.dataset_dir).resolve() == (_SP / "testdataset" / "instance").resolve():
        cache_name = "synthetic"
    else:
        cache_name = Path(args.dataset_dir).name
    cache_dir = out_dir / "_cache" / cache_name
    prefixes = tile_prefixes(args)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    modes = [m.strip() for m in args.prompt_modes.split(",") if m.strip()]
    ckpts = {}
    for c in [c.strip() for c in args.ckpts.split(",") if c.strip()]:
        path = CKPTS.get(c, Path(c))
        if not Path(path).is_absolute():
            path = (_SP / path).resolve()
        if not Path(path).is_file():
            raise SystemExit(f"Missing checkpoint {c}: {path}")
        name = c if c in CKPTS else f"{Path(path).parent.name}/{Path(path).stem}"
        ckpts[name] = Path(path)

    print(f"[bench] {len(prefixes)} tiles, configs={configs}, ckpts={list(ckpts)}, modes={modes}", flush=True)
    tiles = {}
    for p in prefixes:
        t0 = time.time()
        tiles[p.name] = preprocess(p, cache_dir)
        print(f"  prep {p.name}: {len(tiles[p.name]['work']):,} pts ({time.time() - t0:.1f}s)", flush=True)

    prompt_cache: Dict[tuple, list] = {}
    rows = []
    for ck_name, ck_path in ckpts.items():
        seg_by_cfg = {}
        for cfg_name in configs:
            cfg = dict(CONFIGS[cfg_name])
            cfg.pop("prompts")
            seg = SegmenterPointSAM(checkpoint_path=str(ck_path), **cfg)
            if seg_by_cfg:
                seg._model = next(iter(seg_by_cfg.values()))._model
            seg_by_cfg[cfg_name] = seg
        for cfg_name, seg in seg_by_cfg.items():
            src = CONFIGS[cfg_name]["prompts"]
            for mode in modes:
                for tname, t in tiles.items():
                    work, gt = t["work"], t["gt"]
                    if mode == "oracle":
                        prompts = [p.reshape(1, 3) for p in oracle_stem_prompts(work[:, :3].astype(np.float64), gt)]
                        cq = {}
                    else:
                        key = (tname, src)
                        if key not in prompt_cache:
                            prompt_cache[key] = auto_prompts(work, src)
                        prompts = prompt_cache[key]
                        cq = click_quality(work, gt, prompts)
                    t0 = time.time()
                    if seg.mode == "global":
                        prompts_in = [p[0] for p in prompts]
                    else:
                        prompts_in = prompts
                    _, pred = seg.segment(work.astype(np.float64), prompts=prompts_in)
                    dt = time.time() - t0
                    row = {"run": args.run, "ckpt": ck_name, "config": cfg_name, "prompt_mode": mode,
                           "tile": tname, "density_class": t["meta"].get("density_class", "real"),
                           "composition": t["meta"].get("composition", ""), "duration_sec": round(dt, 2), **cq}
                    if pred is None:
                        row["error"] = "segment failed"
                    else:
                        row.update(score(gt, pred, args.iou_thresh, args.ignore_unlabeled))
                    rows.append(row)
                    print(f"  [{ck_name}|{cfg_name}|{mode}] {tname}: F1={row.get('f1', float('nan')):.3f} "
                          f"P={row.get('precision', float('nan')):.3f} R={row.get('recall', float('nan')):.3f} "
                          + (f"clickR={cq['click_recall']:.2f} clickP={cq['click_precision']:.2f} " if cq else "")
                          + f"({dt:.1f}s)", flush=True)
                    if args.merge and pred is not None:
                        from instance_annotator.merge_fragments import merge_fragments

                        t0 = time.time()
                        merged, rep = merge_fragments(work[:, :3].astype(np.float64), np.asarray(pred, dtype=np.int32))
                        mrow = dict(row, config=f"{cfg_name}+merge", duration_sec=round(dt + time.time() - t0, 2),
                                    fragments_merged=rep["merged"])
                        mrow.update(score(gt, merged, args.iou_thresh, args.ignore_unlabeled))
                        rows.append(mrow)
                        print(f"  [{ck_name}|{cfg_name}+merge|{mode}] {tname}: F1={mrow['f1']:.3f} "
                              f"P={mrow['precision']:.3f} R={mrow['recall']:.3f} merged={rep['merged']}", flush=True)
        for s in seg_by_cfg.values():
            s._model = None
        import torch
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    res = out_dir / "results.csv"
    df.to_csv(res, mode="a", header=not res.exists(), index=False)
    agg_cols = [c for c in ["f1", "precision", "recall", "pq", "mean_matched_iou",
                            "tree_precision", "tree_recall", "tree_f1", "tree_iou",
                            "click_recall", "click_precision", "click_dup", "pred_ignored", "fragments_merged",
                            "duration_sec"] if c in df.columns]
    df["hard"] = df["density_class"].isin(["dense", "extreme"])
    summ = df.groupby(["ckpt", "config", "prompt_mode"])[agg_cols].mean().round(3).reset_index()
    hard = df[df["hard"]].groupby(["ckpt", "config", "prompt_mode"])["f1"].mean().round(3).rename("f1_hard")
    summ = summ.merge(hard.reset_index(), on=["ckpt", "config", "prompt_mode"], how="left")
    md = [f"# Point-SAM benchmark run {args.run}", "",
          f"tiles: {len(tiles)} ({'real: ' + args.tiles_dir if args.tiles_dir else args.split_key + ' split'})"
          + (", predictions on unlabelled points ignored" if args.ignore_unlabeled else ""), "",
          "| " + " | ".join(summ.columns) + " |", "|" + "---|" * len(summ.columns)]
    for _, r in summ.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.values) + " |")
    text = "\n".join(md) + "\n"
    (out_dir / f"{args.run}_summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"[bench] rows appended to {res}")


if __name__ == "__main__":
    main()
