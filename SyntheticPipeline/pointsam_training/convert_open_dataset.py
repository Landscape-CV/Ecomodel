"""
Convert open tree-instance datasets into Point-SAM training crops (and optional benchmark tiles).

Supported inputs:
  * LAS/LAZ/PLY files carrying a per-point tree id field (FOR-instance / FOR-instanceV2 `treeID`,
    LAUTx, TreeLearn L1W, Weiser 2022, ...). Non-tree points (id 0 / negative / --background_ids,
    or classes outside --tree_classes) become background (-1).
  * --per_tree_dir: one file per tree (e.g. Wytham Woods), optionally with a --ground file.

Output layout matches prepare_voxel_dataset.py, so the directories plug into `train_dirs` / `val_dir`:
  <out>/{train,val}/<dataset>_<file>_cropNNN.npz   (xyz, rgb, instance_ids, tile, origin_xy)
  <out>/meta.json
With --tiles_out, each file is also written as an annotator/benchmark triplet
(<name>_scan.laz, <name>_instances.npy, <name>_meta.json) for `benchmark_pointsam.py --tiles_dir`.

Usage (from SyntheticPipeline/):
  python pointsam_training/convert_open_dataset.py --dataset forinstance \
      --inputs D:/data/FOR-instance/**/*.las --out pointsam_open/forinstance --val_fraction 0.2
  python pointsam_training/convert_open_dataset.py --dataset wytham \
      --per_tree_dir D:/data/wytham/trees --ground D:/data/wytham/ground.ply --out pointsam_open/wytham
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_SP = _HERE.parent
for p in (_SP.parent, _SP, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prepare_voxel_dataset import drop_high_unlabeled, write_tile_crops  # noqa: E402

ID_FIELDS = ("treeID", "treeid", "tree_id", "TreeID", "instance", "instance_id", "PredInstance", "segment_id")
CLASS_FIELDS = ("classification", "semantic", "semantic_seg", "label")


def _inten01(raw: Optional[np.ndarray], n: int) -> np.ndarray:
    if raw is None or len(raw) == 0:
        return np.full(n, 0.5)
    raw = np.asarray(raw, dtype=np.float64)
    mx = float(np.nanmax(raw))
    if mx <= 0:
        return np.full(n, 0.5)
    return np.clip(raw if mx <= 1.5 else raw / 65535.0, 0.0, 1.0)


def _pick(names: Sequence[str], wanted: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if wanted:
        if wanted not in names:
            raise SystemExit(f"field {wanted!r} not found; available: {sorted(names)}")
        return wanted
    low = {n.lower(): n for n in names}
    return next((low[c.lower()] for c in candidates if c.lower() in low), None)


def load_labelled(path: Path, id_field: Optional[str], class_field: Optional[str]):
    """Return xyz (N,3) f64, intensity01 (N,), ids (N,) int64 or None, classes (N,) or None."""
    suf = path.suffix.lower()
    if suf in (".las", ".laz"):
        import laspy

        las = laspy.read(str(path))
        names = list(las.point_format.dimension_names)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        inten = _inten01(np.asarray(las.intensity) if "intensity" in names else None, len(xyz))
        get = lambda f: np.asarray(las[f])  # noqa: E731
    elif suf == ".ply":
        from plyfile import PlyData

        v = PlyData.read(str(path))["vertex"]
        names = list(v.data.dtype.names)
        xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
        key = next((k for k in ("intensity", "Intensity", "scalar_Intensity") if k in names), None)
        inten = _inten01(np.asarray(v[key]) if key else None, len(xyz))
        get = lambda f: np.asarray(v[f])  # noqa: E731
    elif suf in (".txt", ".xyz", ".csv", ".pts"):
        arr = np.loadtxt(path, delimiter="," if suf == ".csv" else None, ndmin=2)
        return arr[:, :3].astype(np.float64), _inten01(arr[:, 3] if arr.shape[1] > 3 else None, len(arr)), None, None
    else:
        raise SystemExit(f"unsupported format: {path}")
    idf = _pick(names, id_field, ID_FIELDS)
    clf = _pick(names, class_field, CLASS_FIELDS)
    ids = get(idf).astype(np.int64) if idf else None
    cls = get(clf).astype(np.int64) if clf else None
    return xyz, inten, ids, cls


def to_instance_ids(
    ids: np.ndarray,
    cls: Optional[np.ndarray],
    background_ids: Sequence[int],
    tree_classes: Optional[Sequence[int]],
    min_tree_points: int,
) -> np.ndarray:
    """Map dataset ids to contiguous tree ids >= 0, background -1."""
    bg = (ids < 0) | np.isin(ids, list(background_ids))
    if tree_classes is not None and cls is not None:
        bg |= ~np.isin(cls, list(tree_classes))
    out = np.full(len(ids), -1, dtype=np.int32)
    uniq, inv, cnt = np.unique(ids[~bg], return_inverse=True, return_counts=True)
    keep = cnt >= min_tree_points
    remap = np.full(len(uniq), -1, dtype=np.int32)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    out[~bg] = remap[inv]
    return out


def load_per_tree_dir(tree_dir: Path, ground: Optional[Path], pattern: str):
    xyzs, intens, ids = [], [], []
    files = sorted(tree_dir.glob(pattern))
    for i, f in enumerate(files):
        xyz, inten, _, _ = load_labelled(f, None, None)
        xyzs.append(xyz)
        intens.append(inten)
        ids.append(np.full(len(xyz), i, dtype=np.int32))
    if ground is not None:
        xyz, inten, _, _ = load_labelled(ground, None, None)
        xyzs.append(xyz)
        intens.append(inten)
        ids.append(np.full(len(xyz), -1, dtype=np.int32))
    if not xyzs:
        raise SystemExit(f"no files matching {pattern} in {tree_dir}")
    return np.concatenate(xyzs), np.concatenate(intens), np.concatenate(ids), len(files)


def split_of(name: str, val_fraction: float, seed: int, forced: str) -> str:
    if forced != "auto":
        return forced
    return "val" if (zlib.crc32(f"{seed}:{name}".encode()) % 10_000) / 10_000 < val_fraction else "train"


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        files += [Path(h) for h in (hits or [pat]) if Path(h).is_file()]
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description="Open tree-instance datasets -> Point-SAM crops")
    ap.add_argument("--dataset", required=True, help="Short tag used as a filename prefix (e.g. forinstance)")
    ap.add_argument("--inputs", nargs="*", default=[], help="Files or globs (** allowed)")
    ap.add_argument("--per_tree_dir", type=Path, default=None, help="Directory with one file per tree")
    ap.add_argument("--tree_glob", default="*.ply")
    ap.add_argument("--ground", type=Path, default=None, help="Ground/background file for --per_tree_dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiles_out", type=Path, default=None, help="Also write benchmark tile triplets here")
    ap.add_argument("--id_field", default=None, help=f"Instance field (auto: {', '.join(ID_FIELDS)})")
    ap.add_argument("--class_field", default=None)
    ap.add_argument("--tree_classes", default=None,
                    help="Comma list of semantic classes that belong to trees (others -> background)")
    ap.add_argument("--background_ids", default="0", help="Comma list of instance ids meaning 'no tree'")
    ap.add_argument("--min_tree_points", type=int, default=200)
    ap.add_argument("--drop_unlabeled_above", type=float, default=None,
                    help="Partially labelled data: drop background points higher than this (m) above the local base")
    ap.add_argument("--split", choices=("auto", "train", "val"), default="auto")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--crop_size", type=float, default=12.0)
    ap.add_argument("--stride", type=float, default=6.0)
    ap.add_argument("--min_points", type=int, default=2048)
    ap.add_argument("--max_points", type=int, default=100000)
    ap.add_argument("--min_instances", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bg_ids = [int(s) for s in args.background_ids.split(",") if s.strip()]
    tree_classes = [int(s) for s in args.tree_classes.split(",")] if args.tree_classes else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, callable]] = []
    if args.per_tree_dir is not None:
        jobs.append((args.per_tree_dir.name, lambda: load_per_tree_dir(args.per_tree_dir, args.ground, args.tree_glob)[:3]))
    for f in expand_inputs(args.inputs):
        def _load(f=f):
            xyz, inten, ids, cls = load_labelled(f, args.id_field, args.class_field)
            if ids is None:
                raise SystemExit(f"{f}: no instance field found (use --id_field)")
            return xyz, inten, to_instance_ids(ids, cls, bg_ids, tree_classes, args.min_tree_points)
        jobs.append((f.stem, _load))
    if not jobs:
        raise SystemExit("nothing to convert: pass --inputs or --per_tree_dir")

    stats = []
    for stem, load in jobs:
        name = f"{args.dataset}_{stem}".replace(" ", "_")
        xyz, inten, ids = load()
        xyz = xyz - xyz.min(axis=0)  # UTM offsets would lose float32 precision
        if args.drop_unlabeled_above is not None:
            keep = drop_high_unlabeled(xyz, ids, args.drop_unlabeled_above)
            xyz, inten, ids = xyz[keep], inten[keep], ids[keep]
        split = split_of(name, args.val_fraction, args.seed, args.split)
        rgb = np.repeat(inten.reshape(-1, 1), 3, axis=1)
        st = write_tile_crops(
            name, xyz, rgb, ids, out / split,
            voxel_size=args.voxel_size, crop_size=args.crop_size, stride=args.stride,
            min_points=args.min_points, max_points=args.max_points,
            min_instances=args.min_instances, seed=args.seed,
        )
        st["split"] = split
        stats.append(st)
        print(f"[{split}] {name}: pts={len(xyz):,} trees={st['n_instances']} crops={st['n_crops']}", flush=True)
        if args.tiles_out is not None:
            from instance_annotator.io import save_tile

            save_tile(args.tiles_out.resolve(), name, xyz, inten, ids,
                      {"source": args.dataset, "split": split}, write_preview_ply=False)

    meta = {
        "dataset": args.dataset,
        "args": {k: str(v) for k, v in vars(args).items()},
        "tiles": stats,
        "n_crops_train": sum(t["n_crops"] for t in stats if t["split"] == "train"),
        "n_crops_val": sum(t["n_crops"] for t in stats if t["split"] == "val"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done: train crops={meta['n_crops_train']} val crops={meta['n_crops_val']} -> {out}")


if __name__ == "__main__":
    main()
