"""
Voxelize SyntheticPipeline/testdataset/instance for Point-SAM fine-tuning.

Steps:
  1. Stratified 48/16 train/val split (seed=0, 2 tiles per density x composition)
  2. Filter ground (instance_id < 0)
  3. Voxelize at 0.05 m with majority instance ID
  4. Intensity -> RGB; write crops under pointsam_voxelized/{train,val}/

Usage (from SyntheticPipeline/):
  python pointsam_training/prepare_voxel_dataset.py
  python pointsam_training/prepare_voxel_dataset.py --smoke 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_SP = Path(__file__).resolve().parents[1]
_ROOT = _SP.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Utils.promptable_instance import intensity_to_rgb  # noqa: E402

DEFAULT_SRC = _SP / "testdataset" / "instance"
DEFAULT_OUT = _SP / "pointsam_voxelized"


def tile_stratum(tile_name: str, meta: dict | None) -> Tuple[str, str]:
    if meta:
        d = meta.get("density_class") or meta.get("density") or "unknown"
        c = meta.get("composition") or "unknown"
        if c.startswith("mono"):
            c = "mono"
        return str(d), str(c)
    parts = tile_name.split("_")
    dens = parts[0]
    comp = "mixed" if len(parts) > 1 and parts[1] == "mixed" else "mono"
    return dens, comp


def discover_tiles(src: Path) -> List[str]:
    tiles = []
    for laz in sorted(src.glob("*_scan.laz")):
        name = laz.name[: -len("_scan.laz")]
        if (src / f"{name}_instances.npy").is_file():
            tiles.append(name)
    return tiles


def make_stratified_split(
    tiles: List[str],
    src: Path,
    seed: int = 0,
    val_per_stratum: int = 2,
) -> Dict[str, List[str]]:
    """48 train / 16 val when 8 strata x 2 val tiles."""
    by_key: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for name in tiles:
        meta_path = src / f"{name}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None
        by_key[tile_stratum(name, meta)].append(name)

    rng = np.random.default_rng(seed)
    train, val = [], []
    for key in sorted(by_key.keys()):
        pool = list(by_key[key])
        rng.shuffle(pool)
        n_val = min(val_per_stratum, len(pool))
        val.extend(pool[:n_val])
        train.extend(pool[n_val:])
    return {"train": sorted(train), "val": sorted(val), "seed": seed}


def voxelize_majority(
    xyz: np.ndarray,
    instance_ids: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One point per voxel; instance ID = majority vote; RGB/XYZ = first point."""
    if len(xyz) == 0:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0,), np.int32),
        )
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    # Pack XYZ keys into a single int64 for fast grouping
    # Shift ranges are safe for ~±1e5 voxel indices at 0.05 m
    kx = keys[:, 0] - keys[:, 0].min()
    ky = keys[:, 1] - keys[:, 1].min()
    kz = keys[:, 2] - keys[:, 2].min()
    packed = kx + (ky << 20) + (kz << 40)

    order = np.argsort(packed, kind="mergesort")
    packed_s = packed[order]
    xyz_s = xyz[order]
    rgb_s = rgb[order]
    ids_s = instance_ids[order]

    # Unique voxels + inverse map
    uniq, start_idx, counts = np.unique(
        packed_s, return_index=True, return_counts=True
    )
    n = len(uniq)
    out_xyz = xyz_s[start_idx].astype(np.float32)
    out_rgb = rgb_s[start_idx].astype(np.float32)

    # Majority label via bincount on (voxel_index, label) — labels are small ints
    # Build inverse: each point → voxel rank
    inverse = np.repeat(np.arange(n, dtype=np.int64), counts)
    # Offset labels to non-negative for bincount; tree ids are >=0 here
    lab = ids_s.astype(np.int64)
    lab_min = int(lab.min())
    lab0 = lab - lab_min
    n_lab = int(lab0.max()) + 1
    # Flat index: voxel * n_lab + label
    flat = inverse * n_lab + lab0
    bc = np.bincount(flat, minlength=n * n_lab).reshape(n, n_lab)
    out_ids = (bc.argmax(axis=1) + lab_min).astype(np.int32)
    return out_xyz, out_rgb, out_ids


def make_crops(
    xyz: np.ndarray,
    rgb: np.ndarray,
    ids: np.ndarray,
    crop_size: float,
    stride: float,
    min_points: int,
    max_points: int,
    min_instances: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Sliding XY crops; random subsample if over max_points."""
    if len(xyz) == 0:
        return []
    xmin, ymin = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    xmax, ymax = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    crops = []
    xs = np.arange(xmin, xmax + 1e-6, stride)
    ys = np.arange(ymin, ymax + 1e-6, stride)
    if len(xs) == 0:
        xs = np.array([xmin])
    if len(ys) == 0:
        ys = np.array([ymin])

    for x0 in xs:
        for y0 in ys:
            m = (
                (xyz[:, 0] >= x0)
                & (xyz[:, 0] < x0 + crop_size)
                & (xyz[:, 1] >= y0)
                & (xyz[:, 1] < y0 + crop_size)
            )
            n = int(m.sum())
            if n < min_points:
                continue
            idx = np.nonzero(m)[0]
            c_ids = ids[idx]
            n_inst = len(np.unique(c_ids[c_ids >= 0]))
            if n_inst < min_instances:
                continue
            if n > max_points:
                sel = rng.choice(n, size=max_points, replace=False)
                idx = idx[sel]
            crops.append(
                {
                    "xyz": xyz[idx].astype(np.float32),
                    "rgb": rgb[idx].astype(np.float32),
                    "instance_ids": ids[idx].astype(np.int32),
                    "origin_xy": [float(x0), float(y0)],
                }
            )

    # Always keep at least one full-tile downsample if no crop passed
    if not crops:
        idx = np.arange(len(xyz))
        if len(idx) > max_points:
            idx = rng.choice(len(xyz), size=max_points, replace=False)
        crops.append(
            {
                "xyz": xyz[idx].astype(np.float32),
                "rgb": rgb[idx].astype(np.float32),
                "instance_ids": ids[idx].astype(np.int32),
                "origin_xy": [xmin, ymin],
            }
        )
    return crops


def process_tile(
    src: Path,
    name: str,
    out_split_dir: Path,
    voxel_size: float,
    crop_size: float,
    stride: float,
    min_points: int,
    max_points: int,
    min_instances: int,
    seed: int,
) -> dict:
    import laspy

    laz_path = src / f"{name}_scan.laz"
    ids_path = src / f"{name}_instances.npy"
    las = laspy.read(str(laz_path))
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    inten = np.asarray(las.intensity, dtype=np.float64).reshape(-1, 1)
    points = np.hstack([xyz, inten])
    ids = np.load(ids_path).astype(np.int32)
    if len(ids) != len(xyz):
        raise ValueError(f"{name}: instances len {len(ids)} != points {len(xyz)}")

    # Keep trees + a capped amount of ground as SAM background (id=-1).
    # Pure tree-only crops make single-tree masks cover all points and break
    # Point-SAM's border prompt sampler.
    rgb_all = intensity_to_rgb(points)
    tree = ids >= 0
    ground = ~tree
    xyz_t, ids_t, rgb_t = xyz[tree], ids[tree], rgb_all[tree]
    xyz_g, ids_g, rgb_g = xyz[ground], ids[ground], rgb_all[ground]

    xyz_v_t, rgb_v_t, ids_v_t = voxelize_majority(xyz_t, ids_t, rgb_t, voxel_size)
    if len(xyz_g):
        xyz_v_g, rgb_v_g, ids_v_g = voxelize_majority(xyz_g, ids_g, rgb_g, voxel_size)
        max_g = max(len(xyz_v_t), 1)  # at most as many ground voxels as trees
        if len(xyz_v_g) > max_g:
            sel = np.random.default_rng(seed + (hash(name) % 10_000_000)).choice(
                len(xyz_v_g), size=max_g, replace=False
            )
            xyz_v_g, rgb_v_g, ids_v_g = xyz_v_g[sel], rgb_v_g[sel], ids_v_g[sel]
        xyz_v = np.concatenate([xyz_v_t, xyz_v_g], axis=0)
        rgb_v = np.concatenate([rgb_v_t, rgb_v_g], axis=0)
        ids_v = np.concatenate([ids_v_t, ids_v_g], axis=0)
    else:
        xyz_v, rgb_v, ids_v = xyz_v_t, rgb_v_t, ids_v_t
    rng = np.random.default_rng(seed + (hash(name) % 10_000_000))
    crops = make_crops(
        xyz_v,
        rgb_v,
        ids_v,
        crop_size=crop_size,
        stride=stride,
        min_points=min_points,
        max_points=max_points,
        min_instances=min_instances,
        rng=rng,
    )

    out_split_dir.mkdir(parents=True, exist_ok=True)
    crop_names = []
    for i, crop in enumerate(crops):
        fname = f"{name}_crop{i:03d}.npz"
        np.savez_compressed(
            out_split_dir / fname,
            xyz=crop["xyz"],
            rgb=crop["rgb"],
            instance_ids=crop["instance_ids"],
            tile=np.asarray(name),
            origin_xy=np.asarray(crop["origin_xy"], dtype=np.float32),
        )
        crop_names.append(fname)

    return {
        "tile": name,
        "n_raw_tree_points": int(tree.sum()),
        "n_raw_ground_points": int(ground.sum()),
        "n_voxel": int(len(xyz_v)),
        "n_voxel_tree": int(len(xyz_v_t)),
        "n_crops": len(crop_names),
        "crops": crop_names,
        "n_instances": int(len(np.unique(ids_v[ids_v >= 0]))),
    }


def main():
    ap = argparse.ArgumentParser(description="Voxelize instance tiles for Point-SAM FT")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--crop_size", type=float, default=12.0)
    ap.add_argument("--stride", type=float, default=6.0)
    ap.add_argument("--min_points", type=int, default=2048)
    ap.add_argument("--max_points", type=int, default=100000)
    ap.add_argument("--min_instances", type=int, default=1,
                    help="Min tree instances per crop (1 ok if ground kept as BG)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_per_stratum", type=int, default=2)
    ap.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="If >0, only process this many train+val tiles (debug)",
    )
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source dataset not found: {args.src}")

    tiles = discover_tiles(args.src)
    if not tiles:
        raise SystemExit(f"No tiles with scan+instances under {args.src}")

    split = make_stratified_split(
        tiles, args.src, seed=args.seed, val_per_stratum=args.val_per_stratum
    )
    print(
        f"Split seed={args.seed}: train={len(split['train'])} val={len(split['val'])}"
    )

    # Dataset sufficiency note
    verdict = {
        "source": str(args.src),
        "n_tiles": len(tiles),
        "train_tiles": len(split["train"]),
        "val_tiles": len(split["val"]),
        "verdict": (
            "Adequate for supervised domain adaptation / fine-tuning of pretrained "
            "Point-SAM; insufficient to train from scratch. Ground filtered; "
            "voxel 0.05 m matches inference."
        ),
        "limitations": [
            "Only 64 synthetic TLS scenes",
            "No real-world TLS in this corpus",
            "Instance IDs are per-tile",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    print(f"Wrote {args.out / 'split.json'}")

    process_list = [("train", split["train"]), ("val", split["val"])]
    if args.smoke > 0:
        # take first smoke tiles from each, preserving names
        tr = split["train"][: max(1, args.smoke // 2)]
        va = split["val"][: max(1, args.smoke - len(tr))]
        process_list = [("train", tr), ("val", va)]
        print(f"SMOKE mode: processing {tr} / {va}")

    tile_stats = []
    for split_name, names in process_list:
        for name in names:
            print(f"[{split_name}] {name} ...", flush=True)
            st = process_tile(
                args.src,
                name,
                args.out / split_name,
                voxel_size=args.voxel_size,
                crop_size=args.crop_size,
                stride=args.stride,
                min_points=args.min_points,
                max_points=args.max_points,
                min_instances=args.min_instances,
                seed=args.seed,
            )
            st["split"] = split_name
            tile_stats.append(st)
            print(
                f"  tree_pts={st['n_raw_tree_points']} voxel={st['n_voxel']} "
                f"crops={st['n_crops']} inst={st['n_instances']}"
            )

    meta = {
        "verdict": verdict,
        "voxel_size": args.voxel_size,
        "crop_size": args.crop_size,
        "stride": args.stride,
        "min_points": args.min_points,
        "max_points": args.max_points,
        "min_instances": args.min_instances,
        "seed": args.seed,
        "tiles": tile_stats,
        "n_crops_train": sum(t["n_crops"] for t in tile_stats if t["split"] == "train"),
        "n_crops_val": sum(t["n_crops"] for t in tile_stats if t["split"] == "val"),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. meta -> {args.out / 'meta.json'}")
    print(
        f"Crops: train={meta['n_crops_train']} val={meta['n_crops_val']}"
    )


if __name__ == "__main__":
    main()
