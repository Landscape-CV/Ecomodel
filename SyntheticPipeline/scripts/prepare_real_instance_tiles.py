"""
Adapt public ground-based forest instance datasets into Ecomodel tile layout:

  {tile}_scan.laz
  {tile}_instances.npy   # int32, trees >=0, non-tree = -1
  {tile}_meta.json
  split.json

Supported inputs:
  - TreeLearn LAZ with extra dim ``treeID`` (QUTUWU / L1W)
  - ISPRS TLS benchmark PLY with ``semantic`` + ``instance`` scalars

Example:
  python scripts/prepare_real_instance_tiles.py \\
      --raw_dir testdataset/real_raw \\
      --out_dir testdataset/real_instance \\
      --tile_size 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import laspy
except ImportError:
    print("Please pip install laspy lazrs")
    sys.exit(1)

_SP_DIR = Path(__file__).resolve().parents[1]


def _remap_tree_ids(raw_ids: np.ndarray) -> Tuple[np.ndarray, int]:
    """Map positive tree IDs to contiguous 0..T-1; non-positive → -1."""
    out = np.full(raw_ids.shape, -1, dtype=np.int32)
    pos = raw_ids > 0
    if not np.any(pos):
        return out, 0
    uniq = np.unique(raw_ids[pos])
    mapping = {int(u): i for i, u in enumerate(uniq)}
    out[pos] = np.vectorize(mapping.get, otypes=[np.int32])(raw_ids[pos])
    return out, int(len(uniq))


def _voxel_downsample(
    xyz: np.ndarray,
    intensity: np.ndarray,
    instances: np.ndarray,
    voxel: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep one point per voxel (first occurrence) to match TreeLearn ~0.1 m density."""
    if voxel <= 0 or len(xyz) == 0:
        return xyz, intensity, instances
    keys = np.floor(xyz / float(voxel)).astype(np.int64)
    keys = keys - keys.min(axis=0, keepdims=True)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return xyz[idx], intensity[idx], instances[idx]


def _write_tile(
    out_dir: Path,
    name: str,
    xyz: np.ndarray,
    intensity_u16: np.ndarray,
    instances: np.ndarray,
    meta: Dict,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / name
    n = len(xyz)
    assert len(intensity_u16) == n and len(instances) == n

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = intensity_u16
    las.write(str(prefix) + "_scan.laz")

    np.save(str(prefix) + "_instances.npy", instances.astype(np.int32))
    with open(str(prefix) + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return name


def _tile_xy(
    xyz: np.ndarray,
    intensity: np.ndarray,
    instances: np.ndarray,
    tile_size: float,
    min_points: int,
    min_trees: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """Non-overlapping XY tiles of side tile_size (meters)."""
    xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
    xmax, ymax = xyz[:, 0].max(), xyz[:, 1].max()
    tiles = []
    ix = 0
    x0 = xmin
    while x0 < xmax - 1e-6:
        iy = 0
        y0 = ymin
        while y0 < ymax - 1e-6:
            x1, y1 = x0 + tile_size, y0 + tile_size
            m = (
                (xyz[:, 0] >= x0)
                & (xyz[:, 0] < x1)
                & (xyz[:, 1] >= y0)
                & (xyz[:, 1] < y1)
            )
            if int(m.sum()) >= min_points:
                sub_xyz = xyz[m]
                sub_inst = instances[m]
                n_trees = int(len(np.unique(sub_inst[sub_inst >= 0])))
                if n_trees >= min_trees:
                    tiles.append(
                        (
                            sub_xyz,
                            intensity[m],
                            sub_inst,
                            {
                                "tile_ix": ix,
                                "tile_iy": iy,
                                "bbox_xy": [float(x0), float(y0), float(x1), float(y1)],
                                "num_points": int(m.sum()),
                                "num_trees": n_trees,
                            },
                        )
                    )
            iy += 1
            y0 = y1
        ix += 1
        x0 = x0 + tile_size
    return tiles


def load_treelearn_laz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    las = laspy.read(str(path))
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    if hasattr(las, "intensity") and las.intensity is not None:
        inten = np.asarray(las.intensity, dtype=np.float64)
        if float(np.nanmax(inten)) <= 0:
            inten_u16 = np.full(len(xyz), 30000, dtype=np.uint16)
        else:
            if inten.max() <= 1.5:
                inten = inten * 65535.0
            inten_u16 = np.clip(np.round(inten), 0, 65535).astype(np.uint16)
    else:
        inten_u16 = np.full(len(xyz), 30000, dtype=np.uint16)

    dims = set(las.point_format.dimension_names) | set(getattr(las, "point_format").extra_dimension_names)
    # laspy extra dims
    extra = list(las.point_format.extra_dimension_names)
    tree_ids = None
    for cand in ("treeID", "tree_id", "TreeID", "instance"):
        if cand in extra or hasattr(las, cand):
            try:
                tree_ids = np.asarray(getattr(las, cand), dtype=np.int32)
                break
            except Exception:
                pass
    if tree_ids is None:
        raise ValueError(f"No treeID field in {path}; extras={extra}")

    # Unlabeled (classification==3) → treat as non-tree for Ecomodel eval
    unlabeled_n = 0
    if hasattr(las, "classification"):
        cls = np.asarray(las.classification)
        unlabeled = cls == 3
        unlabeled_n = int(unlabeled.sum())
        tree_ids = tree_ids.copy()
        tree_ids[unlabeled] = 0

    instances, n_trees = _remap_tree_ids(tree_ids)
    info = {
        "source_file": str(path),
        "source_format": "treelearn_laz",
        "num_trees_full": n_trees,
        "num_unlabeled_points": unlabeled_n,
        "num_points_full": int(len(xyz)),
    }
    return xyz, inten_u16, instances, info


def load_isprs_ply(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """ISPRS TLS benchmark PLY: semantic (0 ground, 1 tree), instance (1..N, -1 ground)."""
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("plyfile required to read ISPRS PLY tiles")

    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.vstack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])]).T.astype(
        np.float64
    )
    names = set(v.data.dtype.names)
    if "instance" not in names:
        raise ValueError(f"No 'instance' property in {path}; props={names}")
    raw = np.asarray(v["instance"], dtype=np.int32)
    # Ensure ground / non-tree stay non-positive
    if "semantic" in names:
        sem = np.asarray(v["semantic"])
        raw = raw.copy()
        raw[sem == 0] = -1
    instances, n_trees = _remap_tree_ids(raw)
    inten_u16 = np.full(len(xyz), 30000, dtype=np.uint16)
    if "intensity" in names or "Intensity" in names:
        key = "intensity" if "intensity" in names else "Intensity"
        inten = np.asarray(v[key], dtype=np.float64)
        if inten.max() <= 1.5:
            inten = inten * 65535.0
        inten_u16 = np.clip(np.round(inten), 0, 65535).astype(np.uint16)
    info = {
        "source_file": str(path),
        "source_format": "isprs_ply",
        "num_trees_full": n_trees,
        "num_points_full": int(len(xyz)),
    }
    return xyz, inten_u16, instances, info


def process_plot(
    xyz: np.ndarray,
    intensity: np.ndarray,
    instances: np.ndarray,
    info: Dict,
    out_dir: Path,
    plot_tag: str,
    composition: str,
    tile_size: float,
    min_points: int,
    min_trees: int,
    write_full: bool,
    voxel: float = 0.0,
    max_points_per_tile: int = 2_500_000,
) -> List[str]:
    names: List[str] = []
    if write_full and len(xyz) <= 8_000_000:
        name = f"{plot_tag}_full"
        f_xyz, f_inten, f_inst = xyz, intensity, instances
        if voxel > 0:
            f_xyz, f_inten, f_inst = _voxel_downsample(f_xyz, f_inten, f_inst, voxel)
        meta = {
            "tile_name": name,
            "density_class": plot_tag,
            "composition": composition,
            "num_trees": int(len(np.unique(f_inst[f_inst >= 0]))),
            "label_schema": {"ground": -1, "trees": ">=0"},
            **info,
            "tile_mode": "full",
            "voxel_m": voxel if voxel > 0 else None,
            "num_points": int(len(f_xyz)),
        }
        names.append(_write_tile(out_dir, name, f_xyz, f_inten, f_inst, meta))

    for t_xyz, t_inten, t_inst, tmeta in _tile_xy(
        xyz, intensity, instances, tile_size, min_points, min_trees
    ):
        used_voxel = 0.0
        # Auto voxel dense native TLS (ISPRS) down to TreeLearn-like density
        if voxel > 0:
            t_xyz, t_inten, t_inst = _voxel_downsample(t_xyz, t_inten, t_inst, voxel)
            used_voxel = voxel
        elif len(t_xyz) > max_points_per_tile:
            used_voxel = 0.1
            t_xyz, t_inten, t_inst = _voxel_downsample(
                t_xyz, t_inten, t_inst, used_voxel
            )
        name = f"{plot_tag}_t{tmeta['tile_ix']:02d}_{tmeta['tile_iy']:02d}"
        n_trees = int(len(np.unique(t_inst[t_inst >= 0])))
        if n_trees < min_trees or len(t_xyz) < min_points:
            continue
        meta = {
            "tile_name": name,
            "density_class": plot_tag,
            "composition": composition,
            "num_trees": n_trees,
            "label_schema": {"ground": -1, "trees": ">=0"},
            **info,
            "tile_mode": "grid",
            **tmeta,
            "num_points": int(len(t_xyz)),
            "voxel_m": used_voxel if used_voxel > 0 else None,
        }
        names.append(_write_tile(out_dir, name, t_xyz, t_inten, t_inst, meta))
    return names


def discover_inputs(raw_dir: Path) -> List[Tuple[str, Path, str, str]]:
    """
    Returns list of (loader, path, plot_tag, composition).
    loader in {'laz','ply'}
    """
    found: List[Tuple[str, Path, str, str]] = []
    # TreeLearn L1W — prefer the evaluation-labeled cloud when both exist
    l1w_files = sorted(raw_dir.rglob("L1W*.laz"))
    l1w_eval = [p for p in l1w_files if "for_eval" in p.name.lower()]
    l1w_use = l1w_eval if l1w_eval else l1w_files
    for p in l1w_use:
        found.append(("laz", p, "l1w", "mls_holdout"))
    # QUTUWU Wytham voxel
    for p in raw_dir.rglob("*wytham*vox*.laz"):
        found.append(("laz", p, "wytham_qutuwu", "deciduous_tls"))
    for p in raw_dir.rglob("wytham_vox0.1.laz"):
        if ("laz", p, "wytham_qutuwu", "deciduous_tls") not in found:
            found.append(("laz", p, "wytham_qutuwu", "deciduous_tls"))
    # ISPRS Wytham PLY — plot-level only (skip trees/** individual clouds)
    isprs_dir = raw_dir / "isprs_wytham"
    if isprs_dir.exists():
        plot_plys = [
            p
            for p in isprs_dir.glob("*.ply")
            if p.is_file() and "tree" not in p.stem.lower()
        ]
        if not plot_plys:
            # Fallback: any PLY not under a trees/ folder
            plot_plys = [
                p
                for p in isprs_dir.rglob("*.ply")
                if "trees" not in {part.lower() for part in p.parts}
            ]
        # Prefer test split for the sanity-check plot; fall back to val then all plots
        preferred = [p for p in plot_plys if "test" in p.stem.lower()]
        if not preferred:
            preferred = [p for p in plot_plys if "val" in p.stem.lower()]
        use_plys = preferred if preferred else plot_plys
        for p in use_plys:
            split = (
                "test"
                if "test" in p.stem.lower()
                else (
                    "val"
                    if "val" in p.stem.lower()
                    else ("train" if "train" in p.stem.lower() else "plot")
                )
            )
            found.append(("ply", p, f"wytham_isprs_{split}", "deciduous_tls"))
    # de-dupe by path
    seen = set()
    uniq = []
    for item in found:
        key = str(item[1].resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


def main():
    parser = argparse.ArgumentParser(description="Prepare real TLS/MLS instance tiles")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(_SP_DIR / "testdataset" / "real_raw"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_SP_DIR / "testdataset" / "real_instance"),
    )
    parser.add_argument("--tile_size", type=float, default=30.0)
    parser.add_argument("--min_points", type=int, default=20_000)
    parser.add_argument("--min_trees", type=int, default=2)
    parser.add_argument(
        "--write_full",
        action="store_true",
        help="Also write a full-plot tile when point count is manageable",
    )
    parser.add_argument(
        "--max_val_tiles",
        type=int,
        default=24,
        help="Cap number of tiles listed in split.json val",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="Optional uniform voxel size (m). ISPRS native TLS auto-voxels at 0.1 m when tiles exceed max points.",
    )
    parser.add_argument(
        "--max_points_per_tile",
        type=int,
        default=2_500_000,
        help="If a tile exceeds this and --voxel is 0, auto voxelize at 0.1 m",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(raw_dir)
    if not inputs:
        print(f"No known inputs under {raw_dir}")
        print("Expected: L1W*.laz, wytham*vox*.laz, and/or isprs_wytham/**/*.ply")
        sys.exit(1)

    all_names: List[str] = []
    for loader, path, tag, composition in inputs:
        print(f"\n=== {tag}: {path} ===")
        if loader == "laz":
            xyz, inten, inst, info = load_treelearn_laz(path)
        else:
            xyz, inten, inst, info = load_isprs_ply(path)
        print(
            f"  points={len(xyz):,} trees={info.get('num_trees_full')} "
            f"extent=[{np.ptp(xyz[:,0]):.1f} x {np.ptp(xyz[:,1]):.1f} m]"
        )
        # Native ISPRS PLY is much denser than TreeLearn 0.1 m voxels — force voxel.
        plot_voxel = args.voxel
        if loader == "ply" and plot_voxel <= 0:
            plot_voxel = 0.1
        names = process_plot(
            xyz,
            inten,
            inst,
            info,
            out_dir,
            plot_tag=tag,
            composition=composition,
            tile_size=args.tile_size,
            min_points=args.min_points,
            min_trees=args.min_trees,
            write_full=args.write_full,
            voxel=plot_voxel,
            max_points_per_tile=args.max_points_per_tile,
        )
        print(f"  wrote {len(names)} tiles")
        all_names.extend(names)

    # Prefer ISPRS test + L1W + QUTUWU tiles for val; shuffle within tag
    rng = np.random.default_rng(0)
    by_tag: Dict[str, List[str]] = {}
    for n in all_names:
        tag = n.split("_t")[0] if "_t" in n else n.replace("_full", "")
        # normalize
        if n.startswith("l1w"):
            key = "l1w"
        elif "isprs_test" in n:
            key = "isprs_test"
        elif "isprs" in n:
            key = "isprs_other"
        elif "qutuwu" in n:
            key = "qutuwu"
        else:
            key = "other"
        by_tag.setdefault(key, []).append(n)

    val: List[str] = []
    # Priority order
    for key in ("isprs_test", "l1w", "qutuwu", "isprs_other", "other"):
        pool = by_tag.get(key, [])
        if not pool:
            continue
        order = list(rng.permutation(pool))
        # take up to remaining slots, keep some diversity
        need = max(0, args.max_val_tiles - len(val))
        take = min(need, max(4, need // max(1, 3)) if key != "isprs_test" else need)
        if key == "isprs_test":
            take = min(need, len(order))
        val.extend(order[:take])
        if len(val) >= args.max_val_tiles:
            break
    val = val[: args.max_val_tiles]

    split = {"train": [], "val": val, "all": all_names, "seed": 0}
    split_path = out_dir / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_names)} tiles → {out_dir}")
    print(f"split.json val={len(val)} → {split_path}")


if __name__ == "__main__":
    main()
