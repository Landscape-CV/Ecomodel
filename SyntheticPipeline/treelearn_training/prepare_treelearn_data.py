"""
Prepare TreeLearn forests from synthetic instance tiles.

- Train: split.train from testdataset/instance + ALL tiles in instance_expand
- Val: frozen split.val from testdataset/instance only
- Label remap: ground/clutter (-1/-2) -> 0; trees >=0 -> id+1

Usage (from SyntheticPipeline/):
  python treelearn_training/prepare_treelearn_data.py
  python treelearn_training/prepare_treelearn_data.py --run_gen
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_SP = Path(__file__).resolve().parents[1]
_ROOT = _SP.parent
_TL = _ROOT / "TreeLearn"

DEFAULT_SPLIT = _SP / "pointsam_voxelized" / "split.json"
DEFAULT_INSTANCE = _SP / "testdataset" / "instance"
DEFAULT_EXPAND = _SP / "testdataset" / "instance_expand"
DEFAULT_OUT = _SP / "treelearn_prepared"
TL_DATA = _TL / "data" / "ecomodel_ft"


def remap_labels(instances: np.ndarray) -> np.ndarray:
    """Synthetic schema -> TreeLearn schema."""
    out = instances.astype(np.int32).copy()
    tree = out >= 0
    out[~tree] = 0  # ground/clutter -> non-tree
    out[tree] = out[tree] + 1  # 0..K -> 1..K+1
    return out


def tile_to_npy(src_dir: Path, tile: str, dst: Path) -> dict:
    import laspy

    laz = src_dir / f"{tile}_scan.laz"
    ids_path = src_dir / f"{tile}_instances.npy"
    if not laz.is_file() or not ids_path.is_file():
        raise FileNotFoundError(f"Missing scan/instances for {tile} under {src_dir}")
    las = laspy.read(str(laz))
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    ids = np.load(ids_path)
    if len(ids) != len(xyz):
        raise ValueError(f"{tile}: len mismatch points={len(xyz)} labels={len(ids)}")
    labels = remap_labels(ids)
    data = np.hstack([xyz, labels.reshape(-1, 1)]).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, data)
    return {
        "tile": tile,
        "n_points": int(len(data)),
        "n_tree": int((labels > 0).sum()),
        "n_non_tree": int((labels == 0).sum()),
        "n_instances": int(len(np.unique(labels[labels > 0]))),
        "path": str(dst),
    }


def discover_expand_tiles(expand_dir: Path) -> List[str]:
    tiles = []
    if not expand_dir.is_dir():
        return tiles
    for laz in sorted(expand_dir.glob("*_scan.laz")):
        name = laz.name[: -len("_scan.laz")]
        if (expand_dir / f"{name}_instances.npy").is_file():
            tiles.append(name)
    return tiles


def build_split(base_split: dict, expand_tiles: List[str]) -> dict:
    train = list(base_split["train"])
    # Expand tiles are train-only; prefix source tag in meta, not in filename
    for t in expand_tiles:
        if t not in train:
            train.append(t)
        # If expand reuses names like sparse_mixed_00, disambiguate via source map
    return {
        "train": sorted(set(train)),
        "val": list(base_split["val"]),
        "seed": base_split.get("seed", 0),
        "expand_tiles": sorted(expand_tiles),
        "note": "val frozen from pointsam split; expand tiles are train-only",
    }


def stage_to_treelearn(prepared: Path) -> None:
    """Copy/symlink npy forests into TreeLearn/data/ecomodel_ft/{train,val}/forests|forest."""
    train_src = prepared / "forests_train"
    val_src = prepared / "forests_val"
    train_dst = TL_DATA / "train" / "forests"
    val_dst = TL_DATA / "val" / "forest"
    if train_dst.exists():
        shutil.rmtree(train_dst)
    if val_dst.exists():
        shutil.rmtree(val_dst)
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)
    for p in train_src.glob("*.npy"):
        shutil.copy2(p, train_dst / p.name)
    for p in val_src.glob("*.npy"):
        shutil.copy2(p, val_dst / p.name)
    print(f"Staged train forests -> {train_dst} ({len(list(train_dst.glob('*.npy')))})")
    print(f"Staged val forests  -> {val_dst} ({len(list(val_dst.glob('*.npy')))})")


def run_treelearn_gens() -> None:
    """Run gen_train_data + per-tile gen_val_data from TreeLearn root."""
    py = sys.executable
    env = os.environ.copy()
    # Ensure TreeLearn package importable
    env["PYTHONPATH"] = str(_TL) + os.pathsep + env.get("PYTHONPATH", "")

    train_cfg = _TL / "configs" / "data_gen" / "ecomodel_ft_gen_train.yaml"
    print(f"Running gen_train_data: {train_cfg}")
    subprocess.check_call(
        [py, "tools/data_gen/gen_train_data.py", "--config", str(train_cfg.relative_to(_TL))],
        cwd=str(_TL),
        env=env,
    )

    val_forest_dir = TL_DATA / "val" / "forest"
    val_cfg_template = _TL / "configs" / "data_gen" / "ecomodel_ft_gen_val.yaml"
    for forest in sorted(val_forest_dir.glob("*.npy")):
        # Write a one-off config with this forest_path
        tmp = _TL / "configs" / "data_gen" / f"_tmp_val_{forest.stem}.yaml"
        text = val_cfg_template.read_text(encoding="utf-8")
        rel = f"data/ecomodel_ft/val/forest/{forest.name}"
        # replace forest_path line
        lines = []
        for line in text.splitlines():
            if line.startswith("forest_path:"):
                lines.append(f"forest_path: '{rel}'")
            else:
                lines.append(line)
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Running gen_val_data for {forest.name}")
        try:
            subprocess.check_call(
                [
                    py,
                    "tools/data_gen/gen_val_data.py",
                    "--config",
                    str(tmp.relative_to(_TL)),
                ],
                cwd=str(_TL),
                env=env,
            )
        finally:
            if tmp.exists():
                tmp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--instance_dir", type=Path, default=DEFAULT_INSTANCE)
    ap.add_argument("--expand_dir", type=Path, default=DEFAULT_EXPAND)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--run_gen",
        action="store_true",
        help="After writing forests, stage into TreeLearn and run gen_train/val",
    )
    ap.add_argument(
        "--min_expand",
        type=int,
        default=0,
        help="Fail if fewer than this many expand tiles found (0=allow partial)",
    )
    args = ap.parse_args()

    base_split = json.loads(args.split.read_text(encoding="utf-8"))
    expand_tiles = discover_expand_tiles(args.expand_dir)
    print(f"Expand tiles found: {len(expand_tiles)} under {args.expand_dir}")
    if len(expand_tiles) < args.min_expand:
        raise SystemExit(
            f"Need >= {args.min_expand} expand tiles, found {len(expand_tiles)}. "
            "Wait for generate_instance_benchmark to finish."
        )

    split = build_split(base_split, expand_tiles)
    # Disambiguate expand tiles that share names with base instance tiles
    # by writing expand as expand__{name}.npy and tracking sources.
    args.out.mkdir(parents=True, exist_ok=True)
    train_dir = args.out / "forests_train"
    val_dir = args.out / "forests_val"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    sources: Dict[str, str] = {}
    stats = []

    # Base train tiles from instance/
    for tile in base_split["train"]:
        out_name = tile
        st = tile_to_npy(args.instance_dir, tile, train_dir / f"{out_name}.npy")
        st["source"] = "instance"
        stats.append(st)
        sources[out_name] = "instance"

    # Expand train tiles (prefix if name collision)
    for tile in expand_tiles:
        out_name = tile if tile not in sources else f"expand__{tile}"
        st = tile_to_npy(args.expand_dir, tile, train_dir / f"{out_name}.npy")
        st["source"] = "expand"
        st["tile"] = out_name
        stats.append(st)
        sources[out_name] = "expand"

    for tile in base_split["val"]:
        st = tile_to_npy(args.instance_dir, tile, val_dir / f"{tile}.npy")
        st["source"] = "instance"
        stats.append(st)

    split_out = {
        **split,
        "train_files": sorted(p.stem for p in train_dir.glob("*.npy")),
        "val_files": sorted(p.stem for p in val_dir.glob("*.npy")),
        "sources": sources,
        "n_train_forests": len(list(train_dir.glob("*.npy"))),
        "n_val_forests": len(list(val_dir.glob("*.npy"))),
    }
    (args.out / "split.json").write_text(json.dumps(split_out, indent=2), encoding="utf-8")
    (args.out / "meta.json").write_text(
        json.dumps({"tiles": stats, "split": split_out}, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {split_out['n_train_forests']} train + {split_out['n_val_forests']} val "
        f"forests -> {args.out}"
    )

    if args.run_gen:
        stage_to_treelearn(args.out)
        run_treelearn_gens()
        print("TreeLearn crop/tile generation done.")


if __name__ == "__main__":
    main()
