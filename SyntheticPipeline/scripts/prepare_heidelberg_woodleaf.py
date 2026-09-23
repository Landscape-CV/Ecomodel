"""
Prepare Heidelberg TLS leaf-wood LAZ tiles (Classification 0=wood, 1=leaf).

Expects LAZ files already in testdataset/heidelberg_woodleaf/raw/.

  python scripts/prepare_heidelberg_woodleaf.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import laspy
except ImportError:
    print("pip install laspy lazrs")
    sys.exit(1)

_SP = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _SP / "testdataset" / "heidelberg_woodleaf"


def _voxel_downsample(
    xyz: np.ndarray,
    wood: np.ndarray,
    intensity: np.ndarray,
    voxel: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if voxel <= 0 or len(xyz) == 0:
        return xyz, wood, intensity
    keys = np.floor(xyz / float(voxel)).astype(np.int64)
    keys = keys - keys.min(axis=0, keepdims=True)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return xyz[idx], wood[idx], intensity[idx]


def _load_laz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = laspy.read(str(path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    cls = np.asarray(las.classification, dtype=np.int32)
    # 0 = wood, 1 = leaf
    wood = cls == 0
    if hasattr(las, "reflectance"):
        inten = np.asarray(las.reflectance, dtype=np.float64)
    elif hasattr(las, "amplitude"):
        inten = np.asarray(las.amplitude, dtype=np.float64)
    elif hasattr(las, "intensity"):
        inten = np.asarray(las.intensity, dtype=np.float64)
    else:
        inten = np.ones(len(xyz), dtype=np.float64)
    # sanitize
    if not np.isfinite(inten).all():
        inten = np.nan_to_num(inten, nan=0.0)
    return xyz, wood, inten


def prepare(root: Path, *, voxel: float = 0.05, tune_frac: float = 7 / 11) -> Dict:
    raw = root / "raw"
    tiles = root / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    laz_files = sorted({p.resolve() for p in list(raw.glob("*.laz")) + list(raw.glob("*.LAZ"))})
    laz_files = sorted(laz_files, key=lambda p: p.name.lower())
    if not laz_files:
        raise FileNotFoundError(f"No LAZ in {raw}")

    # Clear previous tile outputs to avoid stale duplicates
    if tiles.exists():
        for old in tiles.glob("*"):
            old.unlink()
    tiles.mkdir(parents=True, exist_ok=True)

    prepared: List[Dict] = []
    for path in laz_files:
        xyz, wood, inten = _load_laz(path)
        n0 = len(xyz)
        xyz, wood, inten = _voxel_downsample(xyz, wood, inten, voxel)
        tid = path.stem
        np.save(tiles / f"{tid}_xyz.npy", xyz)
        np.save(tiles / f"{tid}_wood.npy", wood.astype(bool))
        np.save(tiles / f"{tid}_intensity.npy", inten)
        meta = {
            "id": tid,
            "source_file": path.name,
            "n_raw": int(n0),
            "n": int(len(xyz)),
            "n_wood": int(wood.sum()),
            "n_leaf": int((~wood).sum()),
            "voxel": float(voxel),
            "intensity": "reflectance_or_amplitude",
            "label_convention": "Classification 0=wood,1=leaf",
            "dataset": "heidelberg_uumedi",
        }
        (tiles / f"{tid}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        prepared.append(meta)
        print(
            f"  {tid}: n={meta['n']:,} wood={meta['n_wood']:,} leaf={meta['n_leaf']:,}"
        )

    ids = sorted(p["id"] for p in prepared)
    n_tune = max(1, int(round(len(ids) * tune_frac)))
    if len(ids) > 1:
        n_tune = min(n_tune, len(ids) - 1)
    split = {
        "dataset": "heidelberg_uumedi",
        "tune": ids[:n_tune],
        "holdout": ids[n_tune:],
        "n_trees": len(ids),
        "voxel": float(voxel),
    }
    (root / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (root / "index.json").write_text(
        json.dumps({"trees": prepared, "split": split}, indent=2), encoding="utf-8"
    )
    print(f"[done] {len(prepared)} trees; tune={len(split['tune'])} holdout={len(split['holdout'])}")
    return {"prepared": prepared, "split": split}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--voxel", type=float, default=0.05)
    args = ap.parse_args()
    prepare(args.root, voxel=args.voxel)


if __name__ == "__main__":
    main()
