"""
Download and prepare LeWoS LabelledPC into per-tree wood/leaf tiles.

  python scripts/prepare_lewos_woodleaf.py
  python scripts/prepare_lewos_woodleaf.py --skip_download  # if zip already present

Outputs under testdataset/lewos_labelled/:
  raw/LabelledPC.zip
  raw/extracted/...
  tiles/{id}_xyz.npy, {id}_wood.npy, {id}_intensity.npy, {id}_meta.json
  split.json  (tune / holdout)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np

_SP = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _SP / "testdataset" / "lewos_labelled"
_ZENODO_URL = "https://zenodo.org/api/records/4946676/files/LabelledPC.zip/content"


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


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"[skip] already have {dest} ({dest.stat().st_size:,} bytes)")
        return
    print(f"[download] {url}")
    print(f"        -> {dest}")

    def _hook(block, block_size, total):
        if total <= 0:
            return
        done = block * block_size
        pct = min(100.0, 100.0 * done / total)
        if block % 200 == 0 or done >= total:
            print(f"\r  {pct:5.1f}%  {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=_hook)
    print()


def _find_cloud_files(extracted: Path) -> List[Path]:
    exts = {".txt", ".xyz", ".csv", ".pcd", ".ply", ".asc"}
    files = []
    for p in extracted.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts or p.suffix == "":
            # skip tiny junk / readmes
            if p.stat().st_size < 1000:
                continue
            if p.name.lower().startswith("readme"):
                continue
            files.append(p)
    return sorted(files)


def _load_labelled_ascii(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load x y z label (1=wood, 0=leaf)."""
    # Try whitespace-delimited float load; last column = label
    try:
        data = np.loadtxt(str(path), dtype=np.float64)
    except Exception:
        # Some files may have headers
        data = np.loadtxt(str(path), dtype=np.float64, comments="#", skiprows=0)
        if data.ndim == 1:
            raise
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{path}: expected Nx4+ array, got {getattr(data, 'shape', None)}")
    xyz = data[:, :3].astype(np.float64)
    labels = data[:, 3].astype(np.int32)
    wood = labels == 1
    # If labels are inverted somehow (no wood), try 0=wood
    if wood.sum() == 0 and (labels == 0).sum() > 0 and (labels == 1).sum() == 0:
        # all zeros would mean all leaf with 1=wood — fine
        pass
    return xyz, wood


def _tree_id_from_path(path: Path, extracted: Path) -> str:
    rel = path.relative_to(extracted)
    stem = path.stem
    stem = re.sub(r"[^\w\-]+", "_", stem)
    parent = re.sub(r"[^\w\-]+", "_", rel.parent.name) if rel.parent != Path(".") else ""
    if parent and parent not in ("extracted", "LabelledPC", "labelledpc", "raw"):
        return f"{parent}__{stem}"
    return stem


def prepare(
    root: Path,
    *,
    skip_download: bool = False,
    voxel: float = 0.03,
    tune_frac: float = 40 / 61,
) -> Dict:
    raw = root / "raw"
    extracted = raw / "extracted"
    tiles = root / "tiles"
    zip_path = raw / "LabelledPC.zip"

    if not skip_download:
        _download(_ZENODO_URL, zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}; run without --skip_download")

    if not extracted.exists() or not any(extracted.iterdir()):
        print(f"[unzip] {zip_path} -> {extracted}")
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted)
    else:
        print(f"[skip] extract exists: {extracted}")

    files = _find_cloud_files(extracted)
    if not files:
        raise RuntimeError(f"No labelled cloud files found under {extracted}")
    print(f"[found] {len(files)} candidate files")

    tiles.mkdir(parents=True, exist_ok=True)
    prepared: List[Dict] = []
    for path in files:
        try:
            xyz, wood = _load_labelled_ascii(path)
        except Exception as exc:
            print(f"[skip] {path.name}: {exc}")
            continue
        if len(xyz) < 100:
            print(f"[skip] {path.name}: too few points ({len(xyz)})")
            continue
        inten = np.ones(len(xyz), dtype=np.float64)  # placeholder — LeWoS has no intensity
        n0 = len(xyz)
        xyz, wood, inten = _voxel_downsample(xyz, wood, inten, voxel)
        tid = _tree_id_from_path(path, extracted)
        # Avoid collisions
        out_id = tid
        i = 2
        while (tiles / f"{out_id}_xyz.npy").exists() and out_id not in [p["id"] for p in prepared]:
            # if file from previous run, overwrite same id
            break
        while any(p["id"] == out_id for p in prepared):
            out_id = f"{tid}_{i}"
            i += 1

        np.save(tiles / f"{out_id}_xyz.npy", xyz.astype(np.float64))
        np.save(tiles / f"{out_id}_wood.npy", wood.astype(bool))
        np.save(tiles / f"{out_id}_intensity.npy", inten.astype(np.float64))
        meta = {
            "id": out_id,
            "source_file": str(path.relative_to(extracted)).replace("\\", "/"),
            "n_raw": int(n0),
            "n": int(len(xyz)),
            "n_wood": int(wood.sum()),
            "n_leaf": int((~wood).sum()),
            "voxel": float(voxel),
            "intensity": "placeholder_ones",
            "label_convention": "1=wood,0=leaf in source; wood.npy bool",
            "dataset": "lewos_labelledpc",
        }
        (tiles / f"{out_id}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        prepared.append(meta)
        print(
            f"  {out_id}: n={meta['n']:,} wood={meta['n_wood']:,} "
            f"leaf={meta['n_leaf']:,} (from {n0:,})"
        )

    if not prepared:
        raise RuntimeError("No trees prepared")

    ids = sorted(p["id"] for p in prepared)
    n_tune = max(1, int(round(len(ids) * tune_frac)))
    n_tune = min(n_tune, len(ids) - 1) if len(ids) > 1 else len(ids)
    split = {
        "dataset": "lewos_labelledpc",
        "tune": ids[:n_tune],
        "holdout": ids[n_tune:],
        "n_trees": len(ids),
        "voxel": float(voxel),
        "note": "Param selection on tune; report final metrics on holdout.",
    }
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    index_path = root / "index.json"
    index_path.write_text(json.dumps({"trees": prepared, "split": split}, indent=2), encoding="utf-8")
    print(f"[done] {len(prepared)} trees -> {tiles}")
    print(f"       tune={len(split['tune'])} holdout={len(split['holdout'])}")
    return {"prepared": prepared, "split": split}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--tune_frac", type=float, default=40 / 61)
    args = ap.parse_args()
    prepare(args.root, skip_download=args.skip_download, voxel=args.voxel, tune_frac=args.tune_frac)


if __name__ == "__main__":
    main()
