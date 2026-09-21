"""Load/save TLS clouds and Ecomodel instance tile triplets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import laspy
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install laspy lazrs") from exc

PathLike = Union[str, Path]
_SP_DIR = Path(__file__).resolve().parents[1]


def _as_path(p: PathLike) -> Path:
    """Resolve path; try SyntheticPipeline-relative for Streamlit cwd safety."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()

    def _looks_real(cand: Path) -> bool:
        return (
            cand.exists()
            or Path(str(cand) + "_scan.laz").exists()
            or Path(str(cand) + "_scan.las").exists()
        )

    sp_cand = (_SP_DIR / path).resolve()
    if _looks_real(sp_cand):
        return sp_cand
    cwd_cand = path.resolve()
    if _looks_real(cwd_cand):
        return cwd_cand
    return sp_cand


def load_laz_xyzi(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Return xyz (N,3) float64 and intensity in [0,1] float64."""
    las = laspy.read(str(_as_path(path)))
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    if hasattr(las, "intensity") and las.intensity is not None:
        inten = np.asarray(las.intensity, dtype=np.float64)
        mx = float(np.nanmax(inten)) if len(inten) else 0.0
        if mx <= 0:
            inten01 = np.full(len(xyz), 0.5, dtype=np.float64)
        elif mx <= 1.5:
            inten01 = np.clip(inten, 0.0, 1.0)
        else:
            inten01 = np.clip(inten / 65535.0, 0.0, 1.0)
    else:
        inten01 = np.full(len(xyz), 0.5, dtype=np.float64)
    return xyz, inten01


def load_ply_xyzi(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Load PLY vertices; intensity mid-scale if missing."""
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install plyfile") from exc

    ply = PlyData.read(str(_as_path(path)))
    v = ply["vertex"]
    xyz = np.vstack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])]).T.astype(
        np.float64
    )
    names = set(v.data.dtype.names)
    inten01 = np.full(len(xyz), 0.5, dtype=np.float64)
    for key in ("intensity", "Intensity"):
        if key in names:
            inten = np.asarray(v[key], dtype=np.float64)
            mx = float(inten.max()) if len(inten) else 0.0
            if mx <= 1.5:
                inten01 = np.clip(inten, 0.0, 1.0)
            else:
                inten01 = np.clip(inten / 65535.0, 0.0, 1.0)
            break
    return xyz, inten01


def load_cloud(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Load .laz/.las/.ply → (xyz, intensity01)."""
    p = _as_path(path)
    suf = p.suffix.lower()
    if suf in (".laz", ".las"):
        return load_laz_xyzi(p)
    if suf == ".ply":
        return load_ply_xyzi(p)
    raise ValueError(f"Unsupported cloud format: {p}")


def resolve_tile_prefix(path_or_prefix: PathLike) -> Path:
    """
    Accept either a tile prefix (.../name) or any of
    name_scan.laz / name_instances.npy / name_meta.json.
    """
    p = _as_path(path_or_prefix)
    if p.is_file():
        name = p.name
        for suf in ("_scan.laz", "_scan.las", "_instances.npy", "_meta.json"):
            if name.endswith(suf):
                return p.parent / name[: -len(suf)]
        return p.with_suffix("")
    if (Path(str(p) + "_scan.laz")).exists() or (Path(str(p) + "_scan.las")).exists():
        return p
    if p.is_dir():
        raise ValueError(
            f"Pass a tile prefix (e.g. .../l1w_t00_03), not a directory: {p}"
        )
    return p


def load_tile_prefix(
    path_or_prefix: PathLike,
    *,
    require_instances: bool = False,
) -> Dict[str, Any]:
    """
    Load prepared tile triplet.

    Returns dict with keys: prefix, name, xyz, intensity, labels (or None), meta.
    """
    prefix = resolve_tile_prefix(path_or_prefix)
    laz = Path(str(prefix) + "_scan.laz")
    las = Path(str(prefix) + "_scan.las")
    cloud_path = laz if laz.exists() else las
    if not cloud_path.exists():
        raise FileNotFoundError(f"Missing scan next to prefix {prefix}")

    xyz, intensity = load_cloud(cloud_path)
    inst_path = Path(str(prefix) + "_instances.npy")
    labels: Optional[np.ndarray] = None
    if inst_path.exists():
        labels = np.load(str(inst_path)).astype(np.int32).reshape(-1)
        if len(labels) != len(xyz):
            raise ValueError(
                f"instances length {len(labels)} != points {len(xyz)} for {prefix}"
            )
    elif require_instances:
        raise FileNotFoundError(inst_path)

    meta_path = Path(str(prefix) + "_meta.json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "prefix": prefix,
        "name": prefix.name,
        "xyz": xyz,
        "intensity": intensity,
        "labels": labels,
        "meta": meta,
        "cloud_path": cloud_path,
    }


def save_tile(
    out_dir: PathLike,
    name: str,
    xyz: np.ndarray,
    intensity01: np.ndarray,
    labels: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    *,
    write_preview_ply: bool = True,
    max_ply_points: int = 500_000,
) -> Dict[str, str]:
    """
    Write Ecomodel tile layout:
      {name}_scan.laz, {name}_instances.npy, {name}_meta.json
    Optionally a colored preview PLY.
    """
    out = Path(out_dir).expanduser()
    if not out.is_absolute():
        out = (_SP_DIR / out).resolve()
    else:
        out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64)
    inten = np.asarray(intensity01, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = len(xyz)
    if not (len(inten) == n and len(labels) == n):
        raise ValueError(f"length mismatch xyz={n} inten={len(inten)} labels={len(labels)}")

    inten_u16 = np.clip(np.round(inten * 65535.0), 0, 65535).astype(np.uint16)

    laz_path = out / f"{name}_scan.laz"
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = inten_u16
    las.write(str(laz_path))

    npy_path = out / f"{name}_instances.npy"
    np.save(str(npy_path), labels)

    n_trees = int(len(np.unique(labels[labels >= 0])))
    payload = {
        "tile_name": name,
        "num_points": n,
        "num_trees": n_trees,
        "label_schema": {"ground": -1, "trees": ">=0"},
        "edited": True,
    }
    if meta:
        payload.update(meta)
        payload["edited"] = True
        payload["num_points"] = n
        payload["num_trees"] = n_trees
    meta_path = out / f"{name}_meta.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    paths = {
        "laz": str(laz_path),
        "instances": str(npy_path),
        "meta": str(meta_path),
    }

    if write_preview_ply:
        from .viz import instance_colors, write_colored_ply

        ply_path = out / f"{name}_preview.ply"
        if n > max_ply_points:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, size=max_ply_points, replace=False))
        else:
            idx = np.arange(n)
        write_colored_ply(ply_path, xyz[idx], instance_colors(labels[idx]))
        paths["ply"] = str(ply_path)

    return paths
