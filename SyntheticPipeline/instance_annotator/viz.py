"""Color helpers and simple geometry picks for instance-labeled point clouds."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def instance_colors(labels: np.ndarray) -> np.ndarray:
    """Deterministic RGB in [0,1] for instance IDs (-1 → gray)."""
    rgb = np.full((len(labels), 3), 0.55, dtype=np.float64)
    ids = labels.astype(np.int64)
    pos = ids >= 0
    if not np.any(pos):
        return rgb
    u = ids[pos]
    h = (u * 2654435761) % (2**32)
    rgb[pos, 0] = ((h % 256) / 255.0) * 0.85 + 0.1
    rgb[pos, 1] = (((h // 256) % 256) / 255.0) * 0.85 + 0.1
    rgb[pos, 2] = (((h // 65536) % 256) / 255.0) * 0.85 + 0.1
    return rgb


def write_colored_ply(path: PathLike, xyz: np.ndarray, rgb01: np.ndarray) -> None:
    """Binary little-endian colored PLY."""
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.clip(np.asarray(rgb01) * 255.0, 0, 255).astype(np.uint8)
    m = len(xyz)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {m}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        f.write(header)
        verts = np.empty(
            m,
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ],
        )
        verts["x"] = xyz[:, 0]
        verts["y"] = xyz[:, 1]
        verts["z"] = xyz[:, 2]
        verts["r"] = rgb[:, 0]
        verts["g"] = rgb[:, 1]
        verts["b"] = rgb[:, 2]
        f.write(verts.tobytes())


# Material reference colors (wood / leaf / unknown)
_WOOD_RGB = np.array([139 / 255.0, 90 / 255.0, 43 / 255.0])   # #8B5A2B
_LEAF_RGB = np.array([46 / 255.0, 139 / 255.0, 87 / 255.0])   # #2E8B57
_UNKNOWN_RGB = np.array([0.55, 0.55, 0.55])


def material_colors(
    wood_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    """RGB in [0,1]: wood brown, leaf green, neither gray. Wood wins on overlap."""
    wood = np.asarray(wood_mask, dtype=bool)
    leaf = np.asarray(leaf_mask, dtype=bool)
    rgb = np.tile(_UNKNOWN_RGB, (len(wood), 1))
    rgb[leaf & ~wood] = _LEAF_RGB
    rgb[wood] = _WOOD_RGB
    return rgb


def selection_from_click(
    xyz: np.ndarray,
    click_full_idx: int,
    radius_m: float,
) -> np.ndarray:
    """Boolean mask: points within radius_m of the given full-cloud point."""
    xyz = np.asarray(xyz, dtype=np.float64)
    i = int(click_full_idx)
    if i < 0 or i >= len(xyz):
        return np.zeros(len(xyz), dtype=bool)
    c = xyz[i]
    d2 = np.sum((xyz - c) ** 2, axis=1)
    return d2 <= (float(radius_m) ** 2)
