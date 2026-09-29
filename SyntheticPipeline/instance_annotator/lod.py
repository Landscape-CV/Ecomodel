"""Voxel level-of-detail for display, with an exact full-resolution mapping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_DISPLAY = 2_000_000


@dataclass
class Lod:
    """``display_idx[j]`` is the full-cloud point shown as display point j.

    ``rep_of_full[i]`` is the display point representing full point i, so an
    edit on display points expands exactly to ``sel_display[rep_of_full]``.
    """

    display_idx: np.ndarray  # (M,) int64
    rep_of_full: np.ndarray  # (N,) int32/int64
    voxel_size: float  # 0.0 when identity

    @property
    def is_identity(self) -> bool:
        return self.voxel_size == 0.0

    def expand(self, display_sel: np.ndarray) -> np.ndarray:
        """Display indices (or display bool mask) -> full-resolution bool mask."""
        sel = np.asarray(display_sel)
        m = len(self.display_idx)
        if sel.dtype != bool:
            mask = np.zeros(m, dtype=bool)
            ids = sel.astype(np.int64)
            ids = ids[(ids >= 0) & (ids < m)]
            mask[ids] = True
        else:
            if len(sel) != m:
                raise ValueError(f"display mask length {len(sel)} != {m}")
            mask = sel
        return mask[self.rep_of_full]

    def display_of_full(self, full_idx: np.ndarray) -> np.ndarray:
        """Unique display indices touched by the given full indices."""
        if len(full_idx) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(self.rep_of_full[np.asarray(full_idx, dtype=np.int64)]).astype(np.int64)


def _voxel_keys(xyz: np.ndarray, vs: float) -> np.ndarray:
    ijk = np.floor((xyz - xyz.min(axis=0)) / vs).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]


def _occupied(xyz: np.ndarray, vs: float) -> int:
    return int(len(np.unique(_voxel_keys(xyz, vs))))


def build_lod(xyz: np.ndarray, max_points: int = DEFAULT_MAX_DISPLAY, seed: int = 0) -> Lod:
    """Voxel-grid LOD with at most ``max_points`` display points.

    Voxel size is searched on a subsample (cheap), then applied to the full
    cloud and enlarged until the cap holds.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n <= max_points:
        return Lod(np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), 0.0)

    rng = np.random.default_rng(seed)
    n_s = min(n, 4 * max_points)
    sample = xyz[rng.choice(n, size=n_s, replace=False)] if n_s < n else xyz
    target_s = max_points * n_s / n * 1.5 if n_s < n else max_points

    ext = np.ptp(xyz, axis=0)
    ext = np.maximum(ext, 1e-6)
    lo, hi = 1e-4, float(ext.max())
    vs = float((np.prod(ext) / max_points) ** (1.0 / 3.0))
    for _ in range(18):
        occ = _occupied(sample, vs)
        if occ > target_s:
            lo = vs
        else:
            hi = vs
        if hi / lo < 1.05:
            break
        vs = float(np.sqrt(lo * hi))
    vs = hi

    for _ in range(8):
        keys = _voxel_keys(xyz, vs)
        perm = rng.permutation(n)
        uniq, first, inv = np.unique(keys[perm], return_index=True, return_inverse=True)
        if len(uniq) <= max_points:
            break
        vs *= (len(uniq) / max_points) ** (1.0 / 2.0) * 1.02

    display_idx = perm[first].astype(np.int64)
    rep_perm = inv.reshape(-1)
    order = np.argsort(display_idx)
    display_idx = display_idx[order]
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    rep_of_full = np.empty(n, dtype=np.int64)
    rep_of_full[perm] = rank[rep_perm]
    return Lod(display_idx, rep_of_full, float(vs))
