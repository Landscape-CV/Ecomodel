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


def _factorize(keys: np.ndarray):
    """(first index of each distinct key, inverse codes); keys numbered by first appearance."""
    try:
        import pandas as pd

        codes, _ = pd.factorize(keys, sort=False)
        codes = codes.astype(np.int64)
        first = np.flatnonzero(np.r_[True, np.diff(np.maximum.accumulate(codes)) > 0])
        return first, codes
    except ImportError:
        _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
        return first, inv.reshape(-1)


def _voxel_keys(xyz: np.ndarray, vs: float, lo=None, hi=None) -> np.ndarray:
    lo = xyz.min(axis=0) if lo is None else lo
    hi = xyz.max(axis=0) if hi is None else hi
    dims = (np.floor((hi - lo) / vs) + 1).astype(np.int64)
    keys = np.floor((xyz[:, 0] - lo[0]) / vs).astype(np.int64) * dims[1]
    keys += np.floor((xyz[:, 1] - lo[1]) / vs).astype(np.int64)
    keys *= dims[2]
    keys += np.floor((xyz[:, 2] - lo[2]) / vs).astype(np.int64)
    return keys


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
    n_s = min(n, max(1_000_000, max_points // 2))
    sample = xyz[rng.choice(n, size=n_s, replace=False)] if n_s < n else xyz
    # A voxel with N/M points appears in an n_s-point sample with p = 1 - exp(-n_s / M).
    target_s = max_points * -np.expm1(-n_s / max_points) if n_s < n else max_points
    target_s *= 0.93  # aim slightly low so the full-cloud pass below usually fits first time

    lo_xyz, hi_xyz = xyz.min(axis=0), xyz.max(axis=0)
    ext = np.maximum(hi_xyz - lo_xyz, 1e-6)
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

    # The first point met in each voxel represents it (first == sorted, so display order is file order).
    for _ in range(8):
        first, codes = _factorize(_voxel_keys(xyz, vs, lo_xyz, hi_xyz))
        if len(first) <= max_points:
            break
        vs *= (len(first) / max_points) ** (1.0 / 2.0) * 1.02
    return Lod(first.astype(np.int64), codes, float(vs))
