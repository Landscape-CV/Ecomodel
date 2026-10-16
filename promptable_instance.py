"""Helpers for promptable instance segmenters (Point-SAM, SNAP)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def intensity_to_rgb(points: np.ndarray) -> np.ndarray:
    """Build [0,1] RGB from intensity (col 3) or constant gray if XYZ-only."""
    n = len(points)
    if points.shape[1] >= 4:
        inten = points[:, 3].astype(np.float64)
        # Heuristic: 16-bit LAS intensity vs already-normalized [0,1]
        if inten.max() > 1.5:
            inten = inten / 65535.0
        inten = np.clip(inten, 0.0, 1.0)
        return np.stack([inten, inten, inten], axis=1)
    return np.full((n, 3), 0.5, dtype=np.float64)


def voxel_downsample_with_index(
    xyz: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (kept_xyz, keep_indices) using a simple grid hash (first point per voxel).
    """
    if voxel_size <= 0 or len(xyz) == 0:
        return xyz.copy(), np.arange(len(xyz), dtype=np.int64)
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    # unique rows → first occurrence
    _, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    return xyz[first], first.astype(np.int64)


def propagate_labels_nn(
    src_xyz: np.ndarray, src_labels: np.ndarray, dst_xyz: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor label transfer from src → dst."""
    from scipy.spatial import cKDTree

    if len(src_xyz) == 0:
        return np.full(len(dst_xyz), -1, dtype=np.int32)
    tree = cKDTree(src_xyz)
    _, nn = tree.query(dst_xyz, k=1)
    return src_labels[nn].astype(np.int32)


def auto_stem_grid_prompts(
    xyz: np.ndarray,
    *,
    grid_spacing: float = 4.0,
    low_z_frac: float = 0.15,
    min_points_per_cell: int = 30,
    max_prompts: int = 64,
) -> List[np.ndarray]:
    """
    Automatic seed clicks: XY grid cells with enough low-Z points → stem candidate.

    Returns list of (3,) arrays (one click each).
    """
    if len(xyz) == 0:
        return []
    z = xyz[:, 2]
    z_lo = np.quantile(z, low_z_frac)
    low = xyz[z <= z_lo]
    if len(low) < min_points_per_cell:
        low = xyz

    xmin, ymin = low[:, 0].min(), low[:, 1].min()
    ix = np.floor((low[:, 0] - xmin) / grid_spacing).astype(int)
    iy = np.floor((low[:, 1] - ymin) / grid_spacing).astype(int)
    prompts = []
    for key in np.unique(np.stack([ix, iy], axis=1), axis=0):
        mask = (ix == key[0]) & (iy == key[1])
        if mask.sum() < min_points_per_cell:
            continue
        cell = low[mask]
        # densest local: median of points near cell median
        click = np.median(cell, axis=0)
        prompts.append(click.astype(np.float64))
        if len(prompts) >= max_prompts:
            break

    if not prompts:
        # fallback: global low-Z median
        prompts.append(np.median(low, axis=0).astype(np.float64))
    return prompts


def oracle_stem_prompts(
    xyz: np.ndarray, gt_labels: np.ndarray, *, low_z_frac: float = 0.10
) -> List[np.ndarray]:
    """
    One stem click per GT instance id (>=0): XY median of lowest-Z fraction.
    """
    prompts = []
    for tid in np.unique(gt_labels):
        if tid < 0:
            continue
        pts = xyz[gt_labels == tid]
        if len(pts) == 0:
            continue
        z_cut = np.quantile(pts[:, 2], low_z_frac)
        stem = pts[pts[:, 2] <= z_cut]
        if len(stem) == 0:
            stem = pts
        click = np.array(
            [np.median(stem[:, 0]), np.median(stem[:, 1]), np.median(stem[:, 2])],
            dtype=np.float64,
        )
        prompts.append(click)
    return prompts


class BaseSurface:
    """Lowest-point terrain proxy on a 2D grid, usable on ground-removed clouds.

    Each cell keeps its minimum Z, capped at the window minimum plus a slope
    allowance so canopy-only cells fall back to nearby low points.
    """

    def __init__(self, xyz: np.ndarray, cell: float = 1.0, win: int = 3, max_slope: float = 0.6):
        from scipy.ndimage import minimum_filter

        self.cell = float(cell)
        self.x0, self.y0 = float(xyz[:, 0].min()), float(xyz[:, 1].min())
        ix = np.floor((xyz[:, 0] - self.x0) / cell).astype(np.int64)
        iy = np.floor((xyz[:, 1] - self.y0) / cell).astype(np.int64)
        self.nx, self.ny = int(ix.max()) + 1, int(iy.max()) + 1
        cmin = np.full(self.nx * self.ny, np.inf)
        np.minimum.at(cmin, iy * self.nx + ix, xyz[:, 2])
        cmin = cmin.reshape(self.ny, self.nx)
        nb = minimum_filter(cmin, size=2 * win + 1, mode="nearest")
        dtm = np.minimum(cmin, nb + max_slope * win * cell)
        for _ in range(64):
            empty = ~np.isfinite(dtm)
            if not empty.any():
                break
            dtm[empty] = minimum_filter(dtm, size=3, mode="nearest")[empty]
        dtm[~np.isfinite(dtm)] = np.nanmin(np.where(np.isfinite(dtm), dtm, np.nan)) if np.isfinite(dtm).any() else 0.0
        self.dtm = dtm

    def z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        fx = np.clip((np.asarray(x) - self.x0) / self.cell - 0.5, 0, self.nx - 1)
        fy = np.clip((np.asarray(y) - self.y0) / self.cell - 0.5, 0, self.ny - 1)
        i, j = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        i1, j1 = np.minimum(i + 1, self.nx - 1), np.minimum(j + 1, self.ny - 1)
        tx, ty, d = fx - i, fy - j, self.dtm
        a = d[j, i] * (1 - tx) + d[j, i1] * tx
        b = d[j1, i] * (1 - tx) + d[j1, i1] * tx
        return a * (1 - ty) + b * ty

    def height(self, xyz: np.ndarray) -> np.ndarray:
        return xyz[:, 2] - self.z(xyz[:, 0], xyz[:, 1])


def detect_stems(
    xyz: np.ndarray,
    *,
    surface: Optional[BaseSurface] = None,
    slice_lo: float = 0.3,
    slice_hi: float = 2.5,
    voxel: float = 0.05,
    eps: float = 0.1,
    min_samples: int = 6,
    min_points: int = 40,
    min_vertical: float = 0.5,
    max_xy_extent: float = 1.0,
    merge_dist: float = 1.5,
    click_height: float = 1.3,
    upper_height: float = 4.0,
) -> List[dict]:
    """Stem candidates from a low height slice: DBSCAN clusters that are vertical and narrow.

    Returns dicts with ``click`` (3,), ``upper`` (3,) or None, ``n`` and ``xy``,
    sorted by point count (largest first).
    """
    from sklearn.cluster import DBSCAN
    from scipy.spatial import cKDTree

    if len(xyz) == 0:
        return []
    surface = surface or BaseSurface(xyz)
    h = surface.height(xyz)
    sl = (h >= slice_lo) & (h <= slice_hi)
    if sl.sum() < min_points:
        return []
    pts, hs = xyz[sl], h[sl]
    _, keep = voxel_downsample_with_index(pts, voxel)
    pts, hs = pts[keep], hs[keep]
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    cands = []
    for c in np.unique(lab[lab >= 0]):
        m = lab == c
        if m.sum() < min_points:
            continue
        p, hh = pts[m], hs[m]
        ext = p[:, :2].max(0) - p[:, :2].min(0)
        if ext.max() > max_xy_extent or (hh.max() - hh.min()) < min_vertical:
            continue
        cands.append((int(m.sum()), np.median(p[:, :2], axis=0)))
    if not cands:
        return []
    cands.sort(key=lambda t: -t[0])
    kept: List[Tuple[int, np.ndarray]] = []
    for n, xy in cands:
        if all(np.hypot(*(xy - k[1])) >= merge_dist for k in kept):
            kept.append((n, xy))

    tree = cKDTree(xyz[:, :2])
    out = []
    for n, xy in kept:
        def near(target_h: float, radius: float):
            idx = tree.query_ball_point(xy, r=radius)
            if not idx:
                return None
            idx = np.asarray(idx)
            best = idx[np.argmin(np.abs(h[idx] - target_h) + 0.5 * np.hypot(*(xyz[idx, :2] - xy).T))]
            return xyz[best].astype(np.float64) if abs(h[best] - target_h) < 1.0 else None

        click = near(click_height, 0.5)
        if click is None:
            continue
        out.append({"click": click, "upper": near(upper_height, 0.8), "n": n, "xy": xy})
    return out


def stem_click_prompts(xyz: np.ndarray, **kw) -> List[np.ndarray]:
    """Stem-based auto prompts: one (K, 3) array of positive clicks per stem."""
    stems = detect_stems(xyz, **kw)
    if not stems:
        return []
    return [np.stack([s["click"]] + ([s["upper"]] if s["upper"] is not None else [])) for s in stems]


def fuse_stem_masks(
    n: int,
    masks: Sequence[np.ndarray],
    logits: Sequence[np.ndarray],
    scores: Sequence[float],
    seeds: Sequence[int],
    *,
    dup_containment: float = 0.8,
    min_points: int = 50,
) -> np.ndarray:
    """Exclusive labels from per-prompt masks (index arrays into 0..n-1).

    - Two prompts are the same tree when each mask contains the other's seed
      point, or one mask is >= ``dup_containment`` inside the other; such
      groups are unioned (union-find).
    - Points claimed by several trees go to the highest per-point logit.
    """
    k = len(masks)
    if k == 0:
        return np.full(n, -1, dtype=np.int32)
    sets = [np.asarray(m, dtype=np.int64) for m in masks]
    sizes = np.array([len(s) for s in sets])
    parent = list(range(k))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Sparse membership: for each point, which masks contain it.
    owner_pt = np.concatenate(sets) if sizes.sum() else np.zeros(0, np.int64)
    owner_id = np.repeat(np.arange(k), sizes)
    order = np.argsort(owner_pt, kind="stable")
    owner_pt, owner_id = owner_pt[order], owner_id[order]

    def masks_at(p: int) -> np.ndarray:
        lo, hi = np.searchsorted(owner_pt, p, "left"), np.searchsorted(owner_pt, p, "right")
        return owner_id[lo:hi]

    seed_in = [set(masks_at(int(s)).tolist()) if s >= 0 else set() for s in seeds]
    for i in range(k):
        for j in seed_in[i]:
            if j != i and i in seed_in[j]:
                parent[find(i)] = find(j)
    # Overlap counts between mask pairs via shared points.
    codes = []
    for d in range(1, k):
        if len(owner_pt) <= d:
            break
        same = owner_pt[d:] == owner_pt[:-d]
        if not same.any():
            break
        a, b = owner_id[:-d][same], owner_id[d:][same]
        codes.append(np.minimum(a, b) * k + np.maximum(a, b))
    if codes:
        pc, inter = np.unique(np.concatenate(codes), return_counts=True)
        for code, c in zip(pc.tolist(), inter.tolist()):
            a, b = divmod(code, k)
            if c / max(1, min(sizes[a], sizes[b])) >= dup_containment:
                parent[find(a)] = find(b)

    best = np.full(n, -np.inf, dtype=np.float32)
    lab = np.full(n, -1, dtype=np.int64)
    for i in range(k):
        r = find(i)
        lg = np.asarray(logits[i], dtype=np.float32) + 1e-3 * float(scores[i])
        s = sets[i]
        upd = lg > best[s]
        best[s[upd]] = lg[upd]
        lab[s[upd]] = r
    ids, inv, counts = np.unique(lab[lab >= 0], return_inverse=True, return_counts=True)
    out = np.full(n, -1, dtype=np.int32)
    remap = np.where(counts >= min_points, np.cumsum(counts >= min_points) - 1, -1)
    out[lab >= 0] = remap[inv]
    return out


def masks_to_instance_labels(
    masks: Sequence[np.ndarray],
    scores: Optional[Sequence[float]] = None,
    *,
    nms_iou: float = 0.5,
    min_points: int = 50,
) -> np.ndarray:
    """
    Convert overlapping binary masks → exclusive instance labels.

    - Drop tiny masks
    - Greedy NMS by score (or mask size)
    - Overlaps: assign to highest-scoring remaining mask
    - Unassigned → -1
    """
    if not masks:
        return np.zeros(0, dtype=np.int32)

    n = len(masks[0])
    kept = []
    for i, m in enumerate(masks):
        m = np.asarray(m, dtype=bool).reshape(-1)
        if m.shape[0] != n:
            raise ValueError(f"Mask {i} length {m.shape[0]} != {n}")
        count = int(m.sum())
        if count < min_points:
            continue
        score = float(scores[i]) if scores is not None else float(count)
        kept.append((score, m))

    if not kept:
        return np.full(n, -1, dtype=np.int32)

    kept.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for score, m in kept:
        suppress = False
        for _, sm in selected:
            inter = np.logical_and(m, sm).sum()
            union = np.logical_or(m, sm).sum()
            iou = inter / union if union > 0 else 0.0
            if iou >= nms_iou:
                suppress = True
                break
        if not suppress:
            selected.append((score, m))

    # confidence canvas
    best_score = np.full(n, -np.inf, dtype=np.float64)
    best_id = np.full(n, -1, dtype=np.int32)
    for inst_id, (score, m) in enumerate(selected):
        update = m & (score > best_score)
        best_score[update] = score
        best_id[update] = inst_id

    return best_id
