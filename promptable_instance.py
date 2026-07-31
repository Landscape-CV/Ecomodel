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
