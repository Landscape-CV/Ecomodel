"""Island-wide tree statistics from label counts plus a coarse octree sample."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .store import Project

SAMPLE_POINTS = 6_000_000


def build_sample(proj: Project, budget: int = SAMPLE_POINTS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xyz float32, tile int16, idx uint32) from the coarsest nodes, up to ~budget points."""
    order = np.argsort(proj.node_level[1:], kind="stable") + 1
    cum = np.cumsum(proj.node_n[order])
    k = int(np.searchsorted(cum, budget)) + 1
    xs, ts, ids = [], [], []
    for g in order[:k]:
        if proj.node_n[g] == 0:
            continue
        xyz, _, idx = proj.read_node(int(g))
        xs.append(xyz)
        ids.append(idx.astype(np.uint32))
        ts.append(np.full(len(idx), proj.node_tile[g], dtype=np.int16))
    if not xs:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int16), np.zeros(0, np.uint32)
    return np.concatenate(xs), np.concatenate(ts), np.concatenate(ids)


def label_counts(proj: Project) -> Dict[int, np.ndarray]:
    """Per-tile bincount of tree labels (index = tree id)."""
    out = {}
    for t in range(len(proj.tiles)):
        lab = np.asarray(proj.labels(t))
        pos = lab[lab >= 0]
        out[t] = np.bincount(pos).astype(np.int64) if len(pos) else np.zeros(0, np.int64)
    return out


def tree_geometry(pts: np.ndarray, ids: np.ndarray, base_band: float = 1.0) -> Dict[int, Dict]:
    """Per tree: center, bmin, bmax, base (xy centroid of the lowest base_band metres), height."""
    stats: Dict[int, Dict] = {}
    if len(ids) == 0:
        return stats
    pts = pts.astype(np.float64)
    uid, inv = np.unique(ids, return_inverse=True)
    k = len(uid)
    cnt = np.bincount(inv, minlength=k).astype(np.float64)
    lo = np.full((k, 3), np.inf)
    hi = np.full((k, 3), -np.inf)
    for ax in range(3):
        np.minimum.at(lo[:, ax], inv, pts[:, ax])
        np.maximum.at(hi[:, ax], inv, pts[:, ax])
    low = pts[:, 2] < lo[inv, 2] + base_band
    bc = np.bincount(inv[low], minlength=k).astype(np.float64)
    bx = np.bincount(inv[low], pts[low, 0], minlength=k) / np.maximum(bc, 1)
    by = np.bincount(inv[low], pts[low, 1], minlength=k) / np.maximum(bc, 1)
    cx = [np.bincount(inv, pts[:, a], minlength=k) / cnt for a in range(3)]
    for j, tid in enumerate(uid.tolist()):
        stats[tid] = {
            "center": [round(float(cx[a][j]), 3) for a in range(3)],
            "bmin": np.round(lo[j], 3).tolist(),
            "bmax": np.round(hi[j], 3).tolist(),
            "base": [round(float(bx[j]), 3), round(float(by[j]), 3)],
            "height": round(float(hi[j, 2] - lo[j, 2]), 2),
        }
    return stats


def island_trees(proj: Project, counts: Dict[int, np.ndarray], sample) -> list:
    """Tree table rows (project-local coordinates) from counts and a sample."""
    total: Dict[int, int] = {}
    tiles_of: Dict[int, list] = {}
    for t, cnt in counts.items():
        nz = np.flatnonzero(cnt)
        for i, c in zip(nz.tolist(), cnt[nz].tolist()):
            total[i] = total.get(i, 0) + c
            tiles_of.setdefault(i, []).append(t)
    sxyz, stile, sidx = sample
    slab = proj.gather_labels(stile.astype(np.int64), sidx) if len(sidx) else np.zeros(0, np.int32)
    keep = slab >= 0
    stats = tree_geometry(sxyz[keep], slab[keep])
    reviewed = set(int(v) for v in proj.meta.get("reviewed", []))
    out = []
    for tid in sorted(total):
        s = stats.get(tid)
        out.append({
            "id": tid, "count": total[tid], "tiles": tiles_of.get(tid, []),
            "reviewed": tid in reviewed,
            "center": s["center"] if s else None,
            "bmin": s["bmin"] if s else None,
            "bmax": s["bmax"] if s else None,
            "base": s["base"] if s else None,
            "height": s["height"] if s else 0.0,
            "zmin": s["bmin"][2] if s else 0.0,
        })
    return out
