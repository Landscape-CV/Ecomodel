"""Island-wide split-tree detection: merge_fragments per tile with a buffer, results as merge candidates.

Each tile is cropped with BUFFER m around it so trees crossing the border are seen whole; a pair is
only reported by the tile that owns the fragment's centroid, so every pair comes from one tile.
Confident merges can be applied straight away; the rest join the candidate list (source "fragments")
next to the border candidates.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..merge_fragments import analyse, find_merges, sample_indices
from .stitch import apply_candidates, save_candidates
from .store import Project

BUFFER = 5.0
SAMPLE_VOXEL = 0.1

Progress = Optional[Callable[[float, float, str], None]]


def _sample_box(proj: Project, lo, hi, voxel: float):
    """One point per voxel inside an xy box, read one octree node at a time.

    A tile crop can be a couple of hundred million points. Voxel codes are built per node, in
    slices, so the full-resolution cloud is never turned into an (N, 3) int64 index.
    """
    lo = np.asarray(lo, np.float32)
    hi = np.asarray(hi, np.float32)
    xs, ts, ids = [], [], []
    for g in proj.nodes_intersecting(lo, hi):
        xyz, _, idx = proj.read_node(int(g))
        m = (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
        if not np.any(m):
            continue
        x = xyz[m]
        keep = sample_indices(x, voxel)
        xs.append(x[keep])
        ids.append(idx[m][keep])
        ts.append(np.full(len(keep), int(proj.node_tile[g]), dtype=np.int16))
    if not xs:
        return None
    xyz = np.concatenate(xs)
    tile = np.concatenate(ts)
    ix = np.concatenate(ids)
    keep = sample_indices(xyz, voxel)
    return xyz[keep], tile[keep], ix[keep]


def find_fragment_candidates(project: Project | str, *, auto_apply: bool = True, progress: Progress = None,
                             log: Callable[[str], None] = print) -> Dict:
    proj = project if isinstance(project, Project) else Project(project)
    prog = progress or (lambda d, t, m: None)
    auto: Dict[str, Dict] = {}
    review: Dict[str, Dict] = {}
    nt = len(proj.tiles)
    for t in range(nt):
        if proj.tiles[t]["n"] == 0:
            continue
        prog(t, nt, f"Checking tile {t + 1}/{nt} for split trees")
        lo, hi = proj.tile_bounds_xy(t)
        sampled = _sample_box(proj, lo - BUFFER, hi + BUFFER, SAMPLE_VOXEL)
        if sampled is None:
            continue
        xyz, tile, ix = sampled
        lab = proj.gather_labels(tile.astype(np.int64), ix)
        if (lab >= 0).sum() == 0:
            continue
        info = analyse(xyz, lab, ground_xyz=xyz)
        owner = proj.owner_tile(info["centroid"])
        own = {int(i) for i, o in zip(info["ids"], owner) if int(o) == t}
        a_list, c_list = find_merges(info)
        for items, bucket in ((a_list, auto), (c_list, review)):
            for c in items:
                if c["b"] not in own:
                    continue
                key = f"{min(c['a'], c['b'])}_{max(c['a'], c['b'])}"
                cx, cy = info["centroid"][np.searchsorted(info["ids"], c["b"])]
                bucket[key] = {"key": key, **c, "contacts": 0, "location": [round(float(cx), 2), round(float(cy), 2),
                               round(float(np.median(xyz[lab == c["b"], 2])), 2)], "status": "open",
                               "source": "fragments"}
    prog(nt, nt, "Done")
    merged = 0
    if auto_apply:
        for k in auto:
            review.pop(k, None)
        if auto:
            merged = apply_candidates(proj, list(auto.values()), log=log)
    else:
        review.update(auto)
    cands = sorted(review.values(), key=lambda c: -c["score"])
    save_candidates(proj, cands, replace_source="fragments")
    log(f"[fragments] merged {merged} split pieces, {len(cands)} pairs for review")
    return {"merged": merged, "candidates": len(cands)}
