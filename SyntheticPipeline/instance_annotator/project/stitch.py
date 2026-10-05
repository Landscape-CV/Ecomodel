"""Border-merge candidates: trees that touch across tile borders (e.g. imported per-tile labels).

For every tile edge that has a neighbouring tile, points within STRIP of the edge are voxelized
at CONTACT; two tree IDs sharing voxels (or face-adjacent voxels) are a contact. Pairs whose
points sit mostly in different tiles are candidates, scored by contacts relative to the smaller
tree's footprint in the strip.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .store import Project

STRIP = 1.0
CONTACT = 0.3
NEIGHBOUR_GAP = 2.0

Progress = Optional[Callable[[float, float, str], None]]


def _keys(ijk: np.ndarray) -> np.ndarray:
    ijk = ijk.astype(np.int64) + (1 << 20)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def _edge_strips(proj: Project) -> List[tuple]:
    """(tile, lo, hi) xy boxes along tile edges that face another tile."""
    boxes = [proj.tile_bounds_xy(t) for t in range(len(proj.tiles))]
    out = []
    for t, (lo, hi) in enumerate(boxes):
        if proj.tiles[t]["n"] == 0:
            continue
        for axis in (0, 1):
            for side, edge in ((0, lo[axis]), (1, hi[axis])):
                # Area just beyond this edge; a neighbour must overlap it.
                olo, ohi = lo.copy(), hi.copy()
                if side == 0:
                    olo[axis], ohi[axis] = edge - NEIGHBOUR_GAP, edge
                else:
                    olo[axis], ohi[axis] = edge, edge + NEIGHBOUR_GAP
                has_nb = any(
                    u != t and proj.tiles[u]["n"] > 0
                    and np.all(np.minimum(ohi, hi2) - np.maximum(olo, lo2) > 0)
                    for u, (lo2, hi2) in enumerate(boxes))
                if not has_nb:
                    continue
                slo, shi = lo.copy() - STRIP, hi.copy() + STRIP
                slo[axis], shi[axis] = edge - STRIP, edge + STRIP
                out.append((t, slo, shi))
    return out


def find_candidates(project: Project | str, *, progress: Progress = None,
                    log: Callable[[str], None] = lambda s: None, save: bool = True) -> List[Dict]:
    proj = project if isinstance(project, Project) else Project(project)
    prog = progress or (lambda d, t, m: None)
    strips = _edge_strips(proj)
    pairs: Dict[tuple, Dict] = {}
    for k, (t, lo, hi) in enumerate(strips):
        prog(k, len(strips), f"Scanning border strip {k + 1}/{len(strips)}")
        crop = proj.crop(lo, hi)
        if len(crop["idx"]) == 0:
            continue
        tile = crop["tile"].astype(np.int64)
        lab = proj.gather_labels(tile, crop["idx"])
        m = lab >= 0
        if not m.any():
            continue
        xyz, lab, tile = crop["xyz"][m], lab[m].astype(np.int64), tile[m]
        ijk = np.floor(xyz / CONTACT).astype(np.int64)
        # Dominant tile of each label inside the strip.
        ul, inv = np.unique(lab, return_inverse=True)
        nt = len(proj.tiles)
        tc = np.bincount(inv * nt + tile, minlength=len(ul) * nt).reshape(len(ul), nt)
        dom = tc.argmax(axis=1)
        vox = np.unique(np.stack([_keys(ijk), inv], axis=1), axis=0)
        foot = np.bincount(vox[:, 1], minlength=len(ul))
        vk, vl = vox[:, 0], vox[:, 1]
        # Contacts: another label in the same voxel or a face neighbour (+x, +y, +z).
        contact_codes = []
        for d in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
            sh = np.unique(np.stack([_keys(ijk + np.array(d)), inv], axis=1), axis=0)
            ak, al = sh[:, 0], sh[:, 1]
            left = np.searchsorted(vk, ak, "left")
            cnt = np.searchsorted(vk, ak, "right") - left
            if cnt.sum() == 0:
                continue
            rep = np.repeat(np.arange(len(ak)), cnt)
            bi = left[rep] + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            la, lb, kk = al[rep], vl[bi], ak[rep]
            diff = la != lb
            contact_codes.append(np.stack([np.minimum(la, lb)[diff], np.maximum(la, lb)[diff], kk[diff]], axis=1))
        if not contact_codes:
            continue
        cc = np.unique(np.concatenate(contact_codes), axis=0)
        if len(cc) == 0:
            continue
        pk, n_contact = np.unique(cc[:, :2], axis=0, return_counts=True)
        for (ia, ib), nc in zip(pk.tolist(), n_contact.tolist()):
            if dom[ia] == dom[ib]:
                continue
            a, b = int(ul[ia]), int(ul[ib])
            if foot[ib] > foot[ia]:
                a, b, ia, ib = b, a, ib, ia
            score = nc / max(1, min(foot[ia], foot[ib]))
            sel = (cc[:, 0] == min(ia, ib)) & (cc[:, 1] == max(ia, ib))
            kk = cc[sel, 2]
            ij = np.stack([(kk >> 42) & 0x1FFFFF, (kk >> 21) & 0x1FFFFF, kk & 0x1FFFFF], axis=1) - (1 << 20)
            loc = ((ij + 0.5) * CONTACT).mean(axis=0)
            key = (min(a, b), max(a, b))
            prev = pairs.get(key)
            if prev is None or score > prev["score"]:
                pairs[key] = {"key": f"{key[0]}_{key[1]}", "a": a, "b": b, "score": round(float(min(score, 1.0)), 3),
                              "contacts": int(nc), "location": np.round(loc, 2).tolist(), "status": "open",
                              "source": "border"}
    cands = sorted(pairs.values(), key=lambda c: -c["score"])
    prog(len(strips), len(strips), "Done")
    log(f"[stitch] {len(cands)} candidates from {len(strips)} border strips")
    if save:
        save_candidates(proj, cands, replace_source="border")
    return cands


def load_candidates(proj: Project) -> List[Dict]:
    f = proj.root / "stitch_candidates.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_candidates(proj: Project, cands: Sequence[Dict], replace_source: Optional[str] = None) -> None:
    """Write candidates, keeping earlier decisions (accepted/rejected keys are not reopened)."""
    old = load_candidates(proj)
    decided = {c["key"]: c["status"] for c in old if c.get("status", "open") != "open"}
    keep = [c for c in old if c.get("status", "open") != "open"]
    if replace_source is not None:
        keep += [c for c in old if c.get("status", "open") == "open" and c.get("source") != replace_source]
    seen = {c["key"] for c in keep}
    for c in cands:
        if c["key"] in decided or c["key"] in seen:
            continue
        keep.append(dict(c))
        seen.add(c["key"])
    (proj.root / "stitch_candidates.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")


def apply_candidates(project: Project | str, cands: Sequence[Dict], min_score: float = 0.0,
                     log: Callable[[str], None] = print) -> int:
    """Merge every candidate with score >= min_score (b into a, transitively). Returns merges done."""
    proj = project if isinstance(project, Project) else Project(project)
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    n = 0
    for c in cands:
        if c.get("status", "open") != "open" or c["score"] < min_score:
            continue
        ra, rb = find(int(c["a"])), find(int(c["b"]))
        if ra != rb:
            parent[rb] = ra
            n += 1
    if not n:
        return 0
    src = np.array(sorted(parent), dtype=np.int64)
    dst = np.array([find(int(s)) for s in src], dtype=np.int64)
    for t in range(len(proj.tiles)):
        lab = proj.labels(t)
        arr = np.asarray(lab)
        m = np.isin(arr, src)
        if not m.any():
            continue
        vals = arr[m]
        pos = np.searchsorted(src, vals)
        lab[np.flatnonzero(m)] = dst[pos].astype(np.int32)
    proj.flush()
    applied = {c["key"] for c in cands if c.get("status", "open") == "open" and c["score"] >= min_score}
    allc = load_candidates(proj)
    for c in allc:
        if c["key"] in applied:
            c["status"] = "accepted"
    (proj.root / "stitch_candidates.json").write_text(json.dumps(allc, indent=1), encoding="utf-8")
    log(f"[stitch] merged {n} tree pairs")
    return n
