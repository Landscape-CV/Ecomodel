"""Find and merge trees that a segmenter split into several pieces.

Works on any labelled cloud (scanline, TreeX, Point-SAM, ...). Each instance is described by its
height above a lowest-point terrain proxy: an instance with points near the ground has a *stem*,
one without is a *fragment* (crown piece, broken-off branch). Touching instances are found by
shared or face-adjacent voxels.

Rules, per touching pair (a, b):
  - fragment + stemmed tree   -> merge the fragment into its best-touching stemmed tree (auto)
  - fragment + fragment       -> merge (auto) when they touch strongly
  - two stems within STEM_SAME m of each other -> one tree split vertically/sideways (auto)
  - two stemmed trees otherwise -> review candidate only (never auto)
  - a fragment touching two hosts about equally -> review candidate only
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

VOXEL = 0.3          # contact voxel size (m)
STEM_HAG = 2.0       # an instance "has a stem" if it has points below this height above ground
STEM_MIN_PTS = 10
STEM_SAME = 0.6      # stems closer than this (xy, m) belong to the same tree
AUTO_FRAG = 0.02     # fragment auto-merge needs contact score >= this
AUTO_FLOAT = 0.15    # fragment+fragment auto-merge threshold
CAND_MIN = 0.05      # stem+stem pairs reported for review at or above this score
MIN_FRAG_PTS = 0     # fragments smaller than this are merged regardless of contact score (0 = off)
AMBIGUOUS = 0.3      # a fragment touching a second host at >= this fraction of the best is left for review


def _keys(ijk: np.ndarray) -> np.ndarray:
    ijk = ijk.astype(np.int64) + (1 << 20)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def _contacts(ijk: np.ndarray, inv: np.ndarray, n_lab: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs (ia, ib) of label indices touching, contact voxel counts, and per-label voxel footprints."""
    vox = np.unique(np.stack([_keys(ijk), inv], axis=1), axis=0)
    foot = np.bincount(vox[:, 1], minlength=n_lab)
    vk, vl = vox[:, 0], vox[:, 1]
    codes = []
    for d in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
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
        codes.append(np.stack([np.minimum(la, lb)[diff], np.maximum(la, lb)[diff], kk[diff]], axis=1))
    if not codes:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64), foot
    cc = np.unique(np.concatenate(codes), axis=0)
    pk, n_contact = np.unique(cc[:, :2], axis=0, return_counts=True)
    return pk, n_contact, foot


def analyse(xyz: np.ndarray, labels: np.ndarray, *, ground_xyz: Optional[np.ndarray] = None,
            voxel: float = VOXEL) -> Dict:
    """Per-instance stem info and touching pairs. ``ground_xyz`` (e.g. all points) builds the terrain."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from promptable_instance import BaseSurface

    surf = BaseSurface(ground_xyz if ground_xyz is not None and len(ground_xyz) else xyz)
    m = labels >= 0
    xyz, lab = xyz[m], labels[m].astype(np.int64)
    ul, inv = np.unique(lab, return_inverse=True)
    hag = surf.height(xyz)
    low = hag < STEM_HAG
    n_low = np.bincount(inv[low], minlength=len(ul))
    has_stem = n_low >= STEM_MIN_PTS
    sx = np.bincount(inv[low], weights=xyz[low, 0], minlength=len(ul)) / np.maximum(n_low, 1)
    sy = np.bincount(inv[low], weights=xyz[low, 1], minlength=len(ul)) / np.maximum(n_low, 1)
    size = np.bincount(inv, minlength=len(ul))
    cx = np.bincount(inv, weights=xyz[:, 0], minlength=len(ul)) / np.maximum(size, 1)
    cy = np.bincount(inv, weights=xyz[:, 1], minlength=len(ul)) / np.maximum(size, 1)
    ijk = np.floor(xyz / voxel).astype(np.int64)
    pk, nc, foot = _contacts(ijk, inv, len(ul))
    return {"ids": ul, "has_stem": has_stem, "stem_xy": np.stack([sx, sy], 1), "size": size,
            "centroid": np.stack([cx, cy], 1), "pairs": pk, "contacts": nc, "foot": foot}


def find_merges(info: Dict) -> Tuple[List[Dict], List[Dict]]:
    """(auto merges, review candidates); each {a, b, score, reason}, b merges into a."""
    ids, stem, foot, size = info["ids"], info["has_stem"], info["foot"], info["size"]
    best_for_frag: Dict[int, List[Tuple[float, int, float, str, bool]]] = {}
    auto, cands = [], []
    for (ia, ib), nc in zip(info["pairs"].tolist(), info["contacts"].tolist()):
        score = float(nc / max(1, min(foot[ia], foot[ib])))
        sa, sb = bool(stem[ia]), bool(stem[ib])
        if sa and sb:
            d = float(np.hypot(*(info["stem_xy"][ia] - info["stem_xy"][ib])))
            big, small = (ia, ib) if size[ia] >= size[ib] else (ib, ia)
            if d < STEM_SAME:
                auto.append({"a": int(ids[big]), "b": int(ids[small]), "score": round(score, 3), "reason": "same stem"})
            elif score >= CAND_MIN:
                cands.append({"a": int(ids[big]), "b": int(ids[small]), "score": round(min(score, 1.0), 3),
                              "reason": "touching trees"})
            continue
        if sa != sb:
            frag, host = (ib, ia) if sa else (ia, ib)
            reason = "crown fragment"
            ok = score >= AUTO_FRAG or (MIN_FRAG_PTS and size[frag] < MIN_FRAG_PTS)
        else:
            frag, host = (ib, ia) if size[ia] >= size[ib] else (ia, ib)
            reason = "floating fragments"
            ok = score >= AUTO_FLOAT
        # A fragment joins only its best host; stemmed hosts beat floating ones.
        rank = (1.0 if stem[host] else 0.0) + score
        best_for_frag.setdefault(frag, []).append((rank, host, score, reason, bool(ok)))
    for frag, opts in best_for_frag.items():
        opts.sort(key=lambda o: -o[0])
        _, host, score, reason, ok = opts[0]
        if len(opts) > 1 and int(opts[1][0]) == int(opts[0][0]) and opts[1][2] >= AMBIGUOUS * score:
            ok, reason = False, f"ambiguous fragment (also touches {int(ids[opts[1][1]])})"
        item = {"a": int(ids[host]), "b": int(ids[frag]), "score": round(min(score, 1.0), 3), "reason": reason}
        (auto if ok else cands).append(item)
    return auto, cands


def union_relabel(labels: np.ndarray, merges: List[Dict]) -> Tuple[np.ndarray, int]:
    """Apply b->a merges transitively. Returns (new labels, merges applied)."""
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    n = 0
    for mg in merges:
        ra, rb = find(mg["a"]), find(mg["b"])
        if ra != rb:
            parent[rb] = ra
            n += 1
    if not n:
        return labels, 0
    src = np.array(sorted(parent), dtype=np.int64)
    dst = np.array([find(int(s)) for s in src], dtype=np.int64)
    out = labels.copy()
    hit = np.isin(out, src)
    out[hit] = dst[np.searchsorted(src, out[hit])].astype(out.dtype)
    return out, n


def merge_fragments(xyz: np.ndarray, labels: np.ndarray, *, ground_xyz: Optional[np.ndarray] = None,
                    voxel: float = VOXEL, sample_voxel: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """Auto-merge split trees. Returns (labels, report{merged, auto, candidates})."""
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels)
    if (labels >= 0).sum() == 0:
        return labels, {"merged": 0, "auto": [], "candidates": []}
    # Analyse a voxel sample: one point per sample_voxel cell keeps contacts and stems intact.
    keys = np.floor(xyz / sample_voxel).astype(np.int64)
    _, first = np.unique(_keys(keys), return_index=True)
    info = analyse(xyz[first], labels[first], ground_xyz=xyz[first] if ground_xyz is None else ground_xyz,
                   voxel=voxel)
    auto, cands = find_merges(info)
    out, n = union_relabel(labels, auto)
    return out, {"merged": n, "auto": auto, "candidates": sorted(cands, key=lambda c: -c["score"])}
