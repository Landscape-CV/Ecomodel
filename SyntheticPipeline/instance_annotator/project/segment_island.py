"""Whole-island instance segmentation with buffered tiles and cross-tile stitching.

Per tile ("chunk"): take the tile box plus a buffer from all neighbouring tiles, voxel-downsample
to seed points, run a segmenter, and keep only the trees whose stem base lies in the tile (so a
tree cut by a tile border is segmented whole, by exactly one chunk). Trees whose base is within
BASE_SLACK of the border are kept by both neighbours and merged afterwards by voxel IoU.
Remaining overlaps go to the tree with the nearest stem base. Labels are then propagated from the
seeds to every full-resolution point and written into the project's label files (old labels are
backed up to labels_backup/).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .store import Project, ProjectError

DEFAULT_BUFFER = 10.0
DEFAULT_VOXEL = 0.05
DEFAULT_MAX_SEEDS = 12_000_000
BASE_BAND = 1.0          # stem base = xy centroid of the lowest metre of a tree
BASE_SLACK = 1.0         # trees with a base this close to the tile border are kept by both sides
IOU_VOXEL = 0.2
IOU_MERGE = 0.3
IOU_CANDIDATE = 0.1      # weaker overlaps are listed for manual review
MIN_PROPAGATE = 0.2
# TreeX's stem search needs dense points (stock TLS preset: 1.5 cm voxels, 2.5 cm DBSCAN, 90 pts);
# at 5 cm seeds it finds no stems at all.
METHOD_DEFAULTS = {"treex": {"voxel": 0.03, "max_seeds": 40_000_000}}
TREEX_STOCK_MAX_VOXEL = 0.04   # coarser seeds use TreeX's relaxed (sparse-cloud) parameters

Runner = Callable[[np.ndarray, np.ndarray], np.ndarray]


def method_defaults(method: str) -> Dict[str, float]:
    d = {"voxel": DEFAULT_VOXEL, "max_seeds": DEFAULT_MAX_SEEDS}
    d.update(METHOD_DEFAULTS.get(method, {}))
    return d
Progress = Optional[Callable[[float, float, str], None]]


def _vox_keys(xyz: np.ndarray, voxel: float, lo: np.ndarray) -> np.ndarray:
    ijk = np.floor((xyz[:, :3] - lo) / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0) if len(ijk) else 0
    dims = ijk.max(axis=0) + 1 if len(ijk) else np.ones(3, np.int64)
    return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]


def _abs_keys(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Voxel keys comparable across chunks (fixed origin at 0, project-local coords)."""
    ijk = np.floor(xyz[:, :3] / voxel).astype(np.int64) + (1 << 20)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def chunk_seeds(proj: Project, lo: np.ndarray, hi: np.ndarray, voxel: float, max_seeds: int):
    """Voxel-downsampled points (one per voxel) inside an xy box, read node by node."""
    zlo = float(proj.meta["bmin"][2])
    lo3 = np.array([lo[0], lo[1], zlo], dtype=np.float64)
    xs, ins = [], []
    lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
    for g in proj.nodes_intersecting(lo, hi):
        xyz, inten, _ = proj.read_node(int(g))
        m = np.all((xyz[:, :2] >= lo32) & (xyz[:, :2] <= hi32), axis=1)
        if not m.any():
            continue
        x = xyz[m]
        _, first = np.unique(_vox_keys(x, voxel, lo3), return_index=True)
        xs.append(x[first])
        ins.append(inten[m][first])
    if not xs:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8), voxel
    xyz = np.concatenate(xs)
    inten = np.concatenate(ins)
    while True:
        _, first = np.unique(_vox_keys(xyz, voxel, lo3), return_index=True)
        xyz, inten = xyz[first], inten[first]
        if len(xyz) <= max_seeds:
            return xyz, inten, voxel
        voxel *= 1.5


def tree_bases(xyz: np.ndarray, lab: np.ndarray):
    """(ids, base xy (k,2), zmin (k,)) for labels >= 0."""
    keep = lab >= 0
    ids, inv = np.unique(lab[keep], return_inverse=True)
    if len(ids) == 0:
        return ids, np.zeros((0, 2)), np.zeros(0)
    p = xyz[keep].astype(np.float64)
    zmin = np.full(len(ids), np.inf)
    np.minimum.at(zmin, inv, p[:, 2])
    low = p[:, 2] <= zmin[inv] + BASE_BAND
    c = np.bincount(inv[low], minlength=len(ids)).astype(np.float64)
    bx = np.bincount(inv[low], p[low, 0], minlength=len(ids)) / np.maximum(c, 1)
    by = np.bincount(inv[low], p[low, 1], minlength=len(ids)) / np.maximum(c, 1)
    return ids, np.stack([bx, by], axis=1), zmin


def _box_dist(xy: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    d = np.maximum(np.maximum(lo[None] - xy, xy - hi[None]), 0.0)
    return np.hypot(d[:, 0], d[:, 1])


def default_runner(method: str, leaf_removal: bool, results_folder: str):
    """Runner taking (xyz, intensity01, seed voxel); the voxel picks TreeX's parameter set."""
    from ..segment import run_method

    def run(xyz: np.ndarray, inten01: np.ndarray, voxel: float) -> np.ndarray:
        lab, info = run_method(xyz, inten01, method, leaf_removal=leaf_removal, results_folder=results_folder,
                               treex_stock_tls=voxel <= TREEX_STOCK_MAX_VOXEL)
        if not info.get("ok"):
            raise RuntimeError(f"{method} failed: {info.get('message', '')}")
        return lab

    return run


def _cache_matches(f: Path, params: np.ndarray) -> bool:
    try:
        with np.load(f) as d:
            return "params" in d.files and np.allclose(d["params"], params)
    except (OSError, ValueError):
        return False


class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def segment_island(project: Project | str, method: str = "treelearn", *, buffer: float = DEFAULT_BUFFER,
                   voxel: Optional[float] = None, max_seeds: Optional[int] = None,
                   tile_names: Optional[Sequence[str]] = None, leaf_removal: bool = False,
                   progress: Progress = None, log: Callable[[str], None] = print,
                   runner: Optional[Runner] = None, reuse_cache: bool = False) -> Dict:
    proj = project if isinstance(project, Project) else Project(project)
    names = [t["name"] for t in proj.tiles]
    if tile_names:
        missing = [n for n in tile_names if n not in names]
        if missing:
            raise ProjectError(f"Unknown tiles: {missing}")
        targets = [names.index(n) for n in tile_names]
    else:
        targets = list(range(len(names)))
    targets = [t for t in targets if proj.tiles[t]["n"] > 0]
    if not targets:
        raise ProjectError("No tiles to segment.")
    cache = proj.root / "seg_cache" / method
    cache.mkdir(parents=True, exist_ok=True)
    md = method_defaults(method)
    voxel = float(voxel or md["voxel"])
    max_seeds = int(max_seeds or md["max_seeds"])
    if runner is None:
        run = default_runner(method, leaf_removal, str(cache / "work"))
    else:
        run = lambda xyz, inten, _vox: runner(xyz, inten)  # noqa: E731
    prog = progress or (lambda d, t, m: None)
    t0 = time.time()
    n_steps = 2 * len(targets) + 1

    # ── A: per-chunk segmentation ────────────────────────────────────────
    chunk_info: Dict[int, Dict] = {}
    for k, t in enumerate(targets):
        name = names[t]
        lo, hi = proj.tile_bounds_xy(t)
        f = cache / f"{name}.npz"
        prog(k, n_steps, f"Segmenting tile {k + 1}/{len(targets)} ({name})")
        params = np.array([voxel, buffer, float(leaf_removal)])
        if reuse_cache and f.exists() and _cache_matches(f, params):
            log(f"[segment] {name}: cached")
        else:
            xyz, inten, vox = chunk_seeds(proj, lo - buffer, hi + buffer, voxel, max_seeds)
            if len(xyz) == 0:
                np.savez(f, xyz=np.zeros((0, 3), np.float32), lab=np.zeros(0, np.int32), tree_ids=np.zeros(0, np.int32),
                         bases=np.zeros((0, 2)), crosses=np.zeros(0, bool), voxel=vox, params=params)
                continue
            shift = xyz.min(axis=0)
            lab = np.asarray(run((xyz - shift).astype(np.float64), inten.astype(np.float64) / 255.0, vox),
                             dtype=np.int32)
            if len(lab) != len(xyz):
                raise RuntimeError(f"segmenter returned {len(lab)} labels for {len(xyz)} points")
            ids, bases, _ = tree_bases(xyz, lab)
            owner = proj.owner_tile(bases) if len(ids) else np.zeros(0, np.int64)
            keep = (owner == t) | (_box_dist(bases, lo, hi) <= BASE_SLACK)
            kept = ids[keep]
            in_core = np.all((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi), axis=1)
            tree_pt = np.isin(lab, kept)
            store = tree_pt | ((lab < 0) & in_core)
            lab_s = np.where(tree_pt, lab, -1)[store]
            xyz_s = xyz[store]
            outside = tree_pt & ~in_core
            n_out = np.bincount(lab[outside], minlength=int(lab.max()) + 1) if outside.any() else np.zeros(int(lab.max()) + 1)
            crosses = n_out[kept] > 0 if len(kept) else np.zeros(0, bool)
            # A neighbour also keeps trees whose base is within BASE_SLACK of its box.
            kb = bases[keep]
            for u in range(len(names)):
                if u != t and len(kb):
                    ulo, uhi = proj.tile_bounds_xy(u)
                    crosses = crosses | (_box_dist(kb, ulo, uhi) <= BASE_SLACK)
            np.savez(f, xyz=xyz_s.astype(np.float32), lab=lab_s.astype(np.int32), tree_ids=kept.astype(np.int32),
                     bases=bases[keep], crosses=crosses, voxel=vox, params=params)
            log(f"[segment] {name}: {len(xyz):,} seeds (voxel {vox:.3f} m), {len(ids)} trees, kept {len(kept)}")
        chunk_info[t] = {"file": f}

    # ── B: merge duplicates of border trees ──────────────────────────────
    prog(len(targets), n_steps, "Matching trees across tile borders")
    trees: List[Dict] = []
    gid_of: Dict[tuple, int] = {}
    vox_max = voxel
    pair_keys, pair_g = [], []
    for t in targets:
        d = np.load(chunk_info[t]["file"]) if t in chunk_info else None
        if d is None:
            continue
        vox_max = max(vox_max, float(d["voxel"]))
        xyz, lab = d["xyz"], d["lab"]
        local_g = {}
        for tid, base, cr in zip(d["tree_ids"].tolist(), d["bases"], d["crosses"]):
            g = len(trees)
            gid_of[(t, tid)] = g
            local_g[tid] = g
            trees.append({"tile": t, "local": int(tid), "base": base.astype(float), "crosses": bool(cr)})
        cross_ids = [tid for tid, cr in zip(d["tree_ids"].tolist(), d["crosses"]) if cr]
        if cross_ids:
            m = np.isin(lab, cross_ids)
            pk = np.unique(np.stack([_abs_keys(xyz[m], IOU_VOXEL), lab[m].astype(np.int64)], axis=1), axis=0)
            lut = np.zeros(int(lab.max()) + 1, dtype=np.int64)
            for tid in cross_ids:
                lut[tid] = local_g[tid]
            pair_keys.append(pk[:, 0])
            pair_g.append(lut[pk[:, 1]])
    G = len(trees)
    uf = _UF(max(G, 1))
    candidates = []
    if pair_keys:
        keys = np.concatenate(pair_keys)
        gs = np.concatenate(pair_g)
        size = np.bincount(gs, minlength=G)
        order = np.argsort(keys, kind="stable")
        keys, gs = keys[order], gs[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        counts = np.diff(np.r_[starts, len(keys)])
        codes = []
        for s, c in zip(starts[counts > 1].tolist(), counts[counts > 1].tolist()):
            grp = np.unique(gs[s:s + c])
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    codes.append(int(grp[a]) * G + int(grp[b]))
        if codes:
            code, inter = np.unique(np.asarray(codes, dtype=np.int64), return_counts=True)
            for cd, it in zip(code.tolist(), inter.tolist()):
                a, b = divmod(cd, G)
                if trees[a]["tile"] == trees[b]["tile"]:
                    continue
                iou = it / (size[a] + size[b] - it)
                if iou > IOU_MERGE:
                    uf.union(a, b)
                elif iou > IOU_CANDIDATE:
                    candidates.append((a, b, float(iou)))
    roots = np.array([uf.find(g) for g in range(G)], dtype=np.int64)
    uroots, final_idx = np.unique(roots, return_inverse=True) if G else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    K = len(uroots)
    first_id = proj.allocate_ids(K) if K else proj.peek_next_id()
    final_of_g = first_id + final_idx
    base_final = np.zeros((K, 2))
    cnt = np.zeros(K)
    for g, tr in enumerate(trees):
        base_final[final_idx[g]] += tr["base"]
        cnt[final_idx[g]] += 1
    base_final /= np.maximum(cnt, 1)[:, None]
    log(f"[segment] {G} kept trees -> {K} after merging duplicates")
    if K == 0:
        raise ProjectError(
            f"{method} found no trees in any tile (seed voxel {voxel} m); existing labels were left unchanged. "
            "Try a smaller voxel or another method. Per-tile results are cached in seg_cache/.")

    # ── C: resolve overlaps and write full-resolution labels ─────────────
    backup = proj.root / "labels_backup"
    backup.mkdir(exist_ok=True)
    proj.flush()
    radius = max(MIN_PROPAGATE, 2.5 * vox_max)
    written = 0
    from scipy.spatial import cKDTree

    for k, t in enumerate(targets):
        name = names[t]
        prog(len(targets) + 1 + k, n_steps, f"Writing labels {k + 1}/{len(targets)} ({name})")
        src = proj.root / "labels" / f"{name}.npy"
        shutil.copyfile(src, backup / f"{name}.npy")
        lo, hi = proj.tile_bounds_xy(t)
        lo1, hi1 = lo - radius - 0.5, hi + radius + 0.5
        sx, sl = [], []
        for c in chunk_info:
            clo, chi = proj.tile_bounds_xy(c)
            if np.any(clo - buffer > hi1) or np.any(chi + buffer < lo1):
                continue
            d = np.load(chunk_info[c]["file"])
            xyz, lab = d["xyz"], d["lab"]
            m = np.all((xyz[:, :2] >= lo1) & (xyz[:, :2] <= hi1), axis=1)
            if not m.any():
                continue
            lab = lab[m]
            fid = np.full(len(lab), -1, dtype=np.int64)
            if (lab >= 0).any():
                lut = np.full(int(lab.max()) + 1, -1, dtype=np.int64)
                for tid in np.unique(lab[lab >= 0]).tolist():
                    lut[tid] = final_of_g[gid_of[(c, tid)]]
                fid[lab >= 0] = lut[lab[lab >= 0]]
            sx.append(xyz[m])
            sl.append(fid)
        lab_arr = proj.labels(t)
        if not sx:
            lab_arr[:] = -1
            continue
        seeds = np.concatenate(sx)
        slab = np.concatenate(sl)
        _resolve_conflicts(seeds, slab, base_final, first_id)
        tree = cKDTree(seeds)
        base_g = int(proj.tile_base[t])
        n_nodes = len(json.loads((proj.root / "tiles" / f"{name}.json").read_text(encoding="utf-8"))["nodes"])
        for g in range(base_g, base_g + n_nodes):
            if proj.node_n[g] == 0:
                continue
            xyz, _, idx = proj.read_node(g)
            dist, nn = tree.query(xyz, k=1, distance_upper_bound=radius, workers=-1)
            hit = np.isfinite(dist)
            out = np.full(len(idx), -1, dtype=np.int32)
            out[hit] = slab[nn[hit]]
            lab_arr[idx] = out
            written += len(idx)
        proj.flush()

    # ── bookkeeping ──────────────────────────────────────────────────────
    members: List[List[int]] = [[] for _ in range(K)]
    for g in range(G):
        members[int(final_idx[g])].append(g)
    owners = proj.owner_tile(base_final) if K else np.zeros(0, np.int64)
    tree_rows = [{"id": int(first_id + j), "base": np.round(base_final[j], 3).tolist(),
                  "tile": names[int(owners[j])],
                  "parts": [[names[trees[g]["tile"]], trees[g]["local"]] for g in members[j]]}
                 for j in range(K)]
    (proj.root / "trees.json").write_text(json.dumps(tree_rows, indent=0), encoding="utf-8")
    if candidates:
        from .stitch import save_candidates

        cands = [{"key": f"{final_of_g[a]}_{final_of_g[b]}", "a": int(final_of_g[a]), "b": int(final_of_g[b]),
                  "score": round(iou, 3), "contacts": 0,
                  "location": np.round(np.r_[(trees[a]["base"] + trees[b]["base"]) / 2, 0.0], 2).tolist(),
                  "status": "open", "source": "segment_iou"}
                 for a, b, iou in candidates if final_of_g[a] != final_of_g[b]]
        save_candidates(proj, cands)
    proj.meta["reviewed"] = []
    proj.meta["segmentation"] = {"method": method, "buffer": buffer, "voxel": voxel,
                                 "tiles": [names[t] for t in targets], "trees": K,
                                 "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    proj.save_meta()
    prog(n_steps, n_steps, "Done")
    summary = {"trees": K, "tiles": len(targets), "kept": G, "points": written,
               "candidates": len(candidates), "seconds": round(time.time() - t0, 1)}
    log(f"[segment] done: {summary}")
    return summary


def _resolve_conflicts(seeds: np.ndarray, fid: np.ndarray, base_final: np.ndarray, first_id: int) -> None:
    """Voxels claimed by several trees go to the tree whose stem base is nearest (in place)."""
    tp = np.flatnonzero(fid >= 0)
    if len(tp) == 0:
        return
    keys = _abs_keys(seeds[tp], IOU_VOXEL)
    pair = np.stack([keys, fid[tp]], axis=1)
    upair, first = np.unique(pair, axis=0, return_index=True)
    ukey, nclaim = np.unique(upair[:, 0], return_counts=True)
    conflict = ukey[nclaim > 1]
    if len(conflict) == 0:
        return
    m = np.isin(upair[:, 0], conflict)
    cp = upair[m]
    xy = seeds[tp[first[m]], :2].astype(np.float64)
    d = np.hypot(*(xy - base_final[cp[:, 1] - first_id]).T)
    order = np.lexsort((d, cp[:, 0]))
    cp = cp[order]
    win_rows = np.r_[True, cp[1:, 0] != cp[:-1, 0]]
    win_key, win_id = cp[win_rows, 0], cp[win_rows, 1]
    sel = np.isin(keys, win_key)
    pos = np.searchsorted(win_key, keys[sel])
    fid[tp[sel]] = win_id[pos]
