"""Build an island project: per-tile additive octrees + label memmaps.

Layout (all coordinates in project-local float = absolute - origin):
  project.json              origin, crs, tiles, next_tree_id, reviewed
  tiles/<name>.oct          node records: idx uint32[n] | xyz uint16[n,3] | intensity uint8[n] | pad
  tiles/<name>.json         per-tile node table (offsets, bounds, quantization, children)
  tiles/<name>.nodeof.npy   uint32 per source point: tile-local node index
  labels/<name>.npy         int32 per source point: island-wide tree id, -1 non-tree
  preview.bin               island overview sample: xyz float32 | tile uint16 | idx uint32
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

PROJECT_VERSION = 1
DEFAULT_GRID = 128
DEFAULT_LEAF_MAX = 150_000
DEFAULT_MAX_DEPTH = 16
PREVIEW_POINTS = 1_500_000
READ_CHUNK = 5_000_000


def record_nbytes(n: int) -> int:
    raw = 11 * n
    return raw + (-raw % 4)


def read_laz_xyzi(path: str | Path, chunk: int = READ_CHUNK):
    """Chunked LAZ read: xyz float64 (file order) and intensity uint8 (raw >> 8)."""
    import laspy

    with laspy.open(str(path)) as f:
        n = int(f.header.point_count)
        xyz = np.empty((n, 3), dtype=np.float64)
        inten = np.zeros(n, dtype=np.uint8)
        has_i = "intensity" in f.header.point_format.dimension_names
        o = 0
        for pts in f.chunk_iterator(chunk):
            m = len(pts)
            xyz[o:o + m, 0] = pts.x
            xyz[o:o + m, 1] = pts.y
            xyz[o:o + m, 2] = pts.z
            if has_i:
                inten[o:o + m] = (np.asarray(pts.intensity, dtype=np.uint16) >> 8).astype(np.uint8)
            o += m
    return xyz[:o], inten[:o]


def laz_header(path: str | Path) -> Dict[str, Any]:
    import laspy

    with laspy.open(str(path)) as f:
        h = f.header
        try:
            crs = h.parse_crs()
            wkt = crs.to_wkt() if crs is not None else ""
        except Exception:
            wkt = ""
        return {
            "n": int(h.point_count),
            "min": [float(v) for v in h.mins],
            "max": [float(v) for v in h.maxs],
            "crs": wkt,
        }


def build_octree(
    local: np.ndarray,
    inten: np.ndarray,
    out_oct: Path,
    *,
    grid: int = DEFAULT_GRID,
    leaf_max: int = DEFAULT_LEAF_MAX,
    max_depth: int = DEFAULT_MAX_DEPTH,
    seed: int = 0,
) -> tuple[List[Dict[str, Any]], np.ndarray]:
    """Additive octree: each node keeps one point per grid cell, the rest go to children.

    Returns (nodes, nodeof) where nodeof[i] is the node holding point i.
    """
    n = len(local)
    nodeof = np.zeros(n, dtype=np.uint32)
    nodes: List[Dict[str, Any]] = []
    if n == 0:
        out_oct.write_bytes(b"")
        return nodes, nodeof
    lo = local.min(axis=0)
    size = float((local.max(axis=0) - lo).max()) * 1.0001 + 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).astype(np.int64)

    nodes.append({"level": 0, "path": "r", "cube": [*lo.tolist(), size]})
    stack = [(0, perm)]
    offset = 0
    with open(out_oct, "wb") as fh:
        while stack:
            nid, idx = stack.pop()
            node = nodes[nid]
            cmin = np.asarray(node["cube"][:3])
            csize = node["cube"][3]
            p = local[idx]
            node["bmin"] = p.min(axis=0).tolist()
            node["bmax"] = p.max(axis=0).tolist()
            node["spacing"] = csize / grid
            if len(idx) <= leaf_max or node["level"] >= max_depth:
                take = idx
                rest = None
            else:
                key = np.zeros(len(idx), dtype=np.int64)
                for ax in range(3):
                    c = np.floor((p[:, ax] - cmin[ax]) * (grid / csize)).astype(np.int64)
                    np.clip(c, 0, grid - 1, out=c)
                    key *= grid
                    key += c
                    del c
                _, first = np.unique(key, return_index=True)
                del key
                keep = np.zeros(len(idx), dtype=bool)
                keep[first] = True
                take = idx[keep]
                rest_mask = ~keep
                rest = idx[rest_mask]
                prest = p[rest_mask]
            # write node record
            pt = local[take]
            qmin = pt.min(axis=0)
            qext = np.maximum(pt.max(axis=0) - qmin, 1e-6)
            q = np.round((pt - qmin) / qext * 65535.0).astype(np.uint16)
            rec = take.astype(np.uint32).tobytes() + q.tobytes() + inten[take].tobytes()
            rec += b"\0" * (-len(rec) % 4)
            fh.write(rec)
            node.update(n=int(len(take)), offset=offset, nbytes=len(rec), qmin=qmin.tolist(), qext=qext.tolist())
            offset += len(rec)
            nodeof[take] = nid
            node["children"] = []
            if rest is not None and len(rest):
                half = csize / 2.0
                oct_ = ((prest[:, 0] >= cmin[0] + half).astype(np.int8)
                        | ((prest[:, 1] >= cmin[1] + half).astype(np.int8) << 1)
                        | ((prest[:, 2] >= cmin[2] + half).astype(np.int8) << 2))
                order = np.argsort(oct_, kind="stable")
                counts = np.bincount(oct_, minlength=8)
                starts = np.concatenate([[0], np.cumsum(counts)])
                for o in range(8):
                    if counts[o] == 0:
                        continue
                    sub = rest[order[starts[o]:starts[o + 1]]]
                    ccmin = cmin + half * np.array([o & 1, (o >> 1) & 1, (o >> 2) & 1])
                    cid = len(nodes)
                    nodes.append({"level": node["level"] + 1, "path": node["path"] + str(o),
                                  "cube": [*ccmin.tolist(), half], "parent": nid})
                    node["children"].append(cid)
                    stack.append((cid, sub))
            del p
    for nd in nodes:
        nd.pop("cube", None)
    return nodes, nodeof


def build_tile(
    tile_path: str,
    root: str,
    name: str,
    origin: Sequence[float],
    grid: int,
    leaf_max: int,
    max_depth: int,
) -> Dict[str, Any]:
    """Worker: build one tile's octree files. Returns a summary dict."""
    t0 = time.time()
    rootp = Path(root)
    tiles_dir = rootp / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    xyz, inten = read_laz_xyzi(tile_path)
    t_read = time.time() - t0
    # float32 relative to the island origin keeps sub-mm precision within a few km.
    local = (xyz - np.asarray(origin, dtype=np.float64)).astype(np.float32)
    del xyz
    oct_tmp = tiles_dir / f"{name}.oct.tmp"
    nodes, nodeof = build_octree(local, inten, oct_tmp, grid=grid, leaf_max=leaf_max, max_depth=max_depth)
    np.save(str(tiles_dir / f"{name}.nodeof.npy"), nodeof)
    oct_tmp.replace(tiles_dir / f"{name}.oct")
    meta = {"name": name, "n": int(len(local)), "nodes": nodes}
    tmp = tiles_dir / f"{name}.json.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(tiles_dir / f"{name}.json")
    labels_path = rootp / "labels" / f"{name}.npy"
    if not labels_path.exists():
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(labels_path), np.full(len(local), -1, dtype=np.int32))
    return {"name": name, "n": int(len(local)), "nodes": len(nodes),
            "read_s": round(t_read, 1), "total_s": round(time.time() - t0, 1)}


def _tile_done(root: Path, name: str) -> bool:
    t = root / "tiles"
    return all((t / f"{name}{s}").exists() for s in (".oct", ".json", ".nodeof.npy")) and \
        (root / "labels" / f"{name}.npy").exists()


def build_project(
    src: str | Path | Sequence[str | Path],
    out: str | Path,
    *,
    grid: int = DEFAULT_GRID,
    leaf_max: int = DEFAULT_LEAF_MAX,
    max_depth: int = DEFAULT_MAX_DEPTH,
    workers: int = 2,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Build (or resume) a project from a folder of LAZ/LAS tiles or an explicit list."""
    if isinstance(src, (str, Path)) and Path(src).is_dir():
        files = sorted([p for p in Path(src).iterdir() if p.suffix.lower() in (".laz", ".las")])
        source_dir = str(Path(src).resolve())
    else:
        files = [Path(p) for p in (src if not isinstance(src, (str, Path)) else [src])]
        source_dir = str(files[0].parent.resolve()) if files else ""
    if not files:
        raise FileNotFoundError(f"No .laz/.las files in {src}")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pj_path = root / "project.json"
    old = json.loads(pj_path.read_text(encoding="utf-8")) if pj_path.exists() else None

    headers = [laz_header(p) for p in files]
    gmin = np.min([h["min"] for h in headers], axis=0)
    gmax = np.max([h["max"] for h in headers], axis=0)
    if old and not force:
        origin = old["origin"]
    else:
        origin = [float(np.round((gmin[i] + gmax[i]) / 2.0)) for i in range(3)]
    tiles = []
    for i, (p, h) in enumerate(zip(files, headers)):
        tiles.append({
            "id": i, "name": p.stem, "path": str(p.resolve()), "n": h["n"],
            "bmin": [h["min"][k] - origin[k] for k in range(3)],
            "bmax": [h["max"][k] - origin[k] for k in range(3)],
        })

    todo = [t for t in tiles if force or not _tile_done(root, t["name"])]
    log(f"[build] {len(tiles)} tiles, {sum(t['n'] for t in tiles):,} points; {len(todo)} to build")
    t0 = time.time()
    if todo:
        args = [(t["path"], str(root), t["name"], origin, grid, leaf_max, max_depth) for t in todo]
        if workers <= 1 or len(todo) == 1:
            for k, a in enumerate(args):
                r = build_tile(*a)
                log(f"[build] {k + 1}/{len(todo)} {r['name']}: {r['n']:,} pts, {r['nodes']} nodes, {r['total_s']}s")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(build_tile, *a): a[2] for a in args}
                for k, fut in enumerate(as_completed(futs)):
                    r = fut.result()
                    log(f"[build] {k + 1}/{len(todo)} {r['name']}: {r['n']:,} pts, {r['nodes']} nodes, {r['total_s']}s")

    next_id = 0
    reviewed: List[int] = []
    if old and not force:
        next_id = int(old.get("next_tree_id", 0))
        reviewed = old.get("reviewed", [])
    pj = {
        "version": PROJECT_VERSION,
        "name": root.name,
        "source_dir": source_dir,
        "origin": origin,
        "crs": headers[0]["crs"],
        "bmin": (gmin - np.asarray(origin)).tolist(),
        "bmax": (gmax - np.asarray(origin)).tolist(),
        "tiles": tiles,
        "next_tree_id": next_id,
        "reviewed": reviewed,
        "build": {"grid": grid, "leaf_max": leaf_max, "max_depth": max_depth},
    }
    pj_path.write_text(json.dumps(pj, indent=1), encoding="utf-8")
    build_preview(root)
    labels_max = _max_label(root, tiles)
    if labels_max + 1 > pj["next_tree_id"]:
        pj["next_tree_id"] = labels_max + 1
        pj_path.write_text(json.dumps(pj, indent=1), encoding="utf-8")
    log(f"[build] done in {time.time() - t0:.0f}s -> {root}")
    return root


def _max_label(root: Path, tiles: List[Dict[str, Any]]) -> int:
    m = -1
    for t in tiles:
        lab = np.load(str(root / "labels" / f"{t['name']}.npy"), mmap_mode="r")
        if len(lab):
            m = max(m, int(lab.max()))
    return m


def build_preview(root: Path, budget: int = PREVIEW_POINTS) -> None:
    """Island overview: random subset of every tile's root node (duplicates, labels by ref)."""
    from .store import Project

    proj = Project(root, writable=False)
    roots = [proj.tile_root(t) for t in range(len(proj.tiles))]
    total = sum(int(proj.node_n[r]) for r in roots)
    frac = min(1.0, budget / max(total, 1))
    rng = np.random.default_rng(0)
    parts_xyz, parts_tile, parts_idx = [], [], []
    for t, r in enumerate(roots):
        xyz, _, idx = proj.read_node(r)
        k = int(round(len(idx) * frac))
        sel = rng.choice(len(idx), size=k, replace=False) if k < len(idx) else np.arange(len(idx))
        parts_xyz.append(xyz[sel])
        parts_tile.append(np.full(len(sel), t, dtype=np.uint16))
        parts_idx.append(idx[sel])
    xyz = np.concatenate(parts_xyz).astype(np.float32)
    tile = np.concatenate(parts_tile)
    idx = np.concatenate(parts_idx).astype(np.uint32)
    with open(root / "preview.bin", "wb") as fh:
        fh.write(np.int64(len(idx)).tobytes())
        fh.write(xyz.tobytes())
        fh.write(tile.tobytes())
        fh.write(b"\0" * (-(8 + xyz.nbytes + tile.nbytes) % 4))
        fh.write(idx.tobytes())
    proj.close()


def import_labels(
    root: str | Path,
    labels_dir: str | Path,
    *,
    offset_per_tile: bool = True,
    log: Callable[[str], None] = print,
) -> int:
    """Import per-tile label arrays (<tile>_instances.npy or <tile>.npy).

    With offset_per_tile, each tile's ids are shifted so they are unique island-wide
    (per-tile segmentations reuse 0..k); run `stitch` afterwards to join border trees.
    Returns the number of tiles imported.
    """
    root = Path(root)
    pj = json.loads((root / "project.json").read_text(encoding="utf-8"))
    ldir = Path(labels_dir)
    next_id = 0
    n_done = 0
    for t in pj["tiles"]:
        cand = [ldir / f"{t['name']}_instances.npy", ldir / f"{t['name']}.npy",
                ldir / f"{t['name']}_segmented_instances.npy"]
        src = next((c for c in cand if c.exists()), None)
        dst = root / "labels" / f"{t['name']}.npy"
        cur = np.load(str(dst), mmap_mode="r+")
        if src is None:
            if offset_per_tile:
                cur[:] = -1
            continue
        lab = np.load(str(src)).astype(np.int64).reshape(-1)
        if len(lab) != len(cur):
            log(f"[import] skip {t['name']}: {len(lab):,} labels vs {len(cur):,} points")
            continue
        out = np.full(len(lab), -1, dtype=np.int32)
        pos = lab >= 0
        if offset_per_tile and pos.any():
            uniq, inv = np.unique(lab[pos], return_inverse=True)
            out[pos] = (inv + next_id).astype(np.int32)
            next_id += len(uniq)
        else:
            out[pos] = lab[pos].astype(np.int32)
            next_id = max(next_id, int(lab.max()) + 1)
        cur[:] = out
        cur.flush()
        n_done += 1
        log(f"[import] {t['name']}: {int(pos.sum()):,} labeled pts")
    pj["next_tree_id"] = max(next_id, _max_label(root, pj["tiles"]) + 1)
    pj["reviewed"] = []
    (root / "project.json").write_text(json.dumps(pj, indent=1), encoding="utf-8")
    return n_done


def default_workers() -> int:
    # Largest tiles peak at ~6 GB each; two workers fit comfortably in 32 GB.
    return max(1, min(2, (os.cpu_count() or 2) // 4))
