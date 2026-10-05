"""Read/write access to a built island project (see build.py for the layout)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PREVIEW_NODE = 0


class ProjectError(Exception):
    def __init__(self, message: str, status: int = 400, extra: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.extra = extra or {}


def is_project_dir(path: str | Path) -> bool:
    p = Path(path)
    if p.is_file() and p.name == "project.json":
        p = p.parent
    return (p / "project.json").exists() and (p / "tiles").is_dir()


class Project:
    """Octree + labels store. Node 0 is the island preview; tile nodes follow."""

    def __init__(self, root: str | Path, *, writable: bool = True):
        root = Path(root)
        if root.is_file():
            root = root.parent
        if not is_project_dir(root):
            raise ProjectError(f"Not a project folder: {root}", 404)
        self.root = root.resolve()
        self.writable = writable
        self.meta: Dict[str, Any] = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        self.tiles: List[Dict[str, Any]] = self.meta["tiles"]
        self.origin = np.asarray(self.meta["origin"], dtype=np.float64)
        self._meta_lock = threading.Lock()
        self._oct: Dict[int, np.memmap] = {}
        self._labels: Dict[int, np.ndarray] = {}
        self._nodeof: Dict[int, np.ndarray] = {}
        self._load_nodes()

    # ── node table ─────────────────────────────────────────────────────────
    def _load_nodes(self) -> None:
        tile_nodes = []
        for t in self.tiles:
            tj = json.loads((self.root / "tiles" / f"{t['name']}.json").read_text(encoding="utf-8"))
            tile_nodes.append(tj["nodes"])
        total = 1 + sum(len(ns) for ns in tile_nodes)
        self.node_tile = np.full(total, -1, dtype=np.int32)
        self.node_local = np.zeros(total, dtype=np.int32)
        self.node_level = np.zeros(total, dtype=np.int16)
        self.node_n = np.zeros(total, dtype=np.int64)
        self.node_offset = np.zeros(total, dtype=np.int64)
        self.node_bmin = np.zeros((total, 3), dtype=np.float64)
        self.node_bmax = np.zeros((total, 3), dtype=np.float64)
        self.node_qmin = np.zeros((total, 3), dtype=np.float64)
        self.node_qext = np.ones((total, 3), dtype=np.float64)
        self.node_spacing = np.zeros(total, dtype=np.float64)
        self.node_parent = np.full(total, -1, dtype=np.int64)
        self.node_children: List[List[int]] = [[] for _ in range(total)]
        self.tile_base = np.zeros(len(self.tiles), dtype=np.int64)
        gid = 1
        for t, nodes in enumerate(tile_nodes):
            self.tile_base[t] = gid
            for li, nd in enumerate(nodes):
                g = gid + li
                self.node_tile[g] = t
                self.node_local[g] = li
                self.node_level[g] = nd["level"]
                self.node_n[g] = nd["n"]
                self.node_offset[g] = nd["offset"]
                self.node_bmin[g] = nd["bmin"]
                self.node_bmax[g] = nd["bmax"]
                self.node_qmin[g] = nd["qmin"]
                self.node_qext[g] = nd["qext"]
                self.node_spacing[g] = nd["spacing"]
                self.node_parent[g] = gid + nd["parent"] if "parent" in nd else PREVIEW_NODE
                self.node_children[g] = [gid + c for c in nd["children"]]
            gid += len(nodes)
        roots = [int(self.tile_base[t]) for t in range(len(self.tiles)) if self.tiles[t]["n"] > 0]
        self.node_children[PREVIEW_NODE] = roots
        self.node_level[PREVIEW_NODE] = -1
        if roots:
            self.node_bmin[PREVIEW_NODE] = self.node_bmin[roots].min(axis=0)
            self.node_bmax[PREVIEW_NODE] = self.node_bmax[roots].max(axis=0)
        self._preview = None
        pv = self.root / "preview.bin"
        if pv.exists():
            self.node_n[PREVIEW_NODE] = int(np.fromfile(str(pv), dtype=np.int64, count=1)[0])

    def tile_root(self, tile: int) -> int:
        return int(self.tile_base[tile])

    @property
    def num_nodes(self) -> int:
        return len(self.node_n)

    @property
    def num_points(self) -> int:
        return int(sum(t["n"] for t in self.tiles))

    def hierarchy(self) -> Dict[str, Any]:
        """Compact node table for the client: flat arrays, children as CSR."""
        child_counts = [len(c) for c in self.node_children]
        return {
            "tile": self.node_tile.tolist(),
            "level": self.node_level.tolist(),
            "n": self.node_n.tolist(),
            "bmin": np.round(self.node_bmin, 3).ravel().tolist(),
            "bmax": np.round(self.node_bmax, 3).ravel().tolist(),
            "spacing": np.round(self.node_spacing, 4).tolist(),
            "child_start": np.concatenate([[0], np.cumsum(child_counts)]).astype(int).tolist(),
            "children": [c for cs in self.node_children for c in cs],
        }

    # ── point data ─────────────────────────────────────────────────────────
    def _oct_map(self, tile: int) -> np.memmap:
        m = self._oct.get(tile)
        if m is None:
            p = self.root / "tiles" / f"{self.tiles[tile]['name']}.oct"
            m = np.memmap(str(p), dtype=np.uint8, mode="r") if p.stat().st_size else np.zeros(0, np.uint8)
            self._oct[tile] = m
        return m

    def _preview_data(self):
        if self._preview is None:
            raw = np.fromfile(str(self.root / "preview.bin"), dtype=np.uint8)
            n = int(raw[:8].view(np.int64)[0])
            o = 8
            xyz = raw[o:o + 12 * n].view(np.float32).reshape(n, 3)
            o += 12 * n
            tile = raw[o:o + 2 * n].view(np.uint16)
            o += 2 * n
            o += -o % 4
            idx = raw[o:o + 4 * n].view(np.uint32)
            self._preview = (xyz, tile.astype(np.int64), idx)
        return self._preview

    def read_node(self, gid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xyz float32 local (n,3), intensity uint8 (n,), source idx uint32 (n,)) for a tile node."""
        gid = int(gid)
        if gid == PREVIEW_NODE:
            xyz, _, idx = self._preview_data()
            return xyz, np.zeros(len(idx), np.uint8), idx
        t = int(self.node_tile[gid])
        n = int(self.node_n[gid])
        o = int(self.node_offset[gid])
        buf = self._oct_map(t)
        idx = np.frombuffer(buf, dtype=np.uint32, count=n, offset=o)
        q = np.frombuffer(buf, dtype=np.uint16, count=3 * n, offset=o + 4 * n).reshape(n, 3)
        inten = np.frombuffer(buf, dtype=np.uint8, count=n, offset=o + 10 * n)
        xyz = (q.astype(np.float32) * (self.node_qext[gid] / 65535.0).astype(np.float32)
               + self.node_qmin[gid].astype(np.float32))
        return xyz, inten, idx

    def node_refs(self, gid: int) -> Tuple[np.ndarray, np.ndarray]:
        """(tile int64 (n,), idx uint32 (n,)) source references for a node."""
        gid = int(gid)
        if gid == PREVIEW_NODE:
            _, tile, idx = self._preview_data()
            return tile, idx
        t = int(self.node_tile[gid])
        n = int(self.node_n[gid])
        idx = np.frombuffer(self._oct_map(t), dtype=np.uint32, count=n, offset=int(self.node_offset[gid]))
        return np.full(n, t, dtype=np.int64), idx

    # ── labels ─────────────────────────────────────────────────────────────
    def labels(self, tile: int) -> np.ndarray:
        lab = self._labels.get(tile)
        if lab is None:
            p = self.root / "labels" / f"{self.tiles[tile]['name']}.npy"
            lab = np.load(str(p), mmap_mode="r+" if self.writable else "r")
            self._labels[tile] = lab
        return lab

    def node_labels(self, gid: int) -> np.ndarray:
        gid = int(gid)
        if gid == PREVIEW_NODE:
            _, tile, idx = self._preview_data()
            out = np.empty(len(idx), dtype=np.int32)
            for t in np.unique(tile):
                m = tile == t
                out[m] = self.labels(int(t))[idx[m]]
            return out
        t = int(self.node_tile[gid])
        _, idx = self.node_refs(gid)
        return np.asarray(self.labels(t)[idx], dtype=np.int32)

    def gather_labels(self, tile: np.ndarray, idx: np.ndarray) -> np.ndarray:
        out = np.empty(len(idx), dtype=np.int32)
        for t in np.unique(tile):
            m = tile == t
            out[m] = self.labels(int(t))[idx[m]]
        return out

    def write_labels(self, tile: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
        if not self.writable:
            raise ProjectError("Project opened read-only.")
        values = np.asarray(values, dtype=np.int32)
        for t in np.unique(tile):
            m = tile == t
            lab = self.labels(int(t))
            lab[idx[m]] = values[m]

    def flush(self) -> None:
        for lab in self._labels.values():
            if isinstance(lab, np.memmap):
                lab.flush()

    def nodeof(self, tile: int) -> np.ndarray:
        m = self._nodeof.get(tile)
        if m is None:
            m = np.load(str(self.root / "tiles" / f"{self.tiles[tile]['name']}.nodeof.npy"), mmap_mode="r")
            self._nodeof[tile] = m
        return m

    def affected_nodes(self, tile: np.ndarray, idx: np.ndarray, cap: int = 20000) -> Optional[List[int]]:
        """Global node ids containing the given points; None when there are too many."""
        out: List[np.ndarray] = []
        total = 0
        for t in np.unique(tile):
            m = tile == t
            loc = np.unique(np.asarray(self.nodeof(int(t))[idx[m]]))
            out.append(loc.astype(np.int64) + self.tile_base[int(t)])
            total += len(loc)
            if total > cap:
                return None
        res = np.concatenate(out).tolist() if out else []
        return res + [PREVIEW_NODE]

    # ── spatial queries ────────────────────────────────────────────────────
    def nodes_intersecting(self, bmin: Sequence[float], bmax: Sequence[float],
                           tiles: Optional[Iterable[int]] = None) -> np.ndarray:
        """Tile nodes whose bounds intersect the box. bmin/bmax may be 2D (xy) or 3D."""
        k = len(bmin)
        lo = np.asarray(bmin, dtype=np.float64)
        hi = np.asarray(bmax, dtype=np.float64)
        ok = np.all(self.node_bmax[:, :k] >= lo, axis=1) & np.all(self.node_bmin[:, :k] <= hi, axis=1)
        ok &= self.node_tile >= 0
        ok &= self.node_n > 0
        if tiles is not None:
            ok &= np.isin(self.node_tile, np.asarray(list(tiles)))
        return np.flatnonzero(ok)

    def count_upper_bound(self, bmin, bmax) -> int:
        return int(self.node_n[self.nodes_intersecting(bmin, bmax)].sum())

    def crop(self, bmin: Sequence[float], bmax: Sequence[float], *, max_points: Optional[int] = None,
             tiles: Optional[Iterable[int]] = None) -> Dict[str, np.ndarray]:
        """All full-resolution points inside the box (xy or xyz, project-local coordinates)."""
        k = len(bmin)
        lo = np.asarray(bmin, dtype=np.float32)
        hi = np.asarray(bmax, dtype=np.float32)
        nodes = self.nodes_intersecting(bmin, bmax, tiles)
        xs, ins, ts, ids = [], [], [], []
        total = 0
        for g in nodes:
            xyz, inten, idx = self.read_node(int(g))
            inside = np.all((xyz[:, :k] >= lo) & (xyz[:, :k] <= hi), axis=1)
            if not inside.any():
                continue
            xs.append(xyz[inside])
            ins.append(inten[inside])
            ids.append(idx[inside])
            ts.append(np.full(int(inside.sum()), self.node_tile[g], dtype=np.int16))
            total += int(inside.sum())
            if max_points is not None and total > max_points:
                raise ProjectError(
                    f"Region has more than {max_points:,} points; draw a smaller box.", 413,
                    {"max_points": max_points})
        if not xs:
            return {"xyz": np.zeros((0, 3), np.float32), "intensity": np.zeros(0, np.uint8),
                    "tile": np.zeros(0, np.int16), "idx": np.zeros(0, np.uint32)}
        return {"xyz": np.concatenate(xs), "intensity": np.concatenate(ins),
                "tile": np.concatenate(ts), "idx": np.concatenate(ids)}

    # ── tree ids / metadata ────────────────────────────────────────────────
    def save_meta(self) -> None:
        with self._meta_lock:
            p = self.root / "project.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
            tmp.replace(p)

    def peek_next_id(self) -> int:
        return int(self.meta.get("next_tree_id", 0))

    def allocate_ids(self, k: int = 1) -> int:
        """Reserve k consecutive island-wide tree ids; returns the first."""
        with self._meta_lock:
            first = int(self.meta.get("next_tree_id", 0))
            self.meta["next_tree_id"] = first + int(k)
        self.save_meta()
        return first

    def bump_next_id(self, used_max: int) -> None:
        if used_max + 1 > self.peek_next_id():
            with self._meta_lock:
                self.meta["next_tree_id"] = int(used_max) + 1
            self.save_meta()

    def tile_bounds_xy(self, tile: int) -> Tuple[np.ndarray, np.ndarray]:
        t = self.tiles[tile]
        return np.asarray(t["bmin"][:2]), np.asarray(t["bmax"][:2])

    def owner_tile(self, xy: np.ndarray) -> np.ndarray:
        """Tile owning each xy position: the tile box containing it, else the nearest box.

        Boxes are treated half-open so a point on a shared edge has exactly one owner.
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        lo = np.array([t["bmin"][:2] for t in self.tiles])
        hi = np.array([t["bmax"][:2] for t in self.tiles])
        d = np.maximum(np.maximum(lo[None] - xy[:, None], xy[:, None] - hi[None]), 0.0)
        dist = np.hypot(d[..., 0], d[..., 1])
        on_hi_edge = np.any(xy[:, None] >= hi[None], axis=2) & (dist == 0)
        dist = dist + on_hi_edge * 1e-9
        return np.argmin(dist, axis=1)

    def close(self) -> None:
        self.flush()
        self._oct.clear()
        self._labels.clear()
        self._nodeof.clear()
