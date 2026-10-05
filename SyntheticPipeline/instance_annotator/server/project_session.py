"""Island project session: streaming data, island-wide tree stats and tree-level edits.

Region editing reuses the single-cloud `Session`; a `RegionLink` writes its label changes
straight through to the project's per-tile label files.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..project.stats import SAMPLE_POINTS, build_sample, island_trees, label_counts
from ..project.store import PREVIEW_NODE, Project, ProjectError
from .session import Job, start_background_job

UNDO_BYTES = 1024 * 1024 * 1024
REGION_MAX_POINTS = 25_000_000


@dataclass
class Chunk:
    tile: int
    idx: np.ndarray  # int64
    old: np.ndarray  # int32
    new: np.ndarray  # int32

    @property
    def nbytes(self) -> int:
        return int(self.idx.nbytes + self.old.nbytes + self.new.nbytes)


class RegionLink:
    """Back-references from a region cloud to (tile, idx) with write-through."""

    def __init__(self, ps: "ProjectSession", tile: np.ndarray, idx: np.ndarray, bmin, bmax):
        self.ps = ps
        self.tile = tile.astype(np.int64)
        self.idx = idx.astype(np.int64)
        self.bmin = [float(v) for v in bmin]
        self.bmax = [float(v) for v in bmax]

    def write(self, changed: np.ndarray, values: np.ndarray) -> None:
        if len(changed) == 0:
            return
        t = self.tile[changed]
        i = self.idx[changed]
        chunks = []
        with self.ps.lock:
            proj = self.ps.project
            for tt in np.unique(t):
                m = t == tt
                lab = proj.labels(int(tt))
                old = np.asarray(lab[i[m]], dtype=np.int32)
                new = np.asarray(values[m], dtype=np.int32)
                lab[i[m]] = new
                chunks.append(Chunk(int(tt), i[m], old, new))
            proj.flush()
            self.ps._account(chunks)
            if len(values):
                proj.bump_next_id(int(np.max(values)))

    def merge_outside(self, sources: Sequence[int], target: int) -> List[Chunk]:
        """Relabel points of `sources` outside the region too (whole-tree merge)."""
        with self.ps.lock:
            return self.ps._relabel(sources, target)

    def apply(self, chunks: List[Chunk], reverse: bool) -> None:
        with self.ps.lock:
            self.ps._apply(chunks, reverse)

    def allocate_id(self) -> int:
        return self.ps.project.allocate_ids(1)

    def allocate_ids(self, k: int) -> int:
        return self.ps.project.allocate_ids(k)

    def peek_next_id(self) -> int:
        return self.ps.project.peek_next_id()

    def describe(self) -> Dict[str, Any]:
        return {"project": self.ps.project.meta.get("name", ""), "bmin": self.bmin, "bmax": self.bmax}


class ProjectSession:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.project: Optional[Project] = None
        self.jobs: Dict[str, Job] = {}
        self.version = 0
        self._undo: List[List[Chunk]] = []
        self._redo: List[List[Chunk]] = []
        self._counts: Dict[int, np.ndarray] = {}
        self._stats_state = "idle"
        self._sample = None
        self._trees_cache: Optional[List[Dict[str, Any]]] = None
        self._trees_version = -1
        self.region: Optional[RegionLink] = None
        self.candidates: List[Dict[str, Any]] = []

    # ── lifecycle ──────────────────────────────────────────────────────────
    def open(self, path: str) -> Dict[str, Any]:
        if self.busy():
            raise ProjectError("A background job is running.", 409)
        proj = Project(path)
        with self.lock:
            if self.project is not None:
                self.project.close()
            self.project = proj
            self.version = 0
            self._undo.clear()
            self._redo.clear()
            self._counts = {}
            self._sample = None
            self._trees_cache = None
            self.region = None
            self.candidates = self._load_candidates()
            self._stats_state = "computing"
        threading.Thread(target=self._compute_stats_base, daemon=True).start()
        return self.info()

    def close(self) -> None:
        with self.lock:
            if self.project is not None:
                self.project.close()
            self.project = None
            self.region = None

    def require(self) -> Project:
        if self.project is None:
            raise ProjectError("No project open.", 409)
        return self.project

    def busy(self) -> bool:
        return any(j.state == "running" for j in self.jobs.values())

    def info(self) -> Dict[str, Any]:
        p = self.project
        if p is None:
            return {"open": False}
        return {
            "open": True,
            "name": p.meta.get("name", p.root.name),
            "root": str(p.root),
            "origin": p.origin.tolist(),
            "bmin": p.meta["bmin"],
            "bmax": p.meta["bmax"],
            "num_points": p.num_points,
            "num_nodes": p.num_nodes,
            "tiles": [{"id": t["id"], "name": t["name"], "n": t["n"], "bmin": t["bmin"], "bmax": t["bmax"]}
                      for t in p.tiles],
            "next_tree_id": p.peek_next_id(),
            "version": self.version,
            "can_undo": bool(self._undo),
            "can_redo": bool(self._redo),
            "stats": self._stats_state,
            "region": self.region.describe() if self.region else None,
            "busy": self.busy(),
            "num_candidates": len(self.candidates),
            "crs": (p.meta.get("crs") or "")[:120],
        }

    # ── streaming ──────────────────────────────────────────────────────────
    def node_payload(self, gid: int) -> bytes:
        p = self.require()
        if gid < 0 or gid >= p.num_nodes:
            raise ProjectError("unknown node", 404)
        xyz, _, _ = p.read_node(gid)
        lab = p.node_labels(gid)
        return np.ascontiguousarray(xyz, dtype="<f4").tobytes() + np.ascontiguousarray(lab, dtype="<i4").tobytes()

    def node_labels(self, gid: int) -> bytes:
        p = self.require()
        if gid < 0 or gid >= p.num_nodes:
            raise ProjectError("unknown node", 404)
        return np.ascontiguousarray(p.node_labels(gid), dtype="<i4").tobytes()

    # ── stats ──────────────────────────────────────────────────────────────
    def _compute_stats_base(self) -> None:
        proj = self.project
        try:
            counts = label_counts(proj)
            sample = build_sample(proj, SAMPLE_POINTS)
            with self.lock:
                if self.project is proj:
                    self._counts = counts
                    self._sample = sample
                    self._trees_cache = None
                    self._stats_state = "ready"
        except Exception as exc:  # pragma: no cover - surfaced in info()
            self._stats_state = f"error: {exc}"

    def _account(self, chunks: List[Chunk]) -> None:
        """Update per-tile counts for applied chunks and bump the version."""
        for c in chunks:
            cnt = self._counts.get(c.tile)
            if cnt is not None and self._stats_state == "ready":
                mx = int(max(c.new.max(initial=-1), c.old.max(initial=-1)))
                if mx >= len(cnt):
                    cnt = np.concatenate([cnt, np.zeros(mx + 1 - len(cnt), np.int64)])
                o = c.old[c.old >= 0]
                n = c.new[c.new >= 0]
                if len(o):
                    cnt[: o.max() + 1] -= np.bincount(o)
                if len(n):
                    cnt[: n.max() + 1] += np.bincount(n)
                self._counts[c.tile] = cnt
        self.version += 1

    def trees(self) -> Dict[str, Any]:
        p = self.require()
        if self._stats_state != "ready":
            return {"ready": False, "state": self._stats_state, "trees": []}
        with self.lock:
            if self._trees_cache is not None and self._trees_version == self.version:
                return {"ready": True, "trees": self._trees_cache}
            out = island_trees(p, self._counts, self._sample)
            self._trees_cache = out
            self._trees_version = self.version
            return {"ready": True, "trees": out}

    # ── island edits ───────────────────────────────────────────────────────
    def _relabel(self, sources: Sequence[int], target: int) -> List[Chunk]:
        p = self.require()
        src = np.asarray(sorted(set(int(s) for s in sources if int(s) != int(target))), dtype=np.int32)
        if len(src) == 0:
            return []
        tiles = range(len(p.tiles))
        if self._stats_state == "ready":
            tiles = [t for t, c in self._counts.items()
                     if any(s < len(c) and c[s] > 0 for s in src.tolist())]
        chunks = []
        for t in tiles:
            lab = p.labels(t)
            idx = np.flatnonzero(np.isin(np.asarray(lab), src))
            if len(idx) == 0:
                continue
            old = np.asarray(lab[idx], dtype=np.int32)
            new = np.full(len(idx), int(target), dtype=np.int32)
            lab[idx] = new
            chunks.append(Chunk(int(t), idx.astype(np.int64), old, new))
        p.flush()
        self._account(chunks)
        return chunks

    def _apply(self, chunks: List[Chunk], reverse: bool) -> None:
        p = self.require()
        applied = []
        for c in chunks:
            vals = c.old if reverse else c.new
            p.labels(c.tile)[c.idx] = vals
            applied.append(Chunk(c.tile, c.idx, c.new, c.old) if reverse else c)
        p.flush()
        self._account(applied)

    def _check_editable(self) -> None:
        self.require()
        if self.region is not None:
            raise ProjectError("Close the edit region before island-wide edits.", 409)
        if self.busy():
            raise ProjectError("A background job is running.", 409)

    def _result(self, chunks: List[Chunk], extra: Optional[dict] = None) -> Dict[str, Any]:
        p = self.project
        n = int(sum(len(c.idx) for c in chunks))
        affected: Optional[List[int]] = []
        if chunks:
            tile = np.concatenate([np.full(len(c.idx), c.tile, np.int64) for c in chunks])
            idx = np.concatenate([c.idx for c in chunks])
            affected = p.affected_nodes(tile, idx)
        res = {"changed_points": n, "affected_nodes": affected, "info": self.info()}
        if extra:
            res.update(extra)
        return res

    def _push(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self._undo.append(chunks)
        self._redo.clear()
        total = sum(c.nbytes for step in self._undo for c in step)
        while len(self._undo) > 1 and total > UNDO_BYTES:
            total -= sum(c.nbytes for c in self._undo.pop(0))

    def merge(self, sources: Sequence[int], target: int) -> Dict[str, Any]:
        self._check_editable()
        if int(target) < -1:
            raise ProjectError("target must be >= -1")
        with self.lock:
            chunks = self._relabel(sources, int(target))
            if not chunks:
                raise ProjectError("Nothing to merge.")
            self._push(chunks)
            rev = set(int(v) for v in self.project.meta.get("reviewed", []))
            rev.difference_update(int(s) for s in sources)
            self.project.meta["reviewed"] = sorted(rev)
            if int(target) >= 0:
                self.project.bump_next_id(int(target))
            self.project.save_meta()
            return self._result(chunks, {"merged": sorted(set(int(s) for s in sources) - {int(target)})})

    def undo(self) -> Dict[str, Any]:
        self._check_editable()
        with self.lock:
            if not self._undo:
                raise ProjectError("Nothing to undo.")
            step = self._undo.pop()
            self._apply(step, reverse=True)
            self._redo.append(step)
            return self._result(step)

    def redo(self) -> Dict[str, Any]:
        self._check_editable()
        with self.lock:
            if not self._redo:
                raise ProjectError("Nothing to redo.")
            step = self._redo.pop()
            self._apply(step, reverse=False)
            self._undo.append(step)
            return self._result(step)

    def set_reviewed(self, tree_id: int, value: bool) -> Dict[str, Any]:
        p = self.require()
        rev = set(int(v) for v in p.meta.get("reviewed", []))
        (rev.add if value else rev.discard)(int(tree_id))
        p.meta["reviewed"] = sorted(rev)
        p.save_meta()
        with self.lock:
            if self._trees_cache is not None:
                for t in self._trees_cache:
                    if t["id"] == int(tree_id):
                        t["reviewed"] = bool(value)
        return self.info()

    # ── regions ────────────────────────────────────────────────────────────
    def open_region(self, bmin: Sequence[float], bmax: Sequence[float],
                    max_points: int = REGION_MAX_POINTS) -> Dict[str, Any]:
        p = self.require()
        if self.busy():
            raise ProjectError("A background job is running.", 409)
        crop = p.crop(bmin, bmax, max_points=max_points)
        if len(crop["idx"]) == 0:
            raise ProjectError("No points inside that box.")
        tile = crop["tile"].astype(np.int64)
        labels = p.gather_labels(tile, crop["idx"])
        link = RegionLink(self, tile, crop["idx"], bmin, bmax)
        xyz = crop["xyz"].astype(np.float64) + p.origin
        inten = crop["intensity"].astype(np.float64) / 255.0
        return {"xyz": xyz, "intensity": inten, "labels": labels, "link": link}

    def attach_region(self, link: RegionLink) -> None:
        with self.lock:
            self.region = link
            self._undo.clear()
            self._redo.clear()

    def close_region(self) -> None:
        with self.lock:
            self.region = None

    # ── candidates (stitch review queue) ───────────────────────────────────
    def _load_candidates(self) -> List[Dict[str, Any]]:
        from ..project.stitch import load_candidates

        return [c for c in load_candidates(self.project) if c.get("status", "open") == "open"]

    def resolve_candidate(self, key: str, accept: bool) -> Dict[str, Any]:
        import json

        from ..project.stitch import load_candidates

        self._check_editable()
        cand = next((c for c in self.candidates if c["key"] == key), None)
        if cand is None:
            raise ProjectError("unknown candidate", 404)
        res: Dict[str, Any] = {"info": self.info(), "changed_points": 0, "affected_nodes": []}
        if accept:
            res = self.merge([cand["b"]], cand["a"])
        allc = load_candidates(self.project)
        for c in allc:
            if c["key"] == key:
                c["status"] = "accepted" if accept else "rejected"
            elif accept and c.get("status", "open") == "open":
                for k in ("a", "b"):
                    if c[k] == cand["b"]:
                        c[k] = cand["a"]
                if c["a"] == c["b"]:
                    c["status"] = "accepted"
        (self.project.root / "stitch_candidates.json").write_text(json.dumps(allc, indent=1), encoding="utf-8")
        self.candidates = [c for c in allc if c.get("status", "open") == "open"]
        res["info"] = self.info()
        return res

    # ── jobs ───────────────────────────────────────────────────────────────
    def start_job(self, kind: str, fn) -> Job:
        self.require()
        if self.busy():
            raise ProjectError("Another job is already running.", 409)
        if self.region is not None:
            raise ProjectError("Close the edit region first.", 409)
        return start_background_job(self.jobs, kind, fn)

    def after_bulk_change(self) -> None:
        """Labels were rewritten outside the undo system (segmentation, import)."""
        with self.lock:
            self._undo.clear()
            self._redo.clear()
            self.version += 1
            self._stats_state = "computing"
            self.candidates = self._load_candidates()
        self._compute_stats_base()
