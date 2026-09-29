"""Server-side annotator session: full cloud, labels, LOD, jobs, autosave."""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..io import _as_path, load_cloud, load_tile_prefix, save_tile
from ..labels import LabelEditor
from ..lod import DEFAULT_MAX_DISPLAY, Lod, build_lod

_SP_DIR = Path(__file__).resolve().parents[2]
WORKDIR = _SP_DIR / "output" / "annotator_workdir"
DEFAULT_EXPORT_DIR = _SP_DIR / "output" / "annotator_exports"
AUTOSAVE_EVERY = 10
INTENSITY_WOOD_LEAF = ("percentile", "intensity", "otsu")


class SessionError(Exception):
    def __init__(self, message: str, status: int = 400, extra: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.extra = extra or {}


class JobFailed(Exception):
    """Expected job failure: shown to the user without a traceback."""


def is_tile_prefix(path: Path) -> bool:
    if Path(str(path) + "_scan.laz").exists() or Path(str(path) + "_scan.las").exists():
        return True
    return path.is_file() and any(
        path.name.endswith(s) for s in ("_scan.laz", "_scan.las", "_instances.npy", "_meta.json")
    )


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.state = "running"
        self.message = ""
        self.detail = ""
        self.progress: Optional[float] = None
        self.started = time.time()
        self.finished: Optional[float] = None
        self.result: Dict[str, Any] = {}

    def stage(self, message: str, progress: Optional[float] = None) -> None:
        self.message = message
        self.detail = ""
        self.progress = progress

    def note(self, text: str) -> None:
        """Third-party output line; a finished sub-bar gives way to an indeterminate one."""
        self.detail = text
        if self.progress is not None and self.progress >= 1.0:
            self.progress = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "message": self.message,
            "detail": self.detail,
            "progress": self.progress,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "result": self.result,
        }


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean_line(text: str) -> str:
    text = _ANSI.sub("", text).strip().strip("#").strip()
    return text[:160]


class _JobStdout:
    """Tee for sys.stdout that mirrors the job thread's last printed line into job.detail.

    Everything except write() is delegated, so libraries calling fileno()/isatty() still work.
    """

    def __init__(self, inner, job: Job, thread_id: int):
        self.inner, self.job, self.thread_id = inner, job, thread_id

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def write(self, s: str) -> int:
        if threading.get_ident() == self.thread_id:
            for line in reversed(re.split(r"[\r\n]", s)):
                line = _clean_line(line)
                if line:
                    self.job.note(line)
                    break
        return self.inner.write(s)

    def flush(self) -> None:
        self.inner.flush()


class _JobLogHandler(logging.Handler):
    def __init__(self, job: Job):
        super().__init__(logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        msg = _clean_line(record.getMessage())
        if msg:
            self.job.note(msg)


@contextmanager
def _capture_progress(job: Job):
    """Route prints, INFO logs and tqdm bars from third-party code into the job."""
    handler = _JobLogHandler(job)
    root = logging.getLogger()
    root.addHandler(handler)
    old_stdout = sys.stdout
    sys.stdout = _JobStdout(old_stdout, job, threading.get_ident())
    patched = None
    try:
        import tqdm.std as tq

        orig_refresh = tq.tqdm.refresh

        def refresh(self, *a, **k):
            if self.total:
                job.progress = min(float(self.n) / float(self.total), 1.0)
                if self.desc:
                    job.detail = _clean_line(self.desc)
            return orig_refresh(self, *a, **k)

        tq.tqdm.refresh = refresh
        patched = (tq, orig_refresh)
    except Exception:
        pass
    try:
        yield
    finally:
        root.removeHandler(handler)
        if sys.stdout is not old_stdout and isinstance(sys.stdout, _JobStdout):
            sys.stdout = old_stdout
        if patched:
            patched[0].tqdm.refresh = patched[1]


class Session:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.loaded = False
        self.xyz: Optional[np.ndarray] = None
        self.intensity: Optional[np.ndarray] = None
        self.editor: Optional[LabelEditor] = None
        self.lod: Optional[Lod] = None
        self.origin = np.zeros(3)
        self.name = ""
        self.meta: Dict[str, Any] = {}
        self.source = ""
        self.source_method = "manual"
        self.reviewed: set[int] = set()
        self.wood_mask: Optional[np.ndarray] = None
        self.leaf_mask: Optional[np.ndarray] = None
        self.wood_leaf_method: Optional[str] = None
        self.max_display = DEFAULT_MAX_DISPLAY
        self._kdtree = None
        self._edits_since_autosave = 0
        self.last_saved: Optional[str] = None
        self.jobs: Dict[str, Job] = {}

    # ── loading ────────────────────────────────────────────────────────────
    def load(self, path: str, mode: str = "gt", max_display: Optional[int] = None) -> Dict[str, Any]:
        """mode: 'gt' (use *_instances.npy if present) or 'blank'."""
        if self.busy():
            raise SessionError("A background job is running; wait for it to finish.", 409)
        p = _as_path(path)
        labels = None
        meta: Dict[str, Any] = {}
        if is_tile_prefix(p):
            tile = load_tile_prefix(p)
            xyz, inten = tile["xyz"], tile["intensity"]
            name, meta = tile["name"], dict(tile["meta"])
            labels = tile["labels"] if mode == "gt" else None
            source = str(tile["prefix"])
        elif p.is_file():
            xyz, inten = load_cloud(p)
            name = p.stem.replace("_scan", "")
            meta = {"source_file": str(p)}
            source = str(p)
        else:
            raise SessionError(f"Not found: {p}", 404)
        return self._set_cloud(xyz, inten, labels, name, meta, source, max_display)

    def load_arrays(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray,
        labels: Optional[np.ndarray],
        name: str,
        meta: Optional[dict] = None,
        source: str = "",
        max_display: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._set_cloud(xyz, intensity, labels, name, meta or {}, source, max_display)

    def _set_cloud(self, xyz, inten, labels, name, meta, source, max_display) -> Dict[str, Any]:
        xyz = np.asarray(xyz, dtype=np.float64)
        inten = np.asarray(inten, dtype=np.float64).reshape(-1)
        n = len(xyz)
        if len(inten) != n:
            raise SessionError(f"intensity length {len(inten)} != points {n}")
        if labels is not None and len(labels) != n:
            raise SessionError(f"labels length {len(labels)} != points {n}")
        if max_display:
            self.max_display = int(max_display)
        lod = build_lod(xyz, self.max_display)
        with self.lock:
            self.xyz = xyz
            self.intensity = inten
            self.editor = LabelEditor(labels) if labels is not None else LabelEditor(n_points=n)
            self.lod = lod
            disp = xyz[lod.display_idx]
            self.origin = disp.mean(axis=0) if len(disp) else np.zeros(3)
            self.name = name
            self.meta = dict(meta or {})
            self.source = source
            self.source_method = "existing_gt" if labels is not None else "manual"
            self.reviewed = set(int(t) for t in self.meta.get("reviewed_trees", []) or [])
            self.wood_mask = self.leaf_mask = None
            self.wood_leaf_method = None
            self._kdtree = None
            self._edits_since_autosave = 0
            self.last_saved = None
            self.loaded = True
        return self.info()

    # ── views ──────────────────────────────────────────────────────────────
    def require(self) -> None:
        if not self.loaded:
            raise SessionError("No cloud loaded.", 409)

    def autosave_path(self, name: Optional[str] = None) -> Path:
        return WORKDIR / f"{name or self.name}.autosave.npy"

    def autosave_info(self) -> Optional[Dict[str, Any]]:
        p = self.autosave_path()
        if not self.loaded or not p.exists():
            return None
        try:
            arr = np.load(str(p), mmap_mode="r")
        except Exception:
            return None
        if len(arr) != len(self.xyz):
            return None
        return {"path": str(p), "mtime": p.stat().st_mtime}

    def info(self) -> Dict[str, Any]:
        if not self.loaded:
            return {"loaded": False, "busy": self.busy()}
        ed = self.editor
        return {
            "loaded": True,
            "name": self.name,
            "source": self.source,
            "source_method": self.source_method,
            "num_points": int(len(self.xyz)),
            "num_display": int(len(self.lod.display_idx)),
            "voxel_size": self.lod.voxel_size,
            "origin": [float(v) for v in self.origin],
            "num_trees": ed.num_trees,
            "next_tree_id": ed.next_tree_id(),
            "dirty": ed.dirty,
            "can_undo": ed.can_undo,
            "can_redo": ed.can_redo,
            "version": ed.version,
            "has_intensity": self.has_intensity(),
            "wood_leaf_method": self.wood_leaf_method,
            "reviewed": sorted(self.reviewed),
            "autosave": self.autosave_info(),
            "last_saved": self.last_saved,
            "default_export_dir": str(DEFAULT_EXPORT_DIR),
            "busy": self.busy(),
        }

    def display_points(self) -> bytes:
        self.require()
        disp = self.xyz[self.lod.display_idx] - self.origin
        return np.ascontiguousarray(disp, dtype=np.float32).tobytes()

    def display_labels(self) -> np.ndarray:
        return self.editor.labels[self.lod.display_idx]

    def display_material(self) -> bytes:
        """uint8 per display point: 0 unknown, 1 wood, 2 leaf."""
        self.require()
        m = np.zeros(len(self.lod.display_idx), dtype=np.uint8)
        if self.wood_mask is not None and self.leaf_mask is not None:
            idx = self.lod.display_idx
            m[self.leaf_mask[idx]] = 2
            m[self.wood_mask[idx]] = 1
        return m.tobytes()

    def trees(self) -> List[Dict[str, Any]]:
        """Per-tree stats: full-res counts, display-based bbox/centroid."""
        self.require()
        lab = self.editor.labels
        ids = self.editor.tree_ids()
        if len(ids) == 0:
            return []
        if ids.max() < 50_000_000:
            counts_all = np.bincount(lab[lab >= 0])
            counts = counts_all[ids]
        else:
            _, counts = np.unique(lab[lab >= 0], return_counts=True)
        dlab = self.display_labels()
        dxyz = self.xyz[self.lod.display_idx]
        keep = dlab >= 0
        pos = np.searchsorted(ids, dlab[keep])
        valid = (pos < len(ids))
        pos = pos[valid]
        pts = dxyz[keep][valid]
        valid_ids = ids[pos] == dlab[keep][valid]
        pos, pts = pos[valid_ids], pts[valid_ids]
        k = len(ids)
        dn = np.bincount(pos, minlength=k).astype(np.float64)
        cx = np.bincount(pos, pts[:, 0], minlength=k)
        cy = np.bincount(pos, pts[:, 1], minlength=k)
        cz = np.bincount(pos, pts[:, 2], minlength=k)
        zmin = np.full(k, np.inf)
        zmax = np.full(k, -np.inf)
        np.minimum.at(zmin, pos, pts[:, 2])
        np.maximum.at(zmax, pos, pts[:, 2])
        out = []
        for i, t in enumerate(ids.tolist()):
            d = max(dn[i], 1.0)
            c = [cx[i] / d, cy[i] / d, cz[i] / d] if dn[i] > 0 else [0.0, 0.0, 0.0]
            out.append(
                {
                    "id": int(t),
                    "count": int(counts[i]),
                    "height": float(zmax[i] - zmin[i]) if dn[i] > 0 else 0.0,
                    "zmin": float(zmin[i] - self.origin[2]) if dn[i] > 0 else 0.0,
                    "center": [float(c[j] - self.origin[j]) for j in range(3)],
                    "reviewed": int(t) in self.reviewed,
                }
            )
        return out

    # ── editing ────────────────────────────────────────────────────────────
    def _edit_result(self, changed_full: np.ndarray, extra: Optional[dict] = None) -> Dict[str, Any]:
        disp = self.lod.display_of_full(changed_full)
        self._edits_since_autosave += 1
        if self._edits_since_autosave >= AUTOSAVE_EVERY:
            self.autosave()
        res = {
            "changed_points": int(len(changed_full)),
            "display_idx": disp,
            "display_labels": self.editor.labels[self.lod.display_idx[disp]] if len(disp) else np.zeros(0, np.int32),
            "info": self.info(),
        }
        if extra:
            res.update(extra)
        return res

    def edit(self, op: str, display_sel: np.ndarray, target: Optional[int] = None,
             source: Optional[int] = None) -> Dict[str, Any]:
        self.require()
        if self.busy():
            raise SessionError("A background job is running; edits are locked.", 409)
        with self.lock:
            ed = self.editor
            extra: Dict[str, Any] = {}
            if op in ("assign", "new", "nontree"):
                mask = self.lod.expand(display_sel)
                if not mask.any():
                    raise SessionError("Selection is empty.")
                if op == "assign":
                    if target is None:
                        raise SessionError("assign needs a target id")
                    ed.reassign(mask, int(target))
                elif op == "new":
                    extra["new_id"] = ed.paint_new(mask)
                else:
                    ed.mark_nontree(mask)
            elif op == "merge":
                if target is None:
                    raise SessionError("merge needs a target id")
                if source is not None:
                    sources = [int(source)]
                else:
                    mask = self.lod.expand(display_sel)
                    sources = [int(s) for s in np.unique(ed.labels[mask]) if s >= 0]
                sources = [s for s in sources if s != int(target)]
                if not sources:
                    raise SessionError("Nothing to merge (select points from other trees).")
                src_mask = np.isin(ed.labels, np.asarray(sources, dtype=np.int32))
                ed.reassign(src_mask, int(target))
                extra["merged"] = sources
                self.reviewed.difference_update(sources)
            elif op == "compact":
                ed.compact()
                self.reviewed.clear()
            else:
                raise SessionError(f"Unknown op {op!r}")
            return self._edit_result(ed.last_changed, extra)

    def undo(self) -> Dict[str, Any]:
        self.require()
        with self.lock:
            if not self.editor.undo():
                raise SessionError("Nothing to undo.")
            return self._edit_result(self.editor.last_changed)

    def redo(self) -> Dict[str, Any]:
        self.require()
        with self.lock:
            if not self.editor.redo():
                raise SessionError("Nothing to redo.")
            return self._edit_result(self.editor.last_changed)

    def set_reviewed(self, tree_id: int, value: bool) -> None:
        self.require()
        with self.lock:
            if value:
                self.reviewed.add(int(tree_id))
            else:
                self.reviewed.discard(int(tree_id))
            self.editor.dirty = True

    def grow(self, display_sel: np.ndarray, radius: float, same_label: bool = True,
             max_iter: int = 200) -> np.ndarray:
        """Connected region growing on display points from a seed selection."""
        self.require()
        from scipy.spatial import cKDTree

        with self.lock:
            if self._kdtree is None:
                self._kdtree = cKDTree(self.xyz[self.lod.display_idx])
            tree = self._kdtree
            m = len(self.lod.display_idx)
            sel = np.zeros(m, dtype=bool)
            ids = np.asarray(display_sel, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < m)]
            sel[ids] = True
            dlab = self.display_labels()
            allowed = None
            if same_label and len(ids):
                allowed = np.isin(dlab, np.unique(dlab[ids]))
            dxyz = self.xyz[self.lod.display_idx]
            frontier = ids
            for _ in range(max_iter):
                if len(frontier) == 0:
                    break
                nb = tree.query_ball_point(dxyz[frontier], r=float(radius))
                cand = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in nb])) if len(nb) else np.zeros(0, np.int64)
                cand = cand[~sel[cand]]
                if allowed is not None:
                    cand = cand[allowed[cand]]
                sel[cand] = True
                frontier = cand
            return np.flatnonzero(sel)

    # ── persistence ────────────────────────────────────────────────────────
    def autosave(self) -> Optional[str]:
        if not self.loaded:
            return None
        WORKDIR.mkdir(parents=True, exist_ok=True)
        p = self.autosave_path()
        tmp = p.with_suffix(".tmp.npy")
        np.save(str(tmp), self.editor.labels)
        tmp.replace(p)
        side = p.with_suffix(".json")
        side.write_text(
            json.dumps({"source": self.source, "reviewed": sorted(self.reviewed), "time": time.time()}),
            encoding="utf-8",
        )
        self._edits_since_autosave = 0
        return str(p)

    def restore_autosave(self) -> Dict[str, Any]:
        self.require()
        p = self.autosave_path()
        if not p.exists():
            raise SessionError("No autosave for this tile.", 404)
        lab = np.load(str(p)).astype(np.int32)
        if len(lab) != len(self.xyz):
            raise SessionError("Autosave length does not match the loaded cloud.")
        side = p.with_suffix(".json")
        with self.lock:
            self.editor.set_all(lab, record_undo=True)
            if side.exists():
                try:
                    self.reviewed = set(json.loads(side.read_text(encoding="utf-8")).get("reviewed", []))
                except Exception:
                    pass
        return self._edit_result(np.arange(len(lab)))

    def discard_autosave(self) -> None:
        for p in (self.autosave_path(), self.autosave_path().with_suffix(".json")):
            if p.exists():
                p.unlink()

    def save(self, out_dir: str, name: str, overwrite: bool = False) -> Dict[str, str]:
        self.require()
        name = (name or self.name or "tile").strip()
        if not name or any(c in name for c in '\\/:*?"<>|'):
            raise SessionError(f"Invalid tile name {name!r}")
        out = Path(out_dir or DEFAULT_EXPORT_DIR).expanduser()
        if not out.is_absolute():
            out = (_SP_DIR / out).resolve()
        existing = [
            str(out / f"{name}{suf}")
            for suf in ("_scan.laz", "_instances.npy", "_meta.json")
            if (out / f"{name}{suf}").exists()
        ]
        if existing and not overwrite:
            raise SessionError("Files already exist.", 409, {"existing": existing})
        with self.lock:
            meta = dict(self.meta)
            meta["source_method"] = self.source_method
            meta["source"] = self.source
            meta["reviewed_trees"] = sorted(self.reviewed)
            paths = save_tile(out, name, self.xyz, self.intensity, self.editor.labels, meta=meta)
            self.editor.dirty = False
            self.last_saved = time.strftime("%Y-%m-%d %H:%M:%S")
            self.discard_autosave()
        return paths

    # ── background jobs ────────────────────────────────────────────────────
    def busy(self) -> bool:
        return any(j.state == "running" for j in self.jobs.values())

    def start_job(self, kind: str, fn: Callable[[Job], None]) -> Job:
        self.require()
        if self.busy():
            raise SessionError("Another job is already running.", 409)
        job = Job(kind)
        self.jobs[job.id] = job

        def _run() -> None:
            try:
                with _capture_progress(job):
                    fn(job)
                job.progress = 1.0
                job.state = "done"
            except JobFailed as exc:
                job.state = "error"
                job.message = str(exc)
            except Exception as exc:  # surfaced to the UI
                job.state = "error"
                job.message = f"{type(exc).__name__}: {exc}"
                job.result = {"traceback": traceback.format_exc()[-4000:]}
            finally:
                job.finished = time.time()

        threading.Thread(target=_run, daemon=True).start()
        return job

    def run_segmentation(self, method: str, leaf_removal: bool, treex_stock: bool) -> Job:
        from ..segment import run_method

        def work(job: Job) -> None:
            job.stage(f"Running {method} on {len(self.xyz):,} points")
            lab, info = run_method(
                self.xyz, self.intensity, method,
                leaf_removal=leaf_removal, treex_stock_tls=treex_stock,
                results_folder=str(WORKDIR),
            )
            job.result = {"info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}}
            if not info.get("ok") or info.get("num_trees", 0) == 0:
                reason = info.get("message") or "no trees found"
                if info.get("ok"):
                    reason = "no trees found"
                hint = "" if self.has_intensity() else " This cloud has no intensity variation, which some methods rely on."
                raise JobFailed(f"{method}: {reason}. Labels were not changed.{hint}")
            job.stage("Applying labels", 1.0)
            with self.lock:
                self.editor.set_all(lab, record_undo=True)
                self.source_method = method
                self.reviewed.clear()
            job.message = info["message"]

        return self.start_job("segment", work)

    def has_intensity(self) -> bool:
        return bool(len(self.intensity) and np.ptp(self.intensity) > 0)

    def run_wood_leaf(self, method: str, params: Dict[str, Any]) -> Job:
        from ..wood_leaf import classify_wood_leaf, mask_counts

        self.require()
        if method in INTENSITY_WOOD_LEAF and not self.has_intensity():
            raise SessionError(
                f"'{method}' classifies by intensity, but this cloud's intensity is constant. "
                "Use stem_grow or eigen instead.", 400)

        def work(job: Job) -> None:
            job.stage(f"Classifying {len(self.xyz):,} points ({method})")
            wood, leaf = classify_wood_leaf(method, self.xyz, self.intensity, **params)
            job.stage("Storing masks", 1.0)
            with self.lock:
                self.wood_mask, self.leaf_mask = wood, leaf
                self.wood_leaf_method = method
            job.result = {"counts": mask_counts(wood, leaf)}
            c = job.result["counts"]
            job.message = f"wood={c['wood']:,} leaf={c['leaf']:,} unknown={c['unknown']:,}"

        return self.start_job("woodleaf", work)
