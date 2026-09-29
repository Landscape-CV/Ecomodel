"""REST + binary API for the annotator web UI."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..io import _as_path, load_cloud
from .session import INTENSITY_WOOD_LEAF, Session, SessionError, is_tile_prefix

# Above this many changed display points the client refetches /api/labels.
INLINE_DIFF_MAX = 250_000

WOOD_LEAF_PARAM_KEYS = {
    "threshold", "percentile", "linearity_min", "verticality_min",
    "curvature_max", "height_percentile", "grow_radius",
}

router = APIRouter(prefix="/api")
session = Session()


def _err(exc: SessionError) -> JSONResponse:
    return JSONResponse({"error": str(exc), **exc.extra}, status_code=exc.status)


def _edit_payload(res: Dict[str, Any]) -> Dict[str, Any]:
    idx = res.pop("display_idx")
    lab = res.pop("display_labels")
    if len(idx) > INLINE_DIFF_MAX:
        res["reload_labels"] = True
    else:
        res["display_idx"] = idx.tolist()
        res["display_labels"] = np.asarray(lab).astype(int).tolist()
    return res


async def _read_indices(request: Request) -> np.ndarray:
    body = await request.body()
    if len(body) % 4:
        raise SessionError("Index payload must be uint32 little-endian.")
    return np.frombuffer(body, dtype="<u4").astype(np.int64)


class LoadReq(BaseModel):
    path: str
    mode: str = "gt"
    max_display: Optional[int] = None


class SaveReq(BaseModel):
    out_dir: str = ""
    name: str = ""
    overwrite: bool = False


class SegmentReq(BaseModel):
    method: str
    leaf_removal: bool = False
    treex_stock: bool = True


class WoodLeafReq(BaseModel):
    method: str
    params: Dict[str, Any] = {}


class ReviewReq(BaseModel):
    tree_id: int
    reviewed: bool


@router.get("/status")
def status():
    return session.info()


@router.get("/browse")
def browse(dir: str = ""):
    """List sub-directories and loadable clouds / tile prefixes."""
    base = _as_path(dir) if dir else _as_path("testdataset")
    if base.is_file():
        base = base.parent
    if not base.exists():
        return JSONResponse({"error": f"Not found: {base}"}, status_code=404)
    dirs: List[str] = []
    tiles: Dict[str, Dict[str, Any]] = {}
    files: List[Dict[str, Any]] = []
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        return JSONResponse({"error": f"Permission denied: {base}"}, status_code=403)
    for p in entries:
        if p.is_dir():
            dirs.append(p.name)
            continue
        nm = p.name
        low = nm.lower()
        for suf in ("_scan.laz", "_scan.las"):
            if low.endswith(suf):
                pre = nm[: -len(suf)]
                has_gt = (base / f"{pre}_instances.npy").exists()
                tiles[pre] = {"name": pre, "path": str(base / pre), "has_gt": has_gt,
                              "size_mb": round(p.stat().st_size / 1e6, 1)}
                break
        else:
            if low.endswith((".laz", ".las", ".ply")):
                files.append({"name": nm, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 1)})
    return {
        "dir": str(base),
        "parent": str(base.parent) if base.parent != base else None,
        "dirs": dirs,
        "tiles": list(tiles.values()),
        "files": files,
    }


@router.post("/load")
def load(req: LoadReq):
    try:
        return session.load(req.path, req.mode, req.max_display)
    except SessionError as exc:
        return _err(exc)
    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@router.post("/upload")
async def upload(file: UploadFile = File(...), max_display: Optional[int] = Form(None)):
    suffix = Path(file.filename or "cloud.laz").suffix.lower()
    if suffix not in (".laz", ".las", ".ply"):
        return JSONResponse({"error": f"Unsupported file type {suffix}"}, status_code=400)
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            while chunk := await file.read(8 * 1024 * 1024):
                fh.write(chunk)
        xyz, inten = load_cloud(tmp)
        return session.load_arrays(
            xyz, inten, None, Path(file.filename).stem.replace("_scan", ""),
            {"source_file": file.filename}, source=f"upload:{file.filename}",
            max_display=max_display,
        )
    except SessionError as exc:
        return _err(exc)
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


@router.get("/points")
def points():
    try:
        return Response(session.display_points(), media_type="application/octet-stream")
    except SessionError as exc:
        return _err(exc)


@router.get("/labels")
def labels():
    try:
        session.require()
        data = np.ascontiguousarray(session.display_labels(), dtype="<i4").tobytes()
        return Response(data, media_type="application/octet-stream",
                        headers={"X-Version": str(session.editor.version)})
    except SessionError as exc:
        return _err(exc)


@router.get("/material")
def material():
    try:
        return Response(session.display_material(), media_type="application/octet-stream")
    except SessionError as exc:
        return _err(exc)


@router.get("/trees")
def trees():
    try:
        return {"trees": session.trees()}
    except SessionError as exc:
        return _err(exc)


@router.post("/edit/{op}")
async def edit(op: str, request: Request, target: Optional[int] = None, source: Optional[int] = None):
    try:
        sel = await _read_indices(request)
        return _edit_payload(session.edit(op, sel, target=target, source=source))
    except SessionError as exc:
        return _err(exc)


@router.post("/grow")
async def grow(request: Request, radius: float = 0.3, same_label: bool = True):
    try:
        sel = await _read_indices(request)
        out = session.grow(sel, radius=radius, same_label=same_label)
        return Response(out.astype("<u4").tobytes(), media_type="application/octet-stream")
    except SessionError as exc:
        return _err(exc)


@router.post("/undo")
def undo():
    try:
        return _edit_payload(session.undo())
    except SessionError as exc:
        return _err(exc)


@router.post("/redo")
def redo():
    try:
        return _edit_payload(session.redo())
    except SessionError as exc:
        return _err(exc)


@router.post("/review")
def review(req: ReviewReq):
    try:
        session.set_reviewed(req.tree_id, req.reviewed)
        return session.info()
    except SessionError as exc:
        return _err(exc)


@router.post("/autosave/restore")
def autosave_restore():
    try:
        return _edit_payload(session.restore_autosave())
    except SessionError as exc:
        return _err(exc)


@router.post("/autosave/discard")
def autosave_discard():
    session.discard_autosave()
    return session.info()


@router.post("/save")
def save(req: SaveReq):
    try:
        paths = session.save(req.out_dir, req.name, req.overwrite)
        return {"paths": paths, "info": session.info()}
    except SessionError as exc:
        return _err(exc)


@router.post("/segment")
def segment(req: SegmentReq):
    from ..segment import METHODS

    if req.method not in METHODS:
        return JSONResponse({"error": f"Unknown method {req.method}"}, status_code=400)
    try:
        return session.run_segmentation(req.method, req.leaf_removal, req.treex_stock).to_dict()
    except SessionError as exc:
        return _err(exc)


@router.post("/woodleaf")
def woodleaf(req: WoodLeafReq):
    from ..wood_leaf import METHODS

    if req.method not in METHODS:
        return JSONResponse({"error": f"Unknown method {req.method}"}, status_code=400)
    params = {k: v for k, v in req.params.items() if k in WOOD_LEAF_PARAM_KEYS and v is not None}
    try:
        return session.run_wood_leaf(req.method, params).to_dict()
    except SessionError as exc:
        return _err(exc)


@router.get("/jobs/{job_id}")
def job(job_id: str):
    j = session.jobs.get(job_id)
    if j is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return j.to_dict()


@router.get("/methods")
def methods():
    from ..segment import METHODS as SEG
    from ..wood_leaf import METHODS as WL

    return {"segment": list(SEG), "woodleaf": list(WL), "woodleaf_needs_intensity": list(INTENSITY_WOOD_LEAF)}
