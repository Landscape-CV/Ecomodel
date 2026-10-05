"""API for island projects: streaming nodes, island trees, regions, island jobs."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..project.store import ProjectError
from . import routes as R
from .project_session import REGION_MAX_POINTS, ProjectSession
from .session import SessionError

router = APIRouter(prefix="/api")
psession = ProjectSession()


def _err(exc: Exception) -> JSONResponse:
    status = getattr(exc, "status", 400)
    extra = getattr(exc, "extra", {}) or {}
    return JSONResponse({"error": str(exc), **extra}, status_code=status)


def _bin(data: bytes) -> Response:
    return Response(data, media_type="application/octet-stream")


class OpenReq(BaseModel):
    path: str


class MergeReq(BaseModel):
    sources: List[int]
    target: int


class IdsReq(BaseModel):
    ids: List[int]


class ReviewReq(BaseModel):
    tree_id: int
    reviewed: bool


class RegionReq(BaseModel):
    bmin: List[float]
    bmax: List[float]
    max_points: Optional[int] = None
    max_display: Optional[int] = None


class IslandSegReq(BaseModel):
    method: str = "treelearn"
    buffer: float = 10.0
    voxel: Optional[float] = None
    leaf_removal: bool = False
    tiles: Optional[List[str]] = None


class ExportReq(BaseModel):
    out_dir: str = ""
    tree_ids: Optional[List[int]] = None
    write_tiles: bool = True


# ── project lifecycle ─────────────────────────────────────────────────────
@router.get("/project/status")
def project_status():
    return psession.info()


@router.post("/project/open")
def project_open(req: OpenReq):
    try:
        if R.session.link is not None:
            R.session.unload()
        return psession.open(req.path)
    except (ProjectError, SessionError) as exc:
        return _err(exc)


@router.post("/project/close")
def project_close():
    if R.session.link is not None:
        R.session.unload()
    psession.close()
    return psession.info()


@router.get("/project/hierarchy")
def project_hierarchy():
    try:
        return psession.require().hierarchy()
    except ProjectError as exc:
        return _err(exc)


@router.get("/project/estimate")
def project_estimate(x0: float, y0: float, x1: float, y1: float):
    try:
        p = psession.require()
        return {"upper": p.count_upper_bound([min(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)]),
                "max_points": REGION_MAX_POINTS}
    except ProjectError as exc:
        return _err(exc)


@router.get("/node/{gid}")
def node(gid: int):
    try:
        return _bin(psession.node_payload(gid))
    except ProjectError as exc:
        return _err(exc)


@router.get("/node/{gid}/labels")
def node_labels(gid: int):
    try:
        return _bin(psession.node_labels(gid))
    except ProjectError as exc:
        return _err(exc)


# ── island trees and tree-level edits ─────────────────────────────────────
@router.get("/island/trees")
def island_trees():
    try:
        return psession.trees()
    except ProjectError as exc:
        return _err(exc)


@router.post("/island/merge")
def island_merge(req: MergeReq):
    try:
        return psession.merge(req.sources, req.target)
    except ProjectError as exc:
        return _err(exc)


@router.post("/island/nontree")
def island_nontree(req: IdsReq):
    try:
        return psession.merge(req.ids, -1)
    except ProjectError as exc:
        return _err(exc)


@router.post("/island/undo")
def island_undo():
    try:
        return psession.undo()
    except ProjectError as exc:
        return _err(exc)


@router.post("/island/redo")
def island_redo():
    try:
        return psession.redo()
    except ProjectError as exc:
        return _err(exc)


@router.post("/island/review")
def island_review(req: ReviewReq):
    try:
        return psession.set_reviewed(req.tree_id, req.reviewed)
    except ProjectError as exc:
        return _err(exc)


# ── edit regions ──────────────────────────────────────────────────────────
@router.post("/region/open")
def region_open(req: RegionReq):
    try:
        p = psession.require()
        if R.session.busy():
            raise SessionError("A background job is running.", 409)
        lo = [min(a, b) for a, b in zip(req.bmin, req.bmax)]
        hi = [max(a, b) for a, b in zip(req.bmin, req.bmax)]
        data = psession.open_region(lo, hi, req.max_points or REGION_MAX_POINTS)
        link = data["link"]
        name = f"{p.meta.get('name', 'island')}_region"
        meta = {"project": str(p.root), "region_bmin": lo, "region_bmax": hi,
                "reviewed_trees": p.meta.get("reviewed", [])}
        info = R.session.load_arrays(data["xyz"], data["intensity"], data["labels"], name, meta,
                                     source=f"project:{p.root}", max_display=req.max_display,
                                     origin=p.origin, link=link)
        psession.attach_region(link)
        return {"info": info, "project": psession.info()}
    except (ProjectError, SessionError) as exc:
        return _err(exc)


@router.post("/region/close")
def region_close():
    try:
        if R.session.link is not None:
            R.session.unload()
        psession.close_region()
        return psession.info()
    except SessionError as exc:
        return _err(exc)


# ── island jobs ───────────────────────────────────────────────────────────
@router.post("/island/segment")
def island_segment(req: IslandSegReq):
    from ..project.segment_island import segment_island
    from ..segment import METHODS

    if req.method not in METHODS:
        return JSONResponse({"error": f"Unknown method {req.method}"}, status_code=400)
    try:
        p = psession.require()

        def work(job):
            def progress(done, total, msg):
                job.stage(msg, done / max(total, 1))

            summary = segment_island(p, req.method, buffer=req.buffer, voxel=req.voxel,
                                     leaf_removal=req.leaf_removal, tile_names=req.tiles,
                                     progress=progress, log=lambda s: print(s, flush=True))
            job.stage("Refreshing tree stats", None)
            psession.after_bulk_change()
            job.message = f"{summary['trees']} trees across {summary['tiles']} tiles"
            job.result = {"summary": summary}

        return psession.start_job("island_segment", work).to_dict()
    except ProjectError as exc:
        return _err(exc)


@router.post("/island/stitch")
def island_stitch():
    from ..project.stitch import find_candidates

    try:
        p = psession.require()

        def work(job):
            def progress(done, total, msg):
                job.stage(msg, done / max(total, 1))

            cands = find_candidates(p, progress=progress)
            psession.candidates = psession._load_candidates()
            job.message = f"{len(cands)} border contacts, {len(psession.candidates)} open for review"

        return psession.start_job("island_stitch", work).to_dict()
    except ProjectError as exc:
        return _err(exc)


@router.get("/island/candidates")
def island_candidates():
    return {"candidates": psession.candidates}


@router.post("/island/candidates/{key}/{action}")
def island_candidate(key: str, action: str):
    try:
        return psession.resolve_candidate(key, action == "accept")
    except ProjectError as exc:
        return _err(exc)


@router.post("/island/export")
def island_export(req: ExportReq):
    from ..project.export import export_project
    from .session import DEFAULT_EXPORT_DIR

    try:
        p = psession.require()
        out = req.out_dir or str(DEFAULT_EXPORT_DIR / p.meta.get("name", "island"))

        def work(job):
            def progress(done, total, msg):
                job.stage(msg, done / max(total, 1))

            paths = export_project(p, out, tree_ids=req.tree_ids, write_tiles=req.write_tiles,
                                   progress=progress, log=lambda s: None)
            job.message = f"Exported to {out}"
            job.result = {"out_dir": out, "files": len(paths)}

        return psession.start_job("island_export", work).to_dict()
    except ProjectError as exc:
        return _err(exc)
