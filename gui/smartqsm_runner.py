"""
Runs SmartQSM (installed separately, AGPL) on per-segment wood clouds and parses
its .mat output into the Cx8 cylinder array. Optional QSM backend for the lite
pipeline; not imported or bundled.

Two execution modes, chosen automatically:

  * REMOTE  - if a service URL is configured (env ``SMARTQSM_URL`` or
              ``config.smartqsm_url``), each segment's points are POSTed to a
              GPU service (e.g. Colab + cloudflared) that runs the real spconv
              deep-learning method and returns the Cx8 cylinder array. This is
              how the deep-learning method reaches a GPU from the arm64 Mac,
              which cannot run spconv locally.
  * LOCAL   - otherwise, SmartQSM is invoked via subprocess against a local
              checkout (``smartqsm_dir`` / ``smartqsm_python`` / ``smartqsm_config``).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from urllib import error as _urlerror
from urllib import request as _urlrequest

import numpy as np


# Per-segment size guard (see Docs/KNOWN_ISSUES.md #5). A segment far larger than
# this is almost always a segmentation failure — a whole multi-tree tile collapsed
# into one label — and even a genuine single tree this big will not finish inside
# the Modal GPU-job timeout. Fail fast with a clear message instead of waiting ~1h
# for the job to be killed. 3M pts ≈ ~17 min on a T4 (626k → 3.5 min, roughly
# linear), well under the 3600s server cap and well below the 4.9M-pt cloud that
# timed out. Override via env SMARTQSM_MAX_POINTS or config.smartqsm_max_points
# (a value <= 0 disables the guard).
MAX_SEGMENT_POINTS = 3_000_000

# Wall-clock ceiling for the fan-out poll loop. Segments run in PARALLEL on Modal,
# so this bounds the SLOWEST single job, not the sum. Keep >= the server job
# timeout (deploy/modal_app.py, 3600s) plus polling margin.
REMOTE_POLL_TIMEOUT = 3900


# ── .mat parsing (local path) ────────────────────────────────────────────────
def _parse_qsm_mat(mat_path):
    try:
        import scipy.io as sio
        m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        cyl = m.get("cylinder")
        if cyl is None:
            for k in ("QSM", "qsm", "model"):
                if k in m and hasattr(m[k], "cylinder"):
                    cyl = m[k].cylinder
                    break
        if cyl is None:
            return None
        start = np.atleast_2d(np.asarray(cyl.start, dtype=float))
        axis = np.atleast_2d(np.asarray(cyl.axis, dtype=float))
        radius = np.asarray(cyl.radius, dtype=float).reshape(-1, 1)
        length = np.asarray(cyl.length, dtype=float).reshape(-1, 1)
        if start.shape[0] == 3 and start.shape[1] != 3:
            start = start.T
        if axis.shape[0] == 3 and axis.shape[1] != 3:
            axis = axis.T
        return np.concatenate([start, radius, axis, length], axis=1)
    except Exception:
        return None


# ── remote GPU service: client-side fan-out (submit all, then poll all) ──────
#
# The service already runs each segment as an independent Modal job (POST spawns
# it and returns instantly), so the way to run many segments in PARALLEL is simply
# to submit them all BEFORE waiting on any: Modal then autoscales one GPU container
# per pending job. _submit_remote does the fire-and-forget POST; _poll_remote does
# one non-blocking result check; run_smartqsm_on_segments orchestrates the two
# phases. Stdlib-only (urllib) so there is no extra dependency on the Mac side.
def _submit_remote(pts, base):
    """POST one segment and return its job_id immediately, without waiting.

      POST /reconstruct   body = raw ``.npy`` of float32 (N,3) points
                          -> {"job_id": "..."} instantly (the service .spawn()s it)
    """
    buf = io.BytesIO()
    np.save(buf, np.asarray(pts, dtype=np.float32))
    req = _urlrequest.Request(
        base + "/reconstruct",
        data=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with _urlrequest.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())["job_id"]


def _poll_remote(base, job_id):
    """Check one job once (non-blocking). Return its (M,8) Cx8 array if done, or
    None if still running. Raises urllib HTTPError on a server-side failure (500).

      GET  /result/<id>   202 while running, 200 + raw ``.npy`` Cx8 when done,
                          500 (+message) on failure.

    urllib treats 202 as success (getcode()==202 -> return None) and raises
    HTTPError on 500, which the caller turns into a per-segment failure.
    """
    with _urlrequest.urlopen(base + "/result/" + job_id, timeout=120) as r:
        code = r.getcode()
        data = r.read()
    if code == 200:
        arr = np.load(io.BytesIO(data))
        return np.atleast_2d(np.asarray(arr, dtype=float))
    return None  # 202 -> still running


def _resolve_max_pts(config):
    """Effective per-segment point ceiling: env SMARTQSM_MAX_POINTS or
    config.smartqsm_max_points override MAX_SEGMENT_POINTS; a value <= 0 disables
    the guard (returns None).
    """
    raw = os.environ.get("SMARTQSM_MAX_POINTS", "").strip()
    if raw:
        try:
            n = int(float(raw))
            return None if n <= 0 else n
        except ValueError:
            pass
    v = getattr(config, "smartqsm_max_points", None)
    if v is not None:
        try:
            n = int(v)
            return None if n <= 0 else n
        except (TypeError, ValueError):
            pass
    return MAX_SEGMENT_POINTS


def _collect_segments(point_cloud, instance_labels, min_pts=100, max_pts=None, log=None):
    """Yield (segment_id, points(N,3)) for each real instance with enough points.

    Segments larger than max_pts are skipped and logged (the input-size guard):
    they are too big for one GPU job and are almost always a segmentation failure
    that merged multiple trees into one label. See Docs/KNOWN_ISSUES.md #5.
    """
    def _log(msg):
        if log:
            log(msg)

    segs = []
    for seg in np.unique(instance_labels):
        if seg == -1:
            continue
        pts = point_cloud[instance_labels == seg, :3]
        n = len(pts)
        if n < min_pts:
            continue
        if max_pts is not None and n > max_pts:
            _log(f"[SmartQSM]   segment {int(seg)}: SKIPPED — {n:,} pts exceeds the "
                 f"{max_pts:,}-pt limit. Too big for one GPU job (usually a "
                 f"segmentation failure that merged trees, or a single tree too "
                 f"large to finish inside the job timeout). Fix segmentation, "
                 f"downsample, or raise SMARTQSM_MAX_POINTS / use a faster GPU.\n")
            continue
        segs.append((int(seg), pts))
    return segs


def run_smartqsm_on_segments(point_cloud, instance_labels, out_dir, config, log=None):
    """Reconstruct every segmented tree with SmartQSM and return one Cx8 table.

    ``point_cloud`` is the whole tile in the pipeline's NORMALIZED frame — the tile
    mean has already been subtracted (see ecomodel_lite.normalize_point_cloud), so
    coordinates are small and centred near the origin. ``instance_labels`` assigns
    each point to a tree (-1 = unassigned / not a tree).

    Returns an (M,8) array stacking every tree's cylinders, one row per cylinder:

        [start_x, start_y, start_z, radius, axis_x, axis_y, axis_z, length]

    The caller (gui/pipeline_lite.py) un-normalizes this back to world coordinates
    (adds the tile mean to the start columns) before saving it to _cylinders.txt.
    """
    def _log(msg):
        if log:
            log(msg)

    empty = np.empty((0, 8))

    # Split the labelled cloud into one (id, points) pair per tree, dropping blobs
    # that are too small (< min_pts) or too big (> max_pts, the input-size guard).
    max_pts = _resolve_max_pts(config)
    segs = _collect_segments(point_cloud, instance_labels, max_pts=max_pts, log=log)
    if not segs:
        _log("[SmartQSM] no segments to reconstruct "
             "(none with >=100 points, or all over the size limit).\n")
        return empty

    # ── REMOTE path: run all trees in parallel on Modal (client-side fan-out) ──
    # A configured URL means "send the points to the Modal GPU service" — the only
    # way the deep-learning method reaches a GPU from this arm64 Mac. The env var
    # wins over the baked-in config default so a developer can aim at a test
    # deployment without editing config.
    url = os.environ.get("SMARTQSM_URL", "") or getattr(config, "smartqsm_url", "")
    if url:
        base = url.rstrip("/")
        _log(f"[SmartQSM] REMOTE GPU service: {base}\n")
        _log(f"[SmartQSM] {len(segs)} segment(s); submitting all, then polling in "
             f"parallel.\n")

        # ---- Phase 1: SUBMIT every tree up front, without waiting for any ------
        # Each POST returns a job_id in milliseconds because the service .spawn()s
        # the GPU work and replies immediately. Firing them ALL off before waiting
        # on any result is what makes the trees run concurrently: Modal sees N
        # pending jobs and autoscales one GPU container per job. We send each tree
        # in the tile's shared (normalized) frame, NOT re-centred per tree, so the
        # returned cylinders stay correctly positioned relative to every other tree.
        pending = {}                     # job_id -> segment id (jobs still in flight)
        for sid, pts in segs:
            try:
                job_id = _submit_remote(pts, base)
            except Exception as exc:     # network / HTTP error while submitting
                _log(f"[SmartQSM]   segment {sid}: submit FAILED: {exc}\n")
                continue
            pending[job_id] = sid
            _log(f"[SmartQSM]   segment {sid}: submitted {len(pts)} pts.\n")

        # ---- Phase 2: POLL until every job finishes (or we hit the time cap) ---
        # Sweep the outstanding jobs repeatedly, asking each "done yet?". Because
        # they run concurrently on Modal, the total wait is the SLOWEST single tree,
        # not the sum. Results are stored keyed by segment id so the final table is
        # assembled in a stable order no matter which tree finishes first.
        results = {}                     # segment id -> that tree's Cx8 array
        t0 = time.time()
        while pending:
            for job_id in list(pending):        # list() so we can pop while iterating
                sid = pending[job_id]
                try:
                    arr = _poll_remote(base, job_id)
                except _urlerror.HTTPError as exc:
                    # HTTP 500 = the GPU job itself failed. Drop this one tree and
                    # let the rest carry on rather than aborting the whole run.
                    _log(f"[SmartQSM]   segment {sid} FAILED: {exc}\n")
                    pending.pop(job_id, None)
                    continue
                except Exception as exc:
                    # A transient network blip, not a job failure — the job is still
                    # alive on Modal, so keep it pending and retry on the next sweep.
                    _log(f"[SmartQSM]   segment {sid}: poll error, will retry ({exc}).\n")
                    continue
                if arr is None:
                    continue                     # HTTP 202 -> still running, try later
                pending.pop(job_id, None)        # finished -> stop tracking it
                if len(arr):
                    _log(f"[SmartQSM]   segment {sid}: got {len(arr)} cylinders.\n")
                    results[sid] = arr
                else:
                    _log(f"[SmartQSM]   segment {sid}: empty result.\n")

            # Some jobs are still running: give up if we've blown the overall time
            # cap, otherwise pause before the next sweep so we don't hammer the API.
            if pending:
                if time.time() - t0 > REMOTE_POLL_TIMEOUT:
                    stuck = ", ".join(str(s) for s in pending.values())
                    _log(f"[SmartQSM]   timed out after {REMOTE_POLL_TIMEOUT}s; "
                         f"{len(pending)} segment(s) unfinished: {stuck}.\n")
                    break
                time.sleep(4.0)

        # Assemble in segment-id order (not completion order) so the saved cylinder
        # table is byte-for-byte identical run-to-run. Row order is otherwise
        # irrelevant: each row is a self-contained, absolute-coordinate cylinder
        # with no references to any other row.
        blocks = [results[sid] for sid in sorted(results)]
        return np.concatenate(blocks, axis=0) if blocks else empty

    # ── LOCAL path (unchanged) ───────────────────────────────────────────────
    sq_dir = getattr(config, "smartqsm_dir", "")
    sq_py = getattr(config, "smartqsm_python", "")
    sq_cfg = getattr(config, "smartqsm_config", "")
    if not (sq_dir and sq_py and sq_cfg):
        _log("[SmartQSM] no remote URL and dir/python/config not set; skipping.\n")
        return empty

    seg_dir = os.path.join(out_dir, "smartqsm")
    os.makedirs(seg_dir, exist_ok=True)
    files = []
    for sid, pts in segs:
        f = os.path.join(seg_dir, f"seg_{sid}.xyz")
        np.savetxt(f, pts, fmt="%.6f")
        files.append(f)

    cmd = [sq_py, os.path.join("entrypoints", "smartqsm.py"), "-y", "-c", sq_cfg, *files]
    _log(f"[SmartQSM] running locally on {len(files)} segment(s)...\n")
    try:
        subprocess.run(cmd, cwd=sq_dir, capture_output=True, text=True)
    except Exception as exc:
        _log(f"[SmartQSM] failed: {exc}\n")
        return empty

    blocks = []
    for f in files:
        mat = os.path.splitext(f)[0] + "_qsm.mat"
        if os.path.exists(mat):
            arr = _parse_qsm_mat(mat)
            if arr is not None and len(arr):
                blocks.append(arr)
    if not blocks:
        return empty
    return np.concatenate(blocks, axis=0)
