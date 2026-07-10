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
from urllib import request as _urlrequest

import numpy as np


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


# ── remote GPU service (remote path) ─────────────────────────────────────────
def _reconstruct_remote(pts, url, timeout=1200, poll=4.0):
    """Run one segment on the remote GPU service; return an (M,8) Cx8 array.

    Uses a submit + poll protocol so no single HTTP request stays open long
    enough to hit a tunnel edge timeout (Cloudflare quick tunnels cut requests
    at ~100 s, but a SmartQSM reconstruction takes minutes):

      POST /reconstruct   body = raw ``.npy`` of float32 (N,3) points
                          -> {"job_id": "..."} immediately
      GET  /result/<id>   202 while running, 200 + raw ``.npy`` Cx8 when done,
                          500 (+message) on failure.

    Stdlib-only (urllib) so there is no extra dependency on the Mac side.
    """
    base = url.rstrip("/")
    buf = io.BytesIO()
    np.save(buf, np.asarray(pts, dtype=np.float32))
    req = _urlrequest.Request(
        base + "/reconstruct",
        data=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with _urlrequest.urlopen(req, timeout=120) as resp:
        job_id = json.loads(resp.read().decode())["job_id"]

    t0 = time.time()
    while True:
        with _urlrequest.urlopen(base + "/result/" + job_id, timeout=120) as r:
            code = r.getcode()
            data = r.read()
        if code == 200:
            arr = np.load(io.BytesIO(data))
            return np.atleast_2d(np.asarray(arr, dtype=float))
        # 202 -> still running
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job {job_id} exceeded {timeout}s")
        time.sleep(poll)


def _collect_segments(point_cloud, instance_labels, min_pts=100):
    """Yield (segment_id, points(N,3)) for each real instance with enough points."""
    segs = []
    for seg in np.unique(instance_labels):
        if seg == -1:
            continue
        pts = point_cloud[instance_labels == seg, :3]
        if len(pts) < min_pts:
            continue
        segs.append((int(seg), pts))
    return segs


def run_smartqsm_on_segments(point_cloud, instance_labels, out_dir, config, log=None):
    def _log(msg):
        if log:
            log(msg)

    empty = np.empty((0, 8))
    segs = _collect_segments(point_cloud, instance_labels)
    if not segs:
        _log("[SmartQSM] no segments with >=100 points; nothing to reconstruct.\n")
        return empty

    # ── REMOTE path ──────────────────────────────────────────────────────────
    url = os.environ.get("SMARTQSM_URL", "") or getattr(config, "smartqsm_url", "")
    if url:
        _log(f"[SmartQSM] REMOTE GPU service: {url}\n")
        _log(f"[SmartQSM] {len(segs)} segment(s) to send.\n")
        blocks = []
        for sid, pts in segs:
            _log(f"[SmartQSM]   segment {sid}: sending {len(pts)} pts to GPU...\n")
            try:
                arr = _reconstruct_remote(pts, url)
            except Exception as exc:
                _log(f"[SmartQSM]   segment {sid} FAILED: {exc}\n")
                continue
            if arr is not None and len(arr):
                _log(f"[SmartQSM]   segment {sid}: got {len(arr)} cylinders.\n")
                blocks.append(arr)
            else:
                _log(f"[SmartQSM]   segment {sid}: empty result.\n")
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
