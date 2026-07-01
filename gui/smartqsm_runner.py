"""
Runs SmartQSM (installed separately, AGPL) on per-segment wood clouds via
subprocess and parses its .mat output into the Cx8 cylinder array. Optional QSM
backend for the lite pipeline; not imported or bundled.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np


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


def run_smartqsm_on_segments(point_cloud, instance_labels, out_dir, config, log=None):
    def _log(msg):
        if log:
            log(msg)

    empty = np.empty((0, 8))
    sq_dir = getattr(config, "smartqsm_dir", "")
    sq_py = getattr(config, "smartqsm_python", "")
    sq_cfg = getattr(config, "smartqsm_config", "")
    if not (sq_dir and sq_py and sq_cfg):
        _log("[SmartQSM] dir/python/config not set; skipping.\n")
        return empty

    seg_dir = os.path.join(out_dir, "smartqsm")
    os.makedirs(seg_dir, exist_ok=True)
    files = []
    for seg in np.unique(instance_labels):
        if seg == -1:
            continue
        pts = point_cloud[instance_labels == seg, :3]
        if len(pts) < 100:
            continue
        f = os.path.join(seg_dir, f"seg_{int(seg)}.xyz")
        np.savetxt(f, pts, fmt="%.6f")
        files.append(f)
    if not files:
        return empty

    cmd = [sq_py, os.path.join("entrypoints", "smartqsm.py"), "-y", "-c", sq_cfg, *files]
    _log(f"[SmartQSM] running on {len(files)} segment(s)...\n")
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
