"""
Runs SmartQSM (installed separately, AGPL) on per-segment wood clouds via
subprocess and parses its .mat output into the Cx9 cylinder array. Optional QSM
backend for the lite pipeline; not imported or bundled.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

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
        branch_order = np.asarray(
            getattr(cyl, "BranchOrder", np.zeros_like(radius)),
            dtype=float,
        ).reshape(-1, 1)
        if start.shape[0] == 3 and start.shape[1] != 3:
            start = start.T
        if axis.shape[0] == 3 and axis.shape[1] != 3:
            axis = axis.T
        return np.concatenate([start, radius, axis, length, branch_order], axis=1)
    except Exception:
        return None


def run_smartqsm_on_segments(point_cloud, instance_labels, out_dir, config, log=None):
    def _log(msg):
        if log:
            log(msg)

    empty = np.empty((0, 9))
    sq_dir = getattr(config, "smartqsm_dir", "")
    sq_py = getattr(config, "smartqsm_python", "")
    sq_cfg = getattr(config, "smartqsm_config", "")
    timeout = getattr(config, "smartqsm_timeout", None)
    memory_limit_gb = getattr(config, "smartqsm_memory_limit_gb", None)
    peak_rss_gb = 0.0
    setattr(config, "smartqsm_peak_rss_gb", peak_rss_gb)
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

    cmd = [sq_py, os.path.join("entrypoints", "smartqsm.py"), "-y", "-t", "-c", sq_cfg, *files]
    _log(f"[SmartQSM] running on {len(files)} segment(s)...\n")
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, \
             tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(cmd, cwd=sq_dir, stdout=stdout_file, stderr=stderr_file, text=True)
            started = time.time()
            failure = None
            while process.poll() is None:
                if timeout and time.time() - started > timeout:
                    failure = f"timed out after {timeout} seconds"
                    break
                if memory_limit_gb:
                    try:
                        import psutil
                        parent = psutil.Process(process.pid)
                        rss = parent.memory_info().rss
                        rss += sum(
                            child.memory_info().rss
                            for child in parent.children(recursive=True)
                            if child.is_running()
                        )
                        rss_gb = rss / (1024 ** 3)
                        peak_rss_gb = max(peak_rss_gb, rss_gb)
                        setattr(config, "smartqsm_peak_rss_gb", peak_rss_gb)
                        if rss_gb > memory_limit_gb:
                            failure = f"exceeded {memory_limit_gb:.2f} GB"
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                time.sleep(0.2)
            if failure:
                try:
                    import psutil
                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except Exception:
                    process.kill()
                process.wait(timeout=5)
                setattr(config, "smartqsm_peak_rss_gb", peak_rss_gb)
                _log(f"[SmartQSM] {failure}.\n")
                return empty
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
            if process.returncode != 0:
                setattr(config, "smartqsm_peak_rss_gb", peak_rss_gb)
                _log(f"[SmartQSM] subprocess returned {process.returncode}\nSTDOUT: {stdout}\nSTDERR: {stderr}\n")
                return empty
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
    setattr(config, "smartqsm_peak_rss_gb", peak_rss_gb)
    return np.concatenate(blocks, axis=0)
