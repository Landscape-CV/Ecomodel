"""
Runs aRchi (CeCILL, R package) via Rscript and converts the QSM CSV to Cx9.

Requires R + Rtools and: remotes::install_github("umr-amap/aRchi")
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARCHI_SCRIPT = ROOT_DIR / "SyntheticPipeline" / "scripts" / "run_archi_qsm.R"


def _empty_cx9():
    return np.empty((0, 9), dtype=float)


def parse_archi_csv_to_cx9(csv_path):
    """Convert aRchi QSM CSV (start/end/radius_cyl/length/branching_order) to Cx9."""
    import pandas as pd

    frame = pd.read_csv(csv_path)
    required = [
        "startX", "startY", "startZ",
        "endX", "endY", "endZ",
        "radius_cyl", "length", "branching_order",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"aRchi CSV missing columns: {missing}")

    if len(frame) == 0:
        return _empty_cx9()

    start = frame[["startX", "startY", "startZ"]].to_numpy(dtype=float)
    end = frame[["endX", "endY", "endZ"]].to_numpy(dtype=float)
    length = frame["length"].to_numpy(dtype=float)
    radius = frame["radius_cyl"].to_numpy(dtype=float)
    order = frame["branching_order"].to_numpy(dtype=float)

    # Prefer geometric length from endpoints when CSV length is missing/zero.
    deltas = end - start
    geom_len = np.linalg.norm(deltas, axis=1)
    use_geom = ~(np.isfinite(length) & (length > 1e-9))
    length = np.where(use_geom, geom_len, length)
    keep = np.isfinite(length) & (length > 1e-9)
    if not np.any(keep):
        return _empty_cx9()

    start = start[keep]
    deltas = deltas[keep]
    length = length[keep]
    radius = np.nan_to_num(radius[keep], nan=0.0)
    order = np.nan_to_num(order[keep], nan=0.0)
    axes = deltas / length[:, None]
    return np.column_stack((start, radius, axes, length, order))


def _kill_process_tree(process):
    try:
        import psutil

        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        process.kill()
    process.wait(timeout=5)


def run_archi_on_cloud(points, out_dir, config, log=None):
    """Run aRchi on one point cloud and return Cx9 cylinders (normalized frame)."""

    def _log(msg):
        if log:
            log(msg)

    empty = _empty_cx9()
    rscript = getattr(config, "archi_rscript", "") or "Rscript"
    script = getattr(config, "archi_script", "") or str(DEFAULT_ARCHI_SCRIPT)
    timeout = getattr(config, "archi_timeout", None)
    memory_limit_gb = getattr(config, "archi_memory_limit_gb", None)
    d = getattr(config, "archi_d", 0.5)
    cl_dist = getattr(config, "archi_cl_dist", 0.2)
    max_d = getattr(config, "archi_max_d", 1.0)
    sec_length = getattr(config, "archi_sec_length", 0.5)
    peak_rss_gb = 0.0
    setattr(config, "archi_peak_rss_gb", peak_rss_gb)
    setattr(config, "archi_status", "error")
    setattr(config, "archi_error", "")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 100:
        setattr(config, "archi_error", "need at least 100 XYZ points")
        return empty
    if not os.path.isfile(script):
        setattr(config, "archi_error", f"aRchi R script not found: {script}")
        _log(f"[aRchi] {config.archi_error}\n")
        return empty

    work_dir = os.path.join(out_dir, "archi")
    os.makedirs(work_dir, exist_ok=True)
    xyz_path = os.path.join(work_dir, "tree.xyz")
    csv_path = os.path.join(work_dir, "qsm.csv")
    np.savetxt(xyz_path, points[:, :3], fmt="%.10f")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    cmd = [
        rscript,
        "--vanilla",
        script,
        xyz_path,
        csv_path,
        str(d),
        str(cl_dist),
        str(max_d),
        str(sec_length),
    ]
    env = os.environ.copy()
    user_lib = env.get("R_LIBS_USER") or str(Path.home() / "R" / "win-library" / "4.6")
    env.setdefault("R_LIBS_USER", user_lib)
    _log(f"[aRchi] running: {' '.join(cmd)}\n")
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, \
             tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                cmd, stdout=stdout_file, stderr=stderr_file, text=True, env=env
            )
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
                        setattr(config, "archi_peak_rss_gb", peak_rss_gb)
                        if rss_gb > memory_limit_gb:
                            failure = f"exceeded {memory_limit_gb:.2f} GB"
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                time.sleep(0.2)

            if failure:
                _kill_process_tree(process)
                setattr(config, "archi_peak_rss_gb", peak_rss_gb)
                setattr(
                    config,
                    "archi_status",
                    "timeout" if "timed out" in failure else "memory_limit",
                )
                setattr(config, "archi_error", failure)
                _log(f"[aRchi] {failure}.\n")
                return empty

            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
            if process.returncode != 0:
                setattr(config, "archi_status", "error")
                setattr(
                    config,
                    "archi_error",
                    f"Rscript exit={process.returncode}\nSTDOUT: {stdout[-3000:]}\nSTDERR: {stderr[-3000:]}",
                )
                _log(f"[aRchi] {config.archi_error}\n")
                return empty
    except FileNotFoundError:
        setattr(config, "archi_error", f"Rscript not found: {rscript}")
        _log(f"[aRchi] {config.archi_error}\n")
        return empty
    except Exception as exc:
        setattr(config, "archi_error", str(exc))
        _log(f"[aRchi] failed: {exc}\n")
        return empty

    if not os.path.isfile(csv_path):
        setattr(config, "archi_status", "error")
        setattr(config, "archi_error", f"QSM CSV not written: {csv_path}\nSTDOUT: {stdout[-2000:]}\nSTDERR: {stderr[-2000:]}")
        _log(f"[aRchi] {config.archi_error}\n")
        return empty

    try:
        cylinders = parse_archi_csv_to_cx9(csv_path)
    except Exception as exc:
        setattr(config, "archi_status", "error")
        setattr(config, "archi_error", f"failed to parse {csv_path}: {exc}")
        _log(f"[aRchi] {config.archi_error}\n")
        return empty

    setattr(config, "archi_peak_rss_gb", peak_rss_gb)
    if len(cylinders) == 0:
        setattr(config, "archi_status", "error")
        setattr(config, "archi_error", f"aRchi CSV produced zero cylinders: {csv_path}")
        return empty

    setattr(config, "archi_status", "ok")
    setattr(config, "archi_error", "")
    _log(f"[aRchi] parsed {len(cylinders)} cylinders from {csv_path}\n")
    return cylinders
