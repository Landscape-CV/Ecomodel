"""
Runs AdTree (GPL-3, vendored separately) on a single-tree XYZ cloud via CLI
and parses the skeleton PLY into the Cx9 cylinder array.

AdTree exit codes are inverted: success returns 1, failure returns 0. Always
judge success by whether *_skeleton.ply exists.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections import defaultdict, deque

import numpy as np


def _empty_cx9():
    return np.empty((0, 9), dtype=float)


def _branch_orders_from_graph(vertices, edges):
    """BFS depth from the lowest-Z vertex; root order is 0."""
    n = len(vertices)
    if n == 0 or len(edges) == 0:
        return np.zeros(0, dtype=float)

    adj = defaultdict(list)
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i))
        adj[b].append((a, i))

    root = int(np.argmin(vertices[:, 2]))
    edge_order = np.full(len(edges), -1, dtype=float)
    visited = {root}
    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        for neighbor, edge_idx in adj[node]:
            if edge_order[edge_idx] < 0:
                edge_order[edge_idx] = depth
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    edge_order[edge_order < 0] = 0
    return edge_order


def parse_skeleton_ply_to_cx9(ply_path):
    """Convert AdTree skeleton PLY (vertices+radius, edges) to Cx9 cylinders."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        return _empty_cx9()

    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        return _empty_cx9()

    xyz = np.column_stack(
        (
            np.asarray(vertex["x"], dtype=float),
            np.asarray(vertex["y"], dtype=float),
            np.asarray(vertex["z"], dtype=float),
        )
    )
    if "radius" in names:
        radii = np.asarray(vertex["radius"], dtype=float)
    else:
        radii = np.zeros(len(xyz), dtype=float)

    if "edge" not in ply:
        return _empty_cx9()

    edge_data = ply["edge"].data
    edge_names = edge_data.dtype.names or ()
    edges = []
    if "vertex_indices" in edge_names:
        for item in edge_data["vertex_indices"]:
            idx = np.asarray(item, dtype=int).reshape(-1)
            if len(idx) >= 2:
                edges.append((int(idx[0]), int(idx[1])))
    elif {"vertex1", "vertex2"}.issubset(edge_names):
        for a, b in zip(edge_data["vertex1"], edge_data["vertex2"]):
            edges.append((int(a), int(b)))
    else:
        return _empty_cx9()

    if not edges:
        return _empty_cx9()

    edges = np.asarray(edges, dtype=int)
    starts = xyz[edges[:, 0]]
    ends = xyz[edges[:, 1]]
    deltas = ends - starts
    lengths = np.linalg.norm(deltas, axis=1)
    keep = lengths > 1e-9
    if not np.any(keep):
        return _empty_cx9()

    edges = edges[keep]
    starts = starts[keep]
    deltas = deltas[keep]
    lengths = lengths[keep]
    axes = deltas / lengths[:, None]
    radius = 0.5 * (radii[edges[:, 0]] + radii[edges[:, 1]])
    branch_order = _branch_orders_from_graph(xyz, edges)
    return np.column_stack((starts, radius, axes, lengths, branch_order))


def _find_skeleton_ply(out_dir, stem):
    candidates = [
        os.path.join(out_dir, f"{stem}_skeleton.ply"),
        os.path.join(out_dir, f"{stem}.skeleton.ply"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    matches = [
        os.path.join(out_dir, name)
        for name in os.listdir(out_dir)
        if name.lower().endswith("_skeleton.ply") or name.lower().endswith(".skeleton.ply")
    ]
    return matches[0] if matches else None


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


def run_adtree_on_cloud(points, out_dir, config, log=None):
    """Run AdTree on one point cloud and return Cx9 cylinders (normalized frame)."""

    def _log(msg):
        if log:
            log(msg)

    empty = _empty_cx9()
    exe = getattr(config, "adtree_exe", "") or ""
    timeout = getattr(config, "adtree_timeout", None)
    memory_limit_gb = getattr(config, "adtree_memory_limit_gb", None)
    peak_rss_gb = 0.0
    setattr(config, "adtree_peak_rss_gb", peak_rss_gb)
    setattr(config, "adtree_status", "error")
    setattr(config, "adtree_error", "")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 100:
        setattr(config, "adtree_error", "need at least 100 XYZ points")
        return empty
    if not exe or not os.path.isfile(exe):
        setattr(config, "adtree_error", f"AdTree executable not found: {exe}")
        _log(f"[AdTree] {config.adtree_error}\n")
        return empty

    work_dir = os.path.join(out_dir, "adtree")
    os.makedirs(work_dir, exist_ok=True)
    stem = "tree"
    xyz_path = os.path.join(work_dir, f"{stem}.xyz")
    np.savetxt(xyz_path, points[:, :3], fmt="%.10f")

    cmd = [exe, xyz_path, work_dir, "-s"]
    _log(f"[AdTree] running: {' '.join(cmd)}\n")
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, \
             tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(exe) or None,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
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
                        setattr(config, "adtree_peak_rss_gb", peak_rss_gb)
                        if rss_gb > memory_limit_gb:
                            failure = f"exceeded {memory_limit_gb:.2f} GB"
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                time.sleep(0.2)

            if failure:
                _kill_process_tree(process)
                setattr(config, "adtree_peak_rss_gb", peak_rss_gb)
                setattr(
                    config,
                    "adtree_status",
                    "timeout" if "timed out" in failure else "memory_limit",
                )
                setattr(config, "adtree_error", failure)
                _log(f"[AdTree] {failure}.\n")
                return empty

            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
    except Exception as exc:
        setattr(config, "adtree_error", str(exc))
        _log(f"[AdTree] failed: {exc}\n")
        return empty

    skeleton = _find_skeleton_ply(work_dir, stem)
    # AdTree exit codes are inverted; presence of skeleton PLY is the success signal.
    if skeleton is None:
        setattr(config, "adtree_status", "error")
        setattr(
            config,
            "adtree_error",
            f"no skeleton PLY after exit={process.returncode}\nSTDOUT: {stdout[-2000:]}\nSTDERR: {stderr[-2000:]}",
        )
        _log(f"[AdTree] {config.adtree_error}\n")
        return empty

    try:
        cylinders = parse_skeleton_ply_to_cx9(skeleton)
    except Exception as exc:
        setattr(config, "adtree_status", "error")
        setattr(config, "adtree_error", f"failed to parse {skeleton}: {exc}")
        _log(f"[AdTree] {config.adtree_error}\n")
        return empty

    setattr(config, "adtree_peak_rss_gb", peak_rss_gb)
    if len(cylinders) == 0:
        setattr(config, "adtree_status", "error")
        setattr(config, "adtree_error", f"skeleton PLY produced zero cylinders: {skeleton}")
        return empty

    setattr(config, "adtree_status", "ok")
    setattr(config, "adtree_error", "")
    _log(f"[AdTree] parsed {len(cylinders)} cylinders from {skeleton}\n")
    return cylinders
