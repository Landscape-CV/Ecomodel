"""Run benchmark instance methods on an in-memory XYZI cloud."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_SP_DIR = Path(__file__).resolve().parents[1]
_ROOT = _SP_DIR.parent
for p in (str(_ROOT), str(_SP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

METHODS = ("scanline", "treelearn", "treex", "tls2trees", "pointsam")

_DEFAULT_TREELEARN = str(_ROOT / "TreeLearn" / "configs" / "pipeline" / "ecomodel.yaml")
_DEFAULT_POINTSAM = str(
    _ROOT / "thirdparty" / "checkpoints" / "point_sam" / "model.safetensors"
)


def default_ckpt_kwargs(
    *,
    treelearn_config: Optional[str] = None,
    pointsam_ckpt: Optional[str] = None,
    treex_stock_tls: bool = True,
    leaf_removal: bool = False,
    results_folder: Optional[str] = None,
) -> Dict:
    return {
        "treelearn_config_path": treelearn_config or _DEFAULT_TREELEARN,
        "treelearn_use_gpu": True,
        "pointsam_ckpt": pointsam_ckpt or _DEFAULT_POINTSAM,
        "pointsam_config": "large",
        "pointsam_use_gpu": True,
        "run_leaf_removal": bool(leaf_removal),
        "results_folder": results_folder
        or str(_SP_DIR / "output" / "annotator_workdir"),
        "treex_adapt_synthetic": not bool(treex_stock_tls),
    }


SPARSE_COVERAGE = 0.5
DEFAULT_PROPAGATE_RADIUS = 0.2


def propagate_labels(
    xyz: np.ndarray, seed_idx: np.ndarray, seed_labels: np.ndarray, *, radius: float = DEFAULT_PROPAGATE_RADIUS
) -> np.ndarray:
    """Give every point the label of its nearest seed point within `radius` (else -1).

    Seeds with label -1 also count, so ground/non-tree voxels keep nearby points non-tree.
    """
    from scipy.spatial import cKDTree

    full = np.full(len(xyz), -1, dtype=np.int32)
    if len(seed_idx) == 0:
        return full
    tree = cKDTree(xyz[seed_idx])
    dist, nn = tree.query(xyz, k=1, distance_upper_bound=radius, workers=-1)
    hit = np.isfinite(dist)
    full[hit] = seed_labels[nn[hit]]
    full[seed_idx] = seed_labels
    return full


def run_method(
    xyz: np.ndarray,
    intensity01: np.ndarray,
    method: str,
    *,
    leaf_removal: bool = False,
    treex_stock_tls: bool = True,
    treelearn_config: Optional[str] = None,
    pointsam_ckpt: Optional[str] = None,
    results_folder: Optional[str] = None,
    propagate_radius: float = DEFAULT_PROPAGATE_RADIUS,
) -> Tuple[np.ndarray, Dict]:
    """
    Run one instance method; return full-length labels (N,) and info dict.

    Points not covered by the segmenter stay -1. Labels are normalized to
    trees >=0, non-tree -1.
    """
    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {METHODS}")

    from scripts.benchmark_instance_segmentation import run_segmenter_on_tile

    from .labels import remap_treelearn_labels

    xyz = np.asarray(xyz, dtype=np.float64)
    inten = np.asarray(intensity01, dtype=np.float64).reshape(-1)
    if len(inten) != len(xyz):
        raise ValueError("intensity length mismatch")
    points = np.hstack([xyz, inten.reshape(-1, 1)])

    kwargs = default_ckpt_kwargs(
        treelearn_config=treelearn_config,
        pointsam_ckpt=pointsam_ckpt,
        treex_stock_tls=treex_stock_tls,
        leaf_removal=leaf_removal,
        results_folder=results_folder,
    )
    os.makedirs(kwargs["results_folder"], exist_ok=True)

    work, labels, orig_idx = run_segmenter_on_tile(
        points, segmenter_type=method, **kwargs
    )

    full = np.full(len(xyz), -1, dtype=np.int32)
    info: Dict = {
        "method": method,
        "ok": False,
        "num_pred_points": 0,
        "num_trees": 0,
        "message": "",
    }
    if work is None or labels is None or orig_idx is None:
        info["message"] = "segmentation returned no points"
        return full, info

    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    oi = np.asarray(orig_idx, dtype=np.int64).reshape(-1)
    if len(lab) != len(oi):
        info["message"] = f"label/index length mismatch {len(lab)} vs {len(oi)}"
        return full, info

    if method == "treelearn":
        lab = remap_treelearn_labels(lab)
    else:
        lab = lab.copy()
        lab[lab < 0] = -1

    # Guard invalid indices
    valid = (oi >= 0) & (oi < len(full))
    oi, lab = oi[valid], lab[valid]
    full[oi] = lab
    info["ok"] = True
    info["num_pred_points"] = int(len(oi))
    if len(np.unique(oi)) < SPARSE_COVERAGE * len(full):
        # Voxelized output (e.g. TreeLearn at 0.1 m) labels one point per voxel;
        # spread each label to the full-resolution points around it.
        full = propagate_labels(xyz, oi, lab, radius=propagate_radius)
        info["propagated"] = True
        info["num_pred_points"] = int((full >= 0).sum())
    info["num_trees"] = int(len(np.unique(full[full >= 0])))
    info["message"] = f"trees={info['num_trees']} labeled_pts={info['num_pred_points']}"
    return full, info
