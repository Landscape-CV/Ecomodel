"""
Per-tree QSM metrics.

Wraps TreeQSM's own validated chain - branches() + tree_data() - to compute
per-tree attributes (DBH, height, trunk/branch/total volume, branch counts)
from a single tree's cylinder model.  This must run at QSM time, where the full
per-cylinder topology (branch/parent/extension) still exists; the saved cylinder
file does not carry it.

Failures are swallowed and return None so one bad tree never aborts a run.
"""

from __future__ import annotations

import numpy as np

# Order of the flat metric columns written to tree_metrics.csv.
TREE_METRIC_COLS = [
    "tree_id",
    "n_cylinders",
    "height_m",
    "dbh_cm",
    "basal_diam_cm",
    "total_volume_L",
    "trunk_volume_L",
    "branch_volume_L",
    "max_branch_order",
    "n_branches",
]

# Diameter at breast height is only meaningful for stems at least this tall.
_DBH_MIN_HEIGHT_M = 1.3


def _basal_diameter_m(cylinder) -> "float | None":
    """Diameter (m) of the lowest trunk cylinder; None if no trunk cylinders."""
    trunk = np.asarray(cylinder["branch"]) == 0
    if not trunk.any():
        return None
    z = np.asarray(cylinder["start"])[trunk, 2]
    radius = np.asarray(cylinder["radius"])[trunk]
    return float(2.0 * radius[int(np.argmin(z))])


def compute_tree_metrics(cylinder, tree_cloud, inputs, tree_id) -> "dict | None":
    """
    Run branches() + tree_data() for one tree and return a flat metric dict.

    Returns None if the topology/metric step fails for this tree.  ``inputs`` is
    the QSM input dict for the tree; display/plot/triangulation are forced off.
    """
    try:
        from PyTLidar.TreeQSMSteps.branches import branches
        from PyTLidar.TreeQSMSteps.tree_data import tree_data

        br = branches(cylinder)
        inp = dict(inputs)
        inp["Tria"] = False
        inp["disp"] = 0
        inp["plot"] = 0
        td, _ = tree_data(cylinder, br, tree_cloud, inp)
    except Exception:
        return None

    def _num(key):
        v = td.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    height = _num("TreeHeight")
    dbh_m = _num("DBHcyl")                       # tree_data stores DBH as a diameter (m)
    basal_m = _basal_diameter_m(cylinder)

    return {
        "tree_id":          int(tree_id),
        "n_cylinders":      int(len(cylinder["radius"])),
        "height_m":         round(height, 4),
        # DBH (1.3 m) is undefined for short shrubs/mangroves - leave blank there.
        "dbh_cm":           round(dbh_m * 100.0, 2) if height >= _DBH_MIN_HEIGHT_M else "",
        "basal_diam_cm":    round(basal_m * 100.0, 2) if basal_m is not None else "",
        "total_volume_L":   round(_num("TotalVolume"), 3),
        "trunk_volume_L":   round(_num("TrunkVolume"), 3),
        "branch_volume_L":  round(_num("BranchVolume"), 3),
        "max_branch_order": int(_num("MaxBranchOrder")) if np.isfinite(_num("MaxBranchOrder")) else 0,
        "n_branches":       int(_num("NumberBranches")) if np.isfinite(_num("NumberBranches")) else 0,
    }
