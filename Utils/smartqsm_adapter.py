"""
SmartQSM -> PyTLidar adapter.

Translates a SmartQSM ``*_qsm.mat`` reconstruction into the cylinder
representation the rest of PyTLidar already understands, so a SmartQSM result
can flow through the same downstream code (tree_data metrics, the GUI Results
3D viewer, the spatial Query page) as a TreeQSM result.

The adapter does exactly three things and nothing more:

  1. Container change.  TreeQSM returns a Python dict in memory; SmartQSM writes
     a MATLAB ``.mat`` file to disk.  We load the file and rebuild the dict.
  2. Field reconciliation.  SmartQSM stores almost the same fields as TreeQSM.
     It is missing only the two fit-quality diagnostics ``SurfCov`` and ``mad``
     (filled with NaN here) and carries one extra field ``simplified`` (ignored).
  3. Coordinate alignment.  SmartQSM keeps cylinders in the input cloud's
     absolute (projected) coordinates.  Pass ``xyz_offset`` to subtract a
     constant vector if you need them in a recentred / normalized frame.

Field comparison (verified against a real tree_50_qsm.mat, 32,081 cylinders):

    TreeQSM cylinder dict        SmartQSM QSM.cylinder
    ---------------------        ---------------------
    start (N x 3)                start          (same)
    radius (N)                   radius         (same)
    axis (N x 3)                 axis           (same)
    length (N)                   length         (same)
    parent, extension            parent, extension
    branch, BranchOrder          branch, BranchOrder
    PositionInBranch             PositionInBranch
    added, UnmodRadius           added, UnmodRadius
    SurfCov, mad                 -- MISSING --> filled with NaN
    --                           simplified    (extra; ignored)

Public API
----------
load_smartqsm_mat(path) -> dict
    Raw ``QSM`` struct from the .mat (thin wrapper over scipy.io.loadmat).
smartqsm_to_cylinder(mat_path, xyz_offset=None) -> dict
    TreeQSM-compatible cylinder dict.
smartqsm_to_cx8(mat_path_or_cylinder, xyz_offset=None) -> np.ndarray (N x 8)
    Reduced [start(3), radius(1), axis(3), length(1)] array, matching the
    format ecomodel.get_all_cylinders() writes (ecomodel.py:1329).
write_cylinder_txt(cx8, out_path) -> None
    Saves the N x 8 array with np.savetxt, exactly as the GUI expects to read
    it back via np.loadtxt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Columns of the reduced Cx8 array, matching ecomodel.py:1329:
#   [start_x, start_y, start_z, radius, axis_x, axis_y, axis_z, length]
CX8_COLUMNS = ("start_x", "start_y", "start_z",
               "radius",
               "axis_x", "axis_y", "axis_z",
               "length")

# TreeQSM cylinder keys that SmartQSM does not provide; padded with NaN so any
# downstream code that touches them gets a well-defined (missing) value rather
# than a KeyError.  (tree_data() does not read these, but other tools might.)
_MISSING_FIELDS = ("SurfCov", "mad")


def load_smartqsm_mat(path: str | Path) -> dict:
    """
    Load a SmartQSM ``*_qsm.mat`` and return its ``QSM`` struct as a dict.

    Uses ``simplify_cells=True`` so nested MATLAB structs come back as plain
    Python dicts / numpy arrays instead of object arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SmartQSM .mat not found: {path}")
    m = loadmat(str(path), simplify_cells=True)
    # savemat wraps the model under a top-level 'QSM' key; fall back to the
    # whole dict if a future export drops the wrapper.
    return m["QSM"] if "QSM" in m else m


def smartqsm_to_cylinder(mat_path: str | Path,
                         xyz_offset: np.ndarray | None = None) -> dict:
    """
    Convert a SmartQSM ``*_qsm.mat`` into a TreeQSM-compatible cylinder dict.

    Parameters
    ----------
    mat_path
        Path to the SmartQSM ``*_qsm.mat`` file.
    xyz_offset
        Optional length-3 vector subtracted from every ``start`` point.  Use it
        to move cylinders out of absolute projected coordinates into a
        recentred / normalized frame (e.g. PyTLidar's per-tile cloud mean).
        ``None`` (default) leaves the original world coordinates untouched.

    Returns
    -------
    dict
        Keys ``start (N,3)``, ``radius (N,)``, ``axis (N,3)``, ``length (N,)``,
        plus the topology fields and NaN-padded ``SurfCov`` / ``mad``.
    """
    qsm = load_smartqsm_mat(mat_path)
    if "cylinder" not in qsm:
        raise KeyError(
            f"{Path(mat_path).name}: no 'cylinder' field in QSM struct "
            f"(found: {sorted(qsm.keys())})")
    cyl = qsm["cylinder"]

    n = int(np.asarray(cyl["radius"]).ravel().shape[0])

    start = np.asarray(cyl["start"], dtype=np.float64).reshape(n, 3)
    if xyz_offset is not None:
        xyz_offset = np.asarray(xyz_offset, dtype=np.float64).reshape(3)
        start = start - xyz_offset

    out: dict = {
        "start":  start,
        "radius": np.asarray(cyl["radius"], dtype=np.float64).ravel(),
        "axis":   np.asarray(cyl["axis"],   dtype=np.float64).reshape(n, 3),
        "length": np.asarray(cyl["length"], dtype=np.float64).ravel(),
    }

    # Carry through every topology / quality field SmartQSM does provide.
    for key in ("parent", "extension", "branch", "BranchOrder",
                "PositionInBranch", "added", "UnmodRadius"):
        if key in cyl:
            out[key] = np.asarray(cyl[key]).ravel()

    # Pad the two TreeQSM diagnostics SmartQSM omits, so downstream code that
    # expects them finds a defined (NaN) array rather than a missing key.
    for key in _MISSING_FIELDS:
        out[key] = np.full(n, np.nan, dtype=np.float64)

    return out


def smartqsm_to_cx8(mat_path_or_cylinder,
                    xyz_offset: np.ndarray | None = None) -> np.ndarray:
    """
    Produce the reduced ``N x 8`` cylinder array PyTLidar writes to disk.

    Accepts either a path to a ``*_qsm.mat`` or an already-built cylinder dict
    (from :func:`smartqsm_to_cylinder`).  Column order matches
    ``ecomodel.get_all_cylinders()`` (ecomodel.py:1329):
    ``[start(3), radius(1), axis(3), length(1)]``.
    """
    if isinstance(mat_path_or_cylinder, dict):
        cyl = mat_path_or_cylinder
    else:
        cyl = smartqsm_to_cylinder(mat_path_or_cylinder, xyz_offset=xyz_offset)

    return np.concatenate(
        (cyl["start"],
         cyl["radius"].reshape(-1, 1),
         cyl["axis"],
         cyl["length"].reshape(-1, 1)),
        axis=1,
    )


def write_cylinder_txt(cx8: np.ndarray, out_path: str | Path) -> Path:
    """
    Write an ``N x 8`` cylinder array to ``out_path`` with ``np.savetxt``.

    This is the exact format the GUI reads back via ``np.loadtxt`` in
    ``gui/query_engine.py`` and ``gui/results_widgets.py``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(out_path), np.asarray(cx8, dtype=np.float64))
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert a SmartQSM *_qsm.mat into a PyTLidar cylinder .txt "
                    "and print a summary.")
    ap.add_argument("mat", help="path to SmartQSM *_qsm.mat")
    ap.add_argument("-o", "--out", help="output .txt path (Cx8 cylinder data)")
    ap.add_argument("--offset", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="subtract this xyz vector from every start point")
    args = ap.parse_args()

    offset = np.asarray(args.offset) if args.offset else None
    cyl = smartqsm_to_cylinder(args.mat, xyz_offset=offset)
    cx8 = smartqsm_to_cx8(cyl)

    dia_cm = cyl["radius"] * 2.0 * 100.0
    in_band = np.mean((dia_cm >= 1.0) & (dia_cm <= 5.0)) * 100.0
    print(f"cylinders        : {cx8.shape[0]}")
    print(f"diameter cm      : min {dia_cm.min():.2f}  "
          f"median {np.median(dia_cm):.2f}  max {dia_cm.max():.2f}")
    print(f"in 1-5 cm band   : {in_band:.1f}%")
    print(f"branch orders    : {int(cyl['BranchOrder'].min())}.."
          f"{int(cyl['BranchOrder'].max())}"
          if "BranchOrder" in cyl else "")

    if args.out:
        p = write_cylinder_txt(cx8, args.out)
        print(f"wrote            : {p}")
