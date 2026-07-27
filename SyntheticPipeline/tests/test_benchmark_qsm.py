import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "SyntheticPipeline" / "scripts"
for path in (ROOT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gui.smartqsm_runner import _parse_qsm_mat
from gui.adtree_runner import parse_skeleton_ply_to_cx9
from gui.archi_runner import parse_archi_csv_to_cx9
from benchmark_qsm import (
    _sleep_worker,
    _world_cylinders,
    calculate_metrics,
    evaluate_abstraction_vs_hires,
    load_gt_cylinders,
    point_to_cylinder_distances,
    run_treeqsm,
    sample_cylinders,
)


def _cylinder_row():
    return np.array([[0, 0, 0, 0.1, 0, 0, 1, 1.0, 0]], dtype=float)


def test_surface_sampling_is_deterministic():
    first = sample_cylinders(_cylinder_row(), 200, np.random.default_rng(7))
    second = sample_cylinders(_cylinder_row(), 200, np.random.default_rng(7))
    np.testing.assert_allclose(first, second)


def test_identical_geometry_has_perfect_voxel_metrics():
    points = sample_cylinders(_cylinder_row(), 200, np.random.default_rng(3))
    assert calculate_metrics(points, points, 0.1) == (1.0, 1.0, 1.0, 1.0)


def test_world_coordinate_round_trip_only_translates_starts():
    cylinders = _cylinder_row()
    shifted = _world_cylinders(cylinders, np.array([10.0, -4.0, 2.0]))
    np.testing.assert_allclose(shifted[0, :3], [10.0, -4.0, 2.0])
    np.testing.assert_allclose(shifted[0, 3:], cylinders[0, 3:])


@pytest.mark.parametrize("columns", [9, 10])
def test_load_gt_accepts_legacy_and_current_formats(tmp_path, columns):
    path = tmp_path / "gt.txt"
    np.savetxt(path, np.ones((2, columns)))
    assert load_gt_cylinders(path).shape == (2, columns)


def test_load_gt_rejects_unknown_format(tmp_path):
    path = tmp_path / "gt.txt"
    np.savetxt(path, np.ones((2, 8)))
    with pytest.raises(ValueError):
        load_gt_cylinders(path)


def test_smartqsm_parser_returns_branch_order_cx9(tmp_path):
    path = tmp_path / "qsm.mat"
    sio.savemat(path, {
        "cylinder": {
            "start": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            "radius": np.array([0.2, 0.1]),
            "axis": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            "length": np.array([1.0, 0.5]),
            "BranchOrder": np.array([0, 2]),
            "branch": np.array([8, 9]),
        }
    })
    cylinders = _parse_qsm_mat(path)
    assert cylinders.shape == (2, 9)
    np.testing.assert_array_equal(cylinders[:, 8], [0, 2])


def test_worker_timeout_returns_without_waiting_for_probe():
    started = __import__("time").time()
    _, status, _, _, _ = run_treeqsm(
        np.empty((0, 3)),
        {"sleep": 2.0},
        timeout=0.1,
        memory_limit_gb=15.0,
        worker_target=_sleep_worker,
    )
    assert status == "timeout"
    assert __import__("time").time() - started < 5.0


def test_point_to_cylinder_distance_is_zero_on_surface():
    cylinder = _cylinder_row()
    surface = sample_cylinders(cylinder, 64, np.random.default_rng(1))
    dist = point_to_cylinder_distances(surface, cylinder)
    assert float(np.max(dist)) < 1e-6


def test_abstraction_vs_hires_is_near_perfect_for_identical_geometry():
    class Args:
        surface_samples = 2000
        distance_tolerance = 0.05

    cylinder = _cylinder_row()
    hires = sample_cylinders(cylinder, 2000, np.random.default_rng(2))
    metrics = evaluate_abstraction_vs_hires(hires, cylinder, seed=3, args=Args())
    assert metrics["Whole_Precision"] > 0.99
    assert metrics["Whole_Recall"] > 0.99
    assert metrics["Whole_F1"] > 0.99
    assert metrics["Whole_MedianDist_m"] < 1e-3


def test_abstraction_penalizes_offset_cylinders():
    class Args:
        surface_samples = 2000
        distance_tolerance = 0.05

    cylinder = _cylinder_row()
    hires = sample_cylinders(cylinder, 2000, np.random.default_rng(4))
    offset = cylinder.copy()
    offset[0, 0] += 0.5
    metrics = evaluate_abstraction_vs_hires(hires, offset, seed=5, args=Args())
    assert metrics["Whole_F1"] < 0.2
    assert metrics["Whole_MedianDist_m"] > 0.2


def _write_ascii_skeleton_ply(path):
    # Vertical trunk + one side branch; radii 0.2 and 0.1.
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property float radius",
                "element edge 2",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0 0.2",
                "0 0 1 0.2",
                "1 0 1 0.1",
                "2 0 1",
                "2 1 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_adtree_skeleton_ply_parses_to_cx9(tmp_path):
    ply_path = tmp_path / "tree_skeleton.ply"
    _write_ascii_skeleton_ply(ply_path)
    cylinders = parse_skeleton_ply_to_cx9(ply_path)
    assert cylinders.shape == (2, 9)
    np.testing.assert_allclose(cylinders[0, :3], [0, 0, 0])
    np.testing.assert_allclose(cylinders[0, 3], 0.2)
    np.testing.assert_allclose(cylinders[0, 4:7], [0, 0, 1])
    np.testing.assert_allclose(cylinders[0, 7], 1.0)
    np.testing.assert_allclose(cylinders[1, 3], 0.15)  # mean(0.2, 0.1)
    assert cylinders[0, 8] == 0
    assert cylinders[1, 8] == 1


def test_archi_csv_parses_to_cx9(tmp_path):
    csv_path = tmp_path / "qsm.csv"
    csv_path.write_text(
        "\n".join(
            [
                "startX,startY,startZ,endX,endY,endZ,radius_cyl,length,branching_order",
                "0,0,0,0,0,2,0.25,2,1",
                "0,0,2,1,0,2,0.1,1,2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cylinders = parse_archi_csv_to_cx9(csv_path)
    assert cylinders.shape == (2, 9)
    np.testing.assert_allclose(cylinders[0, :3], [0, 0, 0])
    np.testing.assert_allclose(cylinders[0, 3], 0.25)
    np.testing.assert_allclose(cylinders[0, 4:7], [0, 0, 1])
    np.testing.assert_allclose(cylinders[0, 7], 2.0)
    np.testing.assert_allclose(cylinders[0, 8], 1)
    np.testing.assert_allclose(cylinders[1, 4:7], [1, 0, 0])
