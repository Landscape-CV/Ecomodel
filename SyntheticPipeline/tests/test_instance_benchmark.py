"""Unit tests for multi-tree instance GT plumbing (no Arbaro / Open3D required)."""
import json
import numpy as np
import pytest
import trimesh

from pipeline.forest_assembler import ForestAssembler, GROUND_INSTANCE_ID
from evaluation.instance_metrics import match_instances, pairwise_iou_matrix


def _write_dummy_tree(asset_dir, stem: str, tree_id_hint: int = 0):
    """Create a simple upright box tree + cylinder GT JSON."""
    mesh = trimesh.creation.box(extents=[0.4, 0.4, 2.0])
    mesh.apply_translation([0, 0, 1.0])  # sit on z=0
    obj_path = asset_dir / f"{stem}.obj"
    mesh.export(str(obj_path))

    noleaf = mesh.copy()
    noleaf.export(str(asset_dir / f"{stem}_noleaf.obj"))

    gt = [{
        "tree_instance_id": 0,
        "branch_id": 0,
        "start": [0.0, 0.0, 0.0],
        "end": [0.0, 0.0, 2.0],
        "radius": 0.1,
        "length": 2.0,
        "axis": [0.0, 0.0, 1.0],
    }]
    with open(asset_dir / f"{stem}.json", "w") as f:
        json.dump(gt, f)
    return obj_path


def test_assembler_face_tree_ids_and_spacing(tmp_path):
    asset_dir = tmp_path / "assets"
    out_dir = tmp_path / "scenes"
    asset_dir.mkdir()
    _write_dummy_tree(asset_dir, "aspen_0000")
    _write_dummy_tree(asset_dir, "oak_0000")

    assembler = ForestAssembler(asset_dir=str(asset_dir), output_dir=str(out_dir))
    meta = assembler.generate_scene(
        scene_name="test_multi",
        area_size=(20.0, 20.0),
        num_trees=3,
        wind_vector=(0.0, 0.0, 0.0),
        species=None,
        min_spacing=0.5,
        max_spacing=8.0,
        placement_mode="cluster",
    )

    assert meta is not None
    assert meta["num_trees"] == 3
    assert len(meta["species_list"]) == 3
    assert "nn_distance_stats" in meta

    face_ids = np.load(out_dir / "test_multi_face_tree_ids.npy")
    mesh = trimesh.load(str(out_dir / "test_multi.ply"), force="mesh")
    assert len(face_ids) == len(mesh.faces)
    assert np.any(face_ids == GROUND_INSTANCE_ID)
    for tid in range(3):
        assert np.any(face_ids == tid)

    assert (out_dir / "test_multi_gt.json").exists()
    assert (out_dir / "test_multi_gt_mesh.ply").exists()


def test_assembler_mono_species_filter(tmp_path):
    asset_dir = tmp_path / "assets"
    out_dir = tmp_path / "scenes"
    asset_dir.mkdir()
    _write_dummy_tree(asset_dir, "aspen_0000")
    _write_dummy_tree(asset_dir, "oak_0000")

    assembler = ForestAssembler(asset_dir=str(asset_dir), output_dir=str(out_dir))
    meta = assembler.generate_scene(
        scene_name="mono",
        area_size=(20.0, 20.0),
        num_trees=2,
        species="aspen",
        placement_mode="spread",
        min_spacing=5.0,
    )
    assert meta is not None
    assert all(s == "aspen" for s in meta["species_list"])


def test_hungarian_instance_metrics_perfect():
    gt = np.array([0, 0, 0, 1, 1, 1, -1, -1], dtype=np.int32)
    pred = gt.copy()
    iou, gt_ids, pred_ids = pairwise_iou_matrix(gt, pred)
    assert list(gt_ids) == [0, 1]
    assert list(pred_ids) == [0, 1]
    metrics = match_instances(iou, iou_thresh=0.5)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["pq"] == pytest.approx(1.0)


def test_hungarian_instance_metrics_split_merge():
    gt = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    pred = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    iou, _, _ = pairwise_iou_matrix(gt, pred)
    metrics = match_instances(iou, iou_thresh=0.5)
    assert metrics["num_gt"] == 2
    assert metrics["num_pred"] == 1
    assert metrics["tp"] + metrics["fn"] == 2
    assert metrics["fp"] + metrics["tp"] == 1


def test_smoke_config_exists():
    from pathlib import Path
    cfg = Path(__file__).resolve().parents[1] / "configs" / "instance_benchmark_smoke.example.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert "densities" in data
    assert "sparse" in data["densities"]
    assert data.get("replicates") == 1
    assert data.get("max_species") == 2
