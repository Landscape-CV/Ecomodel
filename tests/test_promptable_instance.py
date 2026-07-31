"""Tests for promptable instance helpers (no Point-SAM/SNAP install required)."""
import numpy as np
from Utils.promptable_instance import (
    auto_stem_grid_prompts,
    masks_to_instance_labels,
    oracle_stem_prompts,
)


def test_masks_to_instance_labels_nms():
    n = 100
    m0 = np.zeros(n, dtype=bool)
    m0[:40] = True
    m1 = np.zeros(n, dtype=bool)
    m1[20:60] = True  # overlaps m0
    m2 = np.zeros(n, dtype=bool)
    m2[70:95] = True
    labels = masks_to_instance_labels([m0, m1, m2], scores=[1.0, 0.5, 0.9], nms_iou=0.3, min_points=10)
    assert labels.dtype == np.int32
    assert set(np.unique(labels)).issubset({-1, 0, 1})
    assert (labels[70:95] >= 0).all()


def test_oracle_stem_prompts():
    xyz = np.zeros((60, 3), dtype=float)
    xyz[:30, 0] = 0
    xyz[:30, 2] = np.linspace(0, 10, 30)
    xyz[30:, 0] = 5
    xyz[30:, 2] = np.linspace(0, 10, 30)
    gt = np.array([0] * 30 + [1] * 30)
    prompts = oracle_stem_prompts(xyz, gt)
    assert len(prompts) == 2
    assert prompts[0][0] < prompts[1][0]


def test_auto_stem_grid_prompts():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(500, 3))
    xyz[:, 2] = np.abs(xyz[:, 2])
    prompts = auto_stem_grid_prompts(xyz, grid_spacing=2.0, min_points_per_cell=5)
    assert len(prompts) >= 1
    assert prompts[0].shape == (3,)
