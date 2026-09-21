"""Quick regression checks for annotator core (no Streamlit UI)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SP))

from instance_annotator.io import _as_path, load_tile_prefix, save_tile
from instance_annotator.labels import LabelEditor, compact_ids
from instance_annotator.viz import (
    build_plotly_figure,
    downsample_indices,
    material_colors,
    parse_plotly_point_indices,
    selection_from_click,
)
from instance_annotator.wood_leaf import (
    classify_eigen,
    classify_intensity_percentile,
    classify_intensity_threshold,
    classify_otsu,
    classify_stem_grow,
    classify_wood_leaf,
    mask_counts,
)


def test_path_resolve():
    p = _as_path("testdataset/real_instance/l1w_t00_03")
    assert Path(str(p) + "_scan.laz").exists(), p
    print("path ok", p)


def test_labels():
    lab = np.array([-1, 10, 10, 113, -1, 113], dtype=np.int32)
    c = compact_ids(lab)
    assert set(c[c >= 0].tolist()) == {0, 1}
    ed = LabelEditor(lab)
    ed.merge(10, 113)
    assert int((ed.labels == 10).sum()) == 0
    ed.undo()
    assert int((ed.labels == 10).sum()) == 2
    m = np.array([True, True, False, False, False, False])
    new_id = ed.paint_new(m)
    assert new_id >= 0
    ed.mark_nontree(m)
    assert np.all(ed.labels[m] == -1)
    print("labels ok")


def test_wood_leaf_intensity():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(500, 3))
    inten = np.linspace(0.0, 1.0, 500)
    wood, leaf = classify_intensity_threshold(xyz, inten, 0.5)
    assert len(wood) == 500 and len(leaf) == 500
    assert wood.dtype == bool and leaf.dtype == bool
    assert np.all(wood == ~leaf)
    assert int(wood.sum()) == 250

    wood_p, leaf_p = classify_intensity_percentile(xyz, inten, percentile=40.0)
    assert len(wood_p) == 500
    assert np.all(wood_p | leaf_p)
    assert not np.any(wood_p & leaf_p)
    counts = mask_counts(wood_p, leaf_p)
    assert counts["wood"] + counts["leaf"] + counts["unknown"] == 500

    wood2, leaf2 = classify_wood_leaf("percentile", xyz, inten, percentile=40.0)
    assert np.array_equal(wood2, wood_p) and np.array_equal(leaf2, leaf_p)

    wood_o, leaf_o = classify_otsu(xyz, inten)
    assert len(wood_o) == 500 and np.all(wood_o == ~leaf_o)

    # Material colors length + wood tint
    rgb = material_colors(wood_p, leaf_p)
    assert rgb.shape == (500, 3)
    assert np.allclose(rgb[wood_p][0], np.array([139 / 255.0, 90 / 255.0, 43 / 255.0]))
    print("wood/leaf intensity ok", counts)


def test_wood_leaf_geometry():
    rng = np.random.default_rng(1)
    # Vertical stem cylinder + scattered canopy
    n_stem, n_leaf = 800, 1200
    th = rng.uniform(0, 2 * np.pi, n_stem)
    z = rng.uniform(0, 8, n_stem)
    r = 0.12 + rng.normal(0, 0.01, n_stem)
    stem = np.column_stack([r * np.cos(th), r * np.sin(th), z])
    leaf = rng.normal(size=(n_leaf, 3)) * np.array([1.5, 1.5, 1.0]) + np.array([0, 0, 7])
    xyz = np.vstack([stem, leaf])
    inten = np.concatenate([np.full(n_stem, 0.9), rng.random(n_leaf) * 0.2])

    w_e, l_e = classify_eigen(xyz, inten, max_points=5000)
    assert len(w_e) == len(xyz) and w_e.dtype == bool
    # Stem should retain a non-trivial wood fraction
    assert int(w_e[:n_stem].sum()) > 50

    w_s, l_s = classify_stem_grow(xyz, inten, max_points=5000, grow_radius=0.4)
    assert len(w_s) == len(xyz)
    assert int(w_s[:n_stem].sum()) > 50
    print("wood/leaf geometry ok", int(w_e.sum()), int(w_s.sum()))



def test_viz_material_filter():
    xyz = np.random.randn(200, 3)
    labels = np.full(200, 0, dtype=np.int32)
    labels[100:] = -1
    inten = np.linspace(0, 1, 200)
    wood, leaf = classify_intensity_threshold(xyz, inten, 0.5)
    idx = downsample_indices(200, 200)
    fig, _ = build_plotly_figure(
        xyz,
        labels,
        idx,
        wood_mask=wood,
        leaf_mask=leaf,
        show_wood=True,
        show_leaf=False,
        color_mode="material",
        hide_nontree=False,
    )
    assert fig.data and len(fig.data[0].x) == int((~leaf).sum())
    print("viz material filter ok")


def test_viz_and_pick():
    tile = load_tile_prefix("testdataset/real_instance/l1w_t00_03")
    idx = downsample_indices(len(tile["xyz"]), 5_000)
    fig, origin = build_plotly_figure(
        tile["xyz"],
        tile["labels"],
        idx,
        highlight_id=int(tile["labels"][tile["labels"] >= 0][0]),
        selection_mask_full=(tile["labels"] == tile["labels"][0]),
        hide_nontree=False,
    )
    assert fig.data and len(fig.data[0].x) > 0
    # Fake streamlit-like event
    class Ev:
        class Sel:
            points = [
                {"customdata": [int(idx[0]), 1], "point_index": 0},
                {"point_index": 1},
            ]

        selection = Sel()

    full = parse_plotly_point_indices(Ev(), idx)
    assert full[0] == int(idx[0])
    assert full[1] == int(idx[1])
    mask = selection_from_click(tile["xyz"], int(idx[0]), 0.3)
    assert mask.dtype == bool and mask.sum() >= 1
    print("viz/pick ok", "origin", origin)


def test_save_roundtrip(tmp_path: Path | None = None):
    out = Path(_SP / "output" / "annotator_demos" / "_bugfix_smoke")
    out.mkdir(parents=True, exist_ok=True)
    xyz = np.random.randn(1000, 3)
    inten = np.full(1000, 0.4)
    labels = np.full(1000, -1, dtype=np.int32)
    labels[:100] = 0
    labels[100:250] = 1
    paths = save_tile(out, "bugfix", xyz, inten, labels, meta={"demo": "bugfix"})
    assert Path(paths["laz"]).exists()
    tile = load_tile_prefix(out / "bugfix")
    assert len(tile["xyz"]) == 1000
    assert tile["labels"] is not None and tile["labels"].max() == 1
    print("save ok", paths["laz"])


if __name__ == "__main__":
    test_path_resolve()
    test_labels()
    test_wood_leaf_intensity()
    test_wood_leaf_geometry()
    test_viz_material_filter()
    test_viz_and_pick()
    test_save_roundtrip()
    print("ALL PASS")
