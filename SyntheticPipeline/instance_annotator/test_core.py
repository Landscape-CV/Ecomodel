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
    parse_plotly_point_indices,
    selection_from_click,
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
    test_viz_and_pick()
    test_save_roundtrip()
    print("ALL PASS")
