"""
Regression tests for ``Ecomodel.recombine_tiles``.

These cover the empty-tile contract documented in
``Docs/RESEARCH_DIRECTION.md`` section 3.4: ``segment_trees`` can
legitimately produce sub-tiles with zero-row clouds when no valid tree
segments are found in a sub-cube.  ``recombine_tiles`` must
(a) drop those empty sub-tiles from the concatenation so the combined
cloud's min/max calls do not crash, and (b) raise a clear, actionable
``RuntimeError`` when *every* sub-tile is empty - instead of the
cryptic ``zero-size array to reduction operation minimum`` numpy error
that originally motivated this fix.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from ecomodel import Ecomodel, Tile


def _make_empty_tile() -> Tile:
    """
    Build a ``Tile`` whose per-point arrays are all zero-row.

    ``Tile.__init__`` calls ``np.min`` on the cloud, which crashes on a
    length-0 array, so we initialise with a single placeholder point and
    then overwrite the per-point arrays to zero rows - mirroring exactly
    what ``segment_trees`` does to a sub-tile that contains no valid
    tree segments.
    """
    placeholder = np.zeros((1, 3), dtype=np.float64)
    placeholder_data = np.zeros((1, 4), dtype=np.float64)
    tile = Tile(placeholder, placeholder_data, contains_ground=False)

    tile.cloud          = np.zeros((0, 3), dtype=np.float64)
    tile.point_data     = np.zeros((0, 4), dtype=np.float64)
    tile.cover_sets     = np.zeros((0,),   dtype=np.float64)
    tile.cluster_labels = np.zeros((0,),   dtype=np.float64)
    tile.segment_labels = np.zeros((0,),   dtype=np.float64)
    tile.trunk_points   = np.zeros((0,),   dtype=np.float64)
    return tile


def _make_eco_with_tiles(tiles) -> Ecomodel:
    eco = Ecomodel(results_folder="results")
    eco.mean = np.zeros(3, dtype=np.float64)
    eco.tiles = np.array([tiles], dtype=object)
    return eco


def test_recombine_tiles_raises_clear_error_on_all_empty():
    """
    When every sub-tile is empty, ``recombine_tiles`` must raise a
    ``RuntimeError`` mentioning the empty-tile count and offering
    remediation guidance - not a numpy ``zero-size array`` traceback.
    """
    eco = _make_eco_with_tiles([_make_empty_tile(), _make_empty_tile()])

    with pytest.raises(RuntimeError) as excinfo:
        eco.recombine_tiles()

    msg = str(excinfo.value)
    assert "empty" in msg.lower(), f"expected 'empty' in message, got: {msg!r}"
    # Remediation hint must be present so the lab user knows what to try.
    assert "cube_size" in msg or "base_height" in msg, (
        f"expected remediation hint mentioning cube_size or base_height, "
        f"got: {msg!r}"
    )


def test_recombine_tiles_filters_empty_tiles_among_valid_ones():
    """
    Mixed case: one empty sub-tile, one with real points.  The empty
    tile is skipped silently (with a logged warning - not under test
    here); the combined cloud comes from the valid tile alone.
    """
    placeholder = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    placeholder_data = np.array([[1.0, 2.0, 3.0, 100.0]], dtype=np.float64)
    valid_tile = Tile(placeholder, placeholder_data, contains_ground=False)
    empty_tile = _make_empty_tile()

    eco = _make_eco_with_tiles([empty_tile, valid_tile])
    eco.recombine_tiles()

    assert eco._raw_tiles, "recombine_tiles should populate _raw_tiles"
    combined = eco._raw_tiles[0]
    assert len(combined.cloud) == 1, (
        f"expected the single valid point to survive, got {len(combined.cloud)}"
    )
    # min_x / max_x derived from the one valid point (plus eco.mean which is 0).
    assert eco.min_x == pytest.approx(1.0)
    assert eco.min_y == pytest.approx(2.0)
    assert eco.min_z == pytest.approx(3.0)
