"""
Regression tests for the empty-input guard in ``TreeQSMSteps.cover_sets.cover_sets``.

An empty point cloud (e.g. an empty subdivision cube, or a tile fully removed
by an intensity/wood mask) used to reach ``cubical_partition``'s
``np.min(P, axis=0)`` and raise the cryptic
``ValueError: zero-size array to reduction operation minimum which has no
identity`` - the same crash class that motivated the recombine_tiles fix
(see ``tests/test_recombine_tiles.py``).

``cover_sets`` now short-circuits on zero-row input and returns the same
empty-cover shape that ``create_cover`` produces for the no-balls case, so
every caller's existing ``len(cover['sets']) == 0`` check skips the tile
gracefully.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from TreeQSMSteps.cover_sets import cover_sets


_INPUTS = {"PatchDiam1": 0.15, "BallRad1": 0.15, "nmin1": 25}


def test_cover_sets_empty_input_does_not_raise():
    """Zero-row input must not raise the zero-size-array reduction error."""
    empty = np.zeros((0, 3), dtype=np.float64)
    cover = cover_sets(empty, _INPUTS, qsm=False, device="cpu")

    # The contract every caller relies on: an empty 'sets' array.
    assert "sets" in cover
    assert len(cover["sets"]) == 0


def test_cover_sets_empty_input_shape_matches_caller_expectations():
    """The empty cover must carry the keys callers read before skipping."""
    empty = np.zeros((0, 3), dtype=np.float64)
    cover = cover_sets(empty, _INPUTS, qsm=False, device="cpu")

    assert cover["ball"] == []
    assert isinstance(cover["center"], np.ndarray)
    assert cover["center"].size == 0
    assert isinstance(cover["sets"], np.ndarray)
    assert cover["sets"].size == 0


def test_cover_sets_nonempty_input_still_produces_sets():
    """Guard must not short-circuit a real (non-empty) cloud."""
    rng = np.random.Generator(np.random.Philox(0))
    # A small dense blob so at least one BallRad1 ball reaches nmin1 points.
    cloud = rng.normal(scale=0.05, size=(400, 3)).astype(np.float64)
    cover = cover_sets(cloud, _INPUTS, qsm=False, device="cpu")

    assert len(cover["sets"]) == len(cloud)
    # At least one cover set should have formed on a dense blob.
    assert (cover["sets"] > -1).any()
