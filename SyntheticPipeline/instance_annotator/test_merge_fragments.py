"""Tests for merge_fragments (split-tree detection and merging)."""
import numpy as np

from instance_annotator.merge_fragments import merge_fragments


def _tree(rng, x, y, h=12.0, crown_r=2.5, n_stem=600, n_crown=3000):
    z = rng.uniform(0, h * 0.5, n_stem)
    a = rng.uniform(0, 2 * np.pi, n_stem)
    stem = np.stack([x + 0.15 * np.cos(a), y + 0.15 * np.sin(a), z], 1)
    v = rng.normal(size=(n_crown, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = crown_r * rng.uniform(0.6, 1.0, n_crown)[:, None]
    crown = np.array([x, y, h * 0.7]) + v * r * np.array([1, 1, 1.2])
    return stem, crown


def test_merges_crown_fragment_and_split_stem_but_keeps_neighbours():
    rng = np.random.default_rng(0)
    ground = np.stack([rng.uniform(-5, 25, 8000), rng.uniform(-5, 10, 8000), rng.normal(0, 0.02, 8000)], 1)
    s0, c0 = _tree(rng, 0, 0)
    s1, c1 = _tree(rng, 4.8, 0)      # crowns just touching: must stay separate
    s2, c2 = _tree(rng, 15, 0)
    upper0 = c0[:, 2] > 12.0 * 0.7 + 1.0   # top of tree 0's crown split off as id 7
    lower2 = s2[:, 2] < 2.0           # tree 2 split: low stem id 2, rest id 9
    xyz = np.vstack([ground, s0, c0, s1, c1, s2, c2])
    lab = np.concatenate([
        np.full(len(ground), -1),
        np.zeros(len(s0)), np.where(upper0, 7, 0),
        np.ones(len(s1)), np.ones(len(c1)),
        np.where(lower2, 2, 9), np.full(len(c2), 9),
    ]).astype(np.int32)

    out, rep = merge_fragments(xyz, lab)
    tree0 = out[len(ground):len(ground) + len(s0) + len(c0)]
    assert len(np.unique(tree0)) == 1, rep
    tree2 = out[-(len(s2) + len(c2)):]
    assert len(np.unique(tree2)) == 1, rep
    assert len(np.unique(out[out >= 0])) == 3, rep
    assert (out[: len(ground)] == -1).all()


def test_fragment_between_two_trees_goes_to_review():
    rng = np.random.default_rng(2)
    ground = np.stack([rng.uniform(-5, 10, 4000), rng.uniform(-5, 5, 4000), rng.normal(0, 0.02, 4000)], 1)
    s0, c0 = _tree(rng, 0, 0)
    s1, c1 = _tree(rng, 3.5, 0)
    side = c0[:, 0] > 1.0            # half-crown facing the neighbour: ambiguous owner
    xyz = np.vstack([ground, s0, c0, s1, c1])
    lab = np.concatenate([np.full(len(ground), -1), np.zeros(len(s0)), np.where(side, 7, 0),
                          np.ones(len(s1) + len(c1))]).astype(np.int32)
    out, rep = merge_fragments(xyz, lab)
    assert 7 in set(out.tolist()), rep
    assert any(c["b"] == 7 and c["reason"].startswith("ambiguous") for c in rep["candidates"]), rep


def test_no_labels_is_noop():
    xyz = np.random.default_rng(1).uniform(0, 5, (100, 3))
    lab = np.full(100, -1, np.int32)
    out, rep = merge_fragments(xyz, lab)
    assert (out == lab).all() and rep["merged"] == 0
