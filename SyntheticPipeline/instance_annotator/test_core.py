"""Regression checks for the annotator core and web API (pytest)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SP))

from instance_annotator.io import _as_path, load_tile_prefix, save_tile
from instance_annotator.labels import LabelEditor, compact_ids
from instance_annotator.lod import build_lod
from instance_annotator.viz import material_colors, selection_from_click
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
    assert new_id == 114
    ed.mark_nontree(m)
    assert np.all(ed.labels[m] == -1)


def test_diff_undo_roundtrip():
    rng = np.random.default_rng(0)
    base = rng.integers(-1, 20, size=10_000).astype(np.int32)
    ed = LabelEditor(base)
    states = [ed.labels.copy()]
    for _ in range(25):
        mask = rng.random(len(base)) < 0.05
        ed.reassign(mask, int(rng.integers(-1, 30)))
        states.append(ed.labels.copy())
    ed.compact()
    states.append(ed.labels.copy())
    for s in reversed(states[:-1]):
        assert ed.undo()
        assert np.array_equal(ed.labels, s)
    assert not ed.undo()
    for s in states[1:]:
        assert ed.redo()
        assert np.array_equal(ed.labels, s)
    # Undo stores diffs, not full copies.
    assert ed.undo_bytes() < 25 * base.nbytes


def test_undo_memory_cap():
    ed = LabelEditor(n_points=100_000, max_undo_bytes=1_000_000)
    for k in range(50):
        mask = np.zeros(100_000, dtype=bool)
        mask[:50_000] = True
        ed.reassign(mask, k)  # 50k changes * 16 B = 800 kB per step
    assert ed.undo_bytes() <= 1_000_000 or len(ed._undo) == 1
    assert len(ed._undo) <= 2


def test_noop_edits():
    ed = LabelEditor(np.array([0, 0, 1, 1, -1], dtype=np.int32))
    assert ed.compact() == 0
    assert not ed.can_undo and not ed.dirty
    assert ed.merge(1, 1) == 0
    assert ed.reassign(np.array([True, True, False, False, False]), 0) == 0
    assert not ed.can_undo
    assert ed.reassign(np.array([True, False, True, False, False]), 1) == 1
    assert ed.can_undo and ed.dirty


def test_lod_expand():
    rng = np.random.default_rng(1)
    xyz = rng.random((200_000, 3)) * np.array([30.0, 30.0, 15.0])
    lod = build_lod(xyz, 20_000)
    m = len(lod.display_idx)
    assert 0 < m <= 20_000
    assert np.array_equal(lod.rep_of_full[lod.display_idx], np.arange(m))
    # Every full point maps to a representative in the same voxel.
    vs = lod.voxel_size
    vox = np.floor((xyz - xyz.min(axis=0)) / vs).astype(np.int64)
    assert np.array_equal(vox, vox[lod.display_idx[lod.rep_of_full]])
    sel = np.array([0, 5, 17])
    full = lod.expand(sel)
    assert np.array_equal(np.flatnonzero(full), np.flatnonzero(np.isin(lod.rep_of_full, sel)))
    ident = build_lod(xyz[:1000], 5000)
    assert ident.is_identity and np.array_equal(ident.expand(np.array([3])), np.arange(1000) == 3)


def test_wood_leaf_intensity():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(500, 3))
    inten = np.linspace(0.0, 1.0, 500)
    wood, leaf = classify_intensity_threshold(xyz, inten, 0.5)
    assert np.all(wood == ~leaf) and int(wood.sum()) == 250
    wood_p, leaf_p = classify_intensity_percentile(xyz, inten, percentile=40.0)
    assert np.all(wood_p | leaf_p) and not np.any(wood_p & leaf_p)
    counts = mask_counts(wood_p, leaf_p)
    assert counts["wood"] + counts["leaf"] + counts["unknown"] == 500
    wood2, leaf2 = classify_wood_leaf("percentile", xyz, inten, percentile=40.0)
    assert np.array_equal(wood2, wood_p) and np.array_equal(leaf2, leaf_p)
    wood_o, leaf_o = classify_otsu(xyz, inten)
    assert np.all(wood_o == ~leaf_o)
    rgb = material_colors(wood_p, leaf_p)
    assert np.allclose(rgb[wood_p][0], np.array([139 / 255.0, 90 / 255.0, 43 / 255.0]))


def test_wood_leaf_geometry():
    rng = np.random.default_rng(1)
    n_stem, n_leaf = 800, 1200
    th = rng.uniform(0, 2 * np.pi, n_stem)
    z = rng.uniform(0, 8, n_stem)
    r = 0.12 + rng.normal(0, 0.01, n_stem)
    stem = np.column_stack([r * np.cos(th), r * np.sin(th), z])
    leaf = rng.normal(size=(n_leaf, 3)) * np.array([1.5, 1.5, 1.0]) + np.array([0, 0, 7])
    xyz = np.vstack([stem, leaf])
    inten = np.concatenate([np.full(n_stem, 0.9), rng.random(n_leaf) * 0.2])
    w_e, _ = classify_eigen(xyz, inten, max_points=5000)
    assert int(w_e[:n_stem].sum()) > 50
    w_s, _ = classify_stem_grow(xyz, inten, max_points=5000, grow_radius=0.4)
    assert int(w_s[:n_stem].sum()) > 50


def test_pick_radius():
    tile = load_tile_prefix("testdataset/real_instance/l1w_t00_03")
    mask = selection_from_click(tile["xyz"], 0, 0.3)
    assert mask.dtype == bool and mask[0]


def test_save_roundtrip():
    out = Path(_SP / "output" / "annotator_demos" / "_bugfix_smoke")
    xyz = np.random.randn(1000, 3)
    labels = np.full(1000, -1, dtype=np.int32)
    labels[:100] = 0
    labels[100:250] = 1
    paths = save_tile(out, "bugfix", xyz, np.full(1000, 0.4), labels, meta={"demo": "bugfix"})
    assert Path(paths["laz"]).exists()
    tile = load_tile_prefix(out / "bugfix")
    assert len(tile["xyz"]) == 1000 and tile["labels"].max() == 1


# ── web API ─────────────────────────────────────────────────────────────────
def _client(tmp_path, n=3000, max_display=None):
    from fastapi.testclient import TestClient

    from instance_annotator.server import routes, session as sess_mod
    from instance_annotator.server.main import create_app

    sess_mod.WORKDIR = tmp_path / "work"
    rng = np.random.default_rng(2)
    xyz = rng.random((n, 3)) * np.array([10.0, 10.0, 5.0])
    labels = np.full(n, -1, dtype=np.int32)
    labels[: n // 3] = 4
    labels[n // 3 : n // 2] = 9
    tile_dir = tmp_path / "tiles"
    save_tile(tile_dir, "api", xyz, np.full(n, 0.5), labels, write_preview_ply=False)
    routes.session = sess_mod.Session()
    client = TestClient(create_app())
    body = {"path": str(tile_dir / "api"), "mode": "gt"}
    if max_display:
        body["max_display"] = max_display
    r = client.post("/api/load", json=body)
    assert r.status_code == 200, r.text
    return client, r.json(), labels


def _u32(idx):
    return np.asarray(idx, dtype="<u4").tobytes()


def test_api_load_edit_undo_save(tmp_path):
    client, info, labels = _client(tmp_path)
    n = len(labels)
    assert info["num_points"] == n and info["num_trees"] == 2
    pts = np.frombuffer(client.get("/api/points").content, dtype="<f4")
    assert len(pts) == 3 * info["num_display"]
    lab = np.frombuffer(client.get("/api/labels").content, dtype="<i4")
    assert np.array_equal(lab, labels)
    trees = client.get("/api/trees").json()["trees"]
    assert [t["id"] for t in trees] == [4, 9] and trees[0]["count"] == n // 3

    r = client.post("/api/edit/new", content=_u32(range(10))).json()
    assert r["new_id"] == 10 and r["changed_points"] == 10
    assert sorted(r["display_idx"]) == list(range(10)) and set(r["display_labels"]) == {10}
    assert r["info"]["dirty"] and r["info"]["can_undo"]

    r = client.post("/api/edit/merge?target=4", content=_u32([n // 3 + 1])).json()
    assert r["merged"] == [9]
    assert [t["id"] for t in client.get("/api/trees").json()["trees"]] == [4, 10]

    assert client.post("/api/edit/assign", content=_u32([])).status_code == 400
    client.post("/api/undo")
    client.post("/api/undo")
    lab = np.frombuffer(client.get("/api/labels").content, dtype="<i4")
    assert np.array_equal(lab, labels)
    assert client.post("/api/undo").status_code == 400

    client.post("/api/redo")
    client.post("/api/review", json={"tree_id": 10, "reviewed": True})
    out = tmp_path / "out"
    r = client.post("/api/save", json={"out_dir": str(out), "name": "saved"})
    assert r.status_code == 200, r.text
    assert client.post("/api/save", json={"out_dir": str(out), "name": "saved"}).status_code == 409
    assert client.post("/api/save", json={"out_dir": str(out), "name": "saved", "overwrite": True}).status_code == 200
    tile = load_tile_prefix(out / "saved")
    assert int((tile["labels"] == 10).sum()) == 10
    assert tile["meta"]["reviewed_trees"] == [10]
    assert not client.get("/api/status").json()["dirty"]


def test_api_lod_edit_is_full_res(tmp_path):
    client, info, labels = _client(tmp_path, n=50_000, max_display=5_000)
    assert info["num_display"] <= 5_000 < info["num_points"]
    from instance_annotator.server import routes

    sess = routes.session
    r = client.post("/api/edit/nontree", content=_u32([0, 1, 2])).json()
    expect = np.isin(sess.lod.rep_of_full, [0, 1, 2]) & (labels != -1)
    assert r["changed_points"] == int(expect.sum())
    assert np.all(sess.editor.labels[np.isin(sess.lod.rep_of_full, [0, 1, 2])] == -1)


def test_api_grow_and_autosave(tmp_path):
    client, info, labels = _client(tmp_path)
    grown = np.frombuffer(client.post("/api/grow?radius=2.0&same_label=true", content=_u32([0])).content, dtype="<u4")
    assert 0 in grown and np.all(labels[grown] == labels[0])
    from instance_annotator.server import routes

    for k in range(10):
        client.post("/api/edit/assign?target=20", content=_u32([k]))
    assert routes.session.autosave_path().exists()
    assert client.get("/api/status").json()["autosave"] is not None
    r = client.post("/api/autosave/restore")
    assert r.status_code == 200
    client.post("/api/autosave/discard")
    assert client.get("/api/status").json()["autosave"] is None


def test_api_woodleaf_job(tmp_path):
    import time

    client, _, _ = _client(tmp_path)
    # Fixture intensity is constant, so intensity-only methods are rejected up front.
    r = client.post("/api/woodleaf", json={"method": "percentile", "params": {"percentile": 50}})
    assert r.status_code == 400 and "intensity" in r.json()["error"]

    job = client.post("/api/woodleaf", json={"method": "eigen", "params": {}}).json()
    for _ in range(200):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["state"] != "running":
            break
        time.sleep(0.05)
    assert j["state"] == "done", j
    assert j["progress"] == 1.0 and "wood=" in j["message"]
    mat = np.frombuffer(client.get("/api/material").content, dtype=np.uint8)
    assert set(np.unique(mat).tolist()) <= {1, 2}


def test_propagate_voxel_labels():
    from instance_annotator.segment import propagate_labels

    rng = np.random.default_rng(0)
    a = rng.random((2000, 3)) * 0.5
    b = rng.random((2000, 3)) * 0.5 + [3.0, 0, 0]
    far = np.array([[10.0, 10.0, 10.0]])
    xyz = np.vstack([a, b, far])
    seeds = np.array([0, 5, 2000, 2005])
    full = propagate_labels(xyz, seeds, np.array([1, 1, 2, -1], np.int32), radius=0.6)
    assert np.all(full[:2000] == 1)
    assert set(np.unique(full[2000:4000]).tolist()) <= {2, -1} and (full[2000:4000] == 2).any()
    assert full[-1] == -1


def _wait_job(client, job_id):
    import time

    for _ in range(200):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_api_segment_failure_keeps_labels(tmp_path, monkeypatch):
    from instance_annotator import segment

    client, _, labels = _client(tmp_path)
    n = len(labels)

    def empty(xyz, inten, method, **kw):
        print("preprocessing")
        return np.full(n, -1, np.int32), {"ok": True, "num_trees": 0, "message": "trees=0 labeled_pts=10"}

    monkeypatch.setattr(segment, "run_method", empty)
    j = _wait_job(client, client.post("/api/segment", json={"method": "treex"}).json()["id"])
    assert j["state"] == "error" and "no trees found" in j["message"]
    lab = np.frombuffer(client.get("/api/labels").content, dtype="<i4")
    assert np.array_equal(lab, labels) and not client.get("/api/status").json()["can_undo"]

    def ok(xyz, inten, method, **kw):
        from tqdm import tqdm

        for _ in tqdm(range(4), desc="infer"):
            pass
        out = np.zeros(n, np.int32)
        return out, {"ok": True, "num_trees": 1, "message": "trees=1 labeled_pts=%d" % n}

    monkeypatch.setattr(segment, "run_method", ok)
    j = _wait_job(client, client.post("/api/segment", json={"method": "treex"}).json()["id"])
    assert j["state"] == "done" and j["progress"] == 1.0, j
    assert np.all(np.frombuffer(client.get("/api/labels").content, dtype="<i4") == 0)
