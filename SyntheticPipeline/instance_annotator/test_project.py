"""Island project tests on a synthetic 2x2 tile grid with a tree straddling the shared corner.

Run: python -m pytest instance_annotator/test_project.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SP = Path(__file__).resolve().parents[1]
if str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))

from instance_annotator.project.build import build_project, import_labels  # noqa: E402
from instance_annotator.project.store import Project  # noqa: E402

ORIGIN = np.array([500000.0, 3000000.0, 10.0])
TILE = 20.0
OVERLAP = 0.5          # tiles share a 0.5 m strip (points duplicated in both files)
# (x, y) stem positions; tree 0 sits on the corner shared by all four tiles.
TREES = [(20.0, 20.0), (6.0, 7.0), (33.0, 8.0), (8.0, 31.0), (31.0, 33.0), (18.6, 9.0)]


def _synthetic_cloud(seed: int = 0):
    rng = np.random.default_rng(seed)
    g = np.arange(0.05, 2 * TILE, 0.25)
    gx, gy = np.meshgrid(g, g)
    ground = np.stack([gx.ravel(), gy.ravel(), rng.normal(0, 0.02, gx.size)], axis=1)
    pts, lab = [ground], [np.full(len(ground), -1)]
    for i, (x, y) in enumerate(TREES):
        n_stem, n_crown = 3000, 6000
        a = rng.uniform(0, 2 * np.pi, n_stem)
        z = rng.uniform(0.05, 8.0, n_stem)
        stem = np.stack([x + 0.2 * np.cos(a), y + 0.2 * np.sin(a), z], axis=1)
        d = rng.normal(size=(n_crown, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        crown = np.array([x, y, 9.5]) + d * rng.uniform(0.3, 1.8, (n_crown, 1))
        pts += [stem, crown]
        lab.append(np.full(n_stem + n_crown, i))
    return np.concatenate(pts), np.concatenate(lab)


def _write_laz(path: Path, xyz: np.ndarray, inten: np.ndarray):
    import laspy

    h = laspy.LasHeader(point_format=3, version="1.4")
    h.offsets = np.floor(xyz.min(axis=0))
    h.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = inten
    las.write(str(path))


def cc_runner(xyz: np.ndarray, inten: np.ndarray) -> np.ndarray:
    """Stand-in segmenter: connected components of above-ground voxels."""
    from scipy import ndimage

    lab = np.full(len(xyz), -1, dtype=np.int32)
    above = xyz[:, 2] - xyz[:, 2].min() > 0.3
    if not above.any():
        return lab
    v = 0.4
    ijk = np.floor((xyz[above] - xyz[above].min(axis=0)) / v).astype(np.int64)
    grid = np.zeros(ijk.max(axis=0) + 1, dtype=bool)
    grid[tuple(ijk.T)] = True
    comp, _ = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    lab[above] = comp[tuple(ijk.T)] - 1
    return lab


@pytest.fixture(scope="module")
def island(tmp_path_factory):
    root = tmp_path_factory.mktemp("island")
    src = root / "tiles"
    src.mkdir()
    xyz, truth = _synthetic_cloud()
    inten = (np.random.default_rng(1).integers(0, 65535, len(xyz))).astype(np.uint16)
    files = {}
    for i in range(2):
        for j in range(2):
            lo = np.array([i * TILE - OVERLAP, j * TILE - OVERLAP])
            hi = np.array([(i + 1) * TILE + OVERLAP, (j + 1) * TILE + OVERLAP])
            m = np.all((xyz[:, :2] >= lo) & (xyz[:, :2] < hi), axis=1)
            name = f"t{i}{j}"
            _write_laz(src / f"{name}.laz", xyz[m] + ORIGIN, inten[m])
            files[name] = (m, xyz[m], truth[m])
    proj_dir = root / "proj"
    build_project(src, proj_dir, grid=16, leaf_max=4000, workers=1, log=lambda s: None)
    return {"root": root, "proj": proj_dir, "files": files, "xyz": xyz, "truth": truth}


def _local_truth(island, name):
    import laspy

    las = laspy.read(str(island["root"] / "tiles" / f"{name}.laz"))
    return np.stack([las.x, las.y, las.z], axis=1)


def test_every_point_once_and_roundtrip(island):
    p = Project(island["proj"], writable=False)
    for t, tile in enumerate(p.tiles):
        n = tile["n"]
        seen = np.zeros(n, dtype=np.int64)
        src = _local_truth(island, tile["name"]) - p.origin
        base = int(p.tile_base[t])
        nodes = json.loads((p.root / "tiles" / f"{tile['name']}.json").read_text())["nodes"]
        err = 0.0
        for g in range(base, base + len(nodes)):
            xyz, _, idx = p.read_node(g)
            seen[idx] += 1
            err = max(err, float(np.abs(xyz - src[idx]).max()) if len(idx) else 0.0)
            assert np.all(np.asarray(p.nodeof(t))[idx] == g - base)
        assert np.all(seen == 1), "each point must be in exactly one node"
        assert err < 2e-3
    p.close()


def test_crop_matches_brute_force(island):
    p = Project(island["proj"], writable=False)
    lo, hi = np.array([12.0, 14.0]) - p.origin[:2] + ORIGIN[:2], np.array([27.0, 25.0]) - p.origin[:2] + ORIGIN[:2]
    crop = p.crop(lo, hi)
    got = set(zip(crop["tile"].tolist(), crop["idx"].tolist()))
    want = set()
    for t, tile in enumerate(p.tiles):
        loc = (_local_truth(island, tile["name"]) - p.origin).astype(np.float32)
        m = np.all((loc[:, :2] >= lo.astype(np.float32)) & (loc[:, :2] <= hi.astype(np.float32)), axis=1)
        want |= {(t, int(i)) for i in np.flatnonzero(m)}
    # Quantization can move points sitting exactly on the box edge; allow a tiny mismatch.
    assert len(got ^ want) <= max(3, len(want) // 5000)
    p.close()


def test_segment_island_no_trees_keeps_labels(island):
    from instance_annotator.project.segment_island import segment_island
    from instance_annotator.project.store import ProjectError

    p = Project(island["proj"])
    before = [np.asarray(p.labels(t)).copy() for t in range(len(p.tiles))]
    with pytest.raises(ProjectError, match="no trees"):
        segment_island(p, "none", buffer=4.0, voxel=0.1, log=lambda s: None,
                       runner=lambda xyz, inten: np.full(len(xyz), -1, np.int32))
    for t in range(len(p.tiles)):
        assert np.array_equal(np.asarray(p.labels(t)), before[t])
    p.close()


def test_segment_island_ownership_and_ids(island):
    from instance_annotator.project.segment_island import segment_island

    p = Project(island["proj"])
    before = p.peek_next_id()
    summary = segment_island(p, "cc", buffer=4.0, voxel=0.1, runner=cc_runner, log=lambda s: None)
    assert summary["trees"] == len(TREES), summary
    # Corner tree: one id in all four tiles, no duplicates from the overlap strips.
    ids_at_corner = set()
    all_ids = set()
    for t, tile in enumerate(p.tiles):
        loc = _local_truth(island, tile["name"])
        lab = np.asarray(p.labels(t))
        near = np.hypot(loc[:, 0] - (ORIGIN[0] + 20), loc[:, 1] - (ORIGIN[1] + 20)) < 0.5
        near &= loc[:, 2] - ORIGIN[2] > 1.0
        if near.any():
            vals, cnt = np.unique(lab[near], return_counts=True)
            ids_at_corner.add(int(vals[np.argmax(cnt)]))
        all_ids |= set(np.unique(lab[lab >= 0]).tolist())
    assert len(ids_at_corner) == 1
    assert len(all_ids) == len(TREES)
    assert min(all_ids) >= before and p.peek_next_id() > max(all_ids)
    assert (p.root / "labels_backup").is_dir()
    p.close()


def test_import_and_stitch_merges_split_tree(island):
    from instance_annotator.project.stitch import apply_candidates, find_candidates

    ldir = island["root"] / "per_tile_labels"
    ldir.mkdir(exist_ok=True)
    p = Project(island["proj"], writable=False)
    for tile in p.tiles:
        loc = _local_truth(island, tile["name"])
        np.save(ldir / f"{tile['name']}_instances.npy", cc_runner(loc - loc.min(axis=0), np.zeros(len(loc))))
    p.close()
    import_labels(island["proj"], ldir, log=lambda s: None)
    p = Project(island["proj"])

    def corner_ids():
        s = set()
        for t, tile in enumerate(p.tiles):
            loc = _local_truth(island, tile["name"])
            near = (np.hypot(loc[:, 0] - (ORIGIN[0] + 20), loc[:, 1] - (ORIGIN[1] + 20)) < 0.5) & (loc[:, 2] - ORIGIN[2] > 1)
            s |= set(np.unique(np.asarray(p.labels(t))[near]).tolist())
        return s - {-1}

    assert len(corner_ids()) == 4, "per-tile labels split the corner tree"
    cands = find_candidates(p)
    assert cands, "border contacts expected"
    n = apply_candidates(p, [c for c in cands if c["score"] >= 0.2], log=lambda s: None)
    assert n >= 3
    assert len(corner_ids()) == 1
    p.close()


def test_region_write_through_and_island_undo(island):
    from instance_annotator.server.project_session import ProjectSession

    import time

    ps = ProjectSession()
    ps.open(str(island["proj"]))
    t0 = time.time()
    while ps._stats_state != "ready" and time.time() - t0 < 60:
        time.sleep(0.05)
    assert ps._stats_state == "ready"
    p = ps.project
    c = np.array([20.0, 20.0]) - p.origin[:2] + ORIGIN[:2]
    data = ps.open_region(c - 3, c + 3)
    link = data["link"]
    ps.attach_region(link)
    new_id = link.allocate_id()
    changed = np.arange(0, len(data["labels"]), 7)
    link.write(changed, np.full(len(changed), new_id, dtype=np.int32))
    got = p.gather_labels(link.tile[changed], link.idx[changed])
    assert np.all(got == new_id)
    ps.close_region()
    trees = {t["id"]: t for t in ps.trees()["trees"]}
    assert new_id in trees and trees[new_id]["count"] == len(changed)
    res = ps.merge([new_id], -1)
    assert res["changed_points"] == len(changed)
    ps.undo()
    assert np.all(p.gather_labels(link.tile[changed], link.idx[changed]) == new_id)
    ps.close()


def test_export_tiles_and_trees(island, tmp_path):
    from instance_annotator.io import load_tile_prefix
    from instance_annotator.project.export import export_project

    p = Project(island["proj"], writable=False)
    ids = sorted(set(np.unique(np.asarray(p.labels(0))).tolist()) - {-1})
    paths = export_project(p, tmp_path, tree_ids=ids[:2], log=lambda s: None)
    assert (tmp_path / "trees.csv").exists()
    name = p.tiles[0]["name"]
    tile = load_tile_prefix(tmp_path / name)
    assert len(tile["labels"]) == p.tiles[0]["n"]
    assert np.array_equal(tile["labels"], np.asarray(p.labels(0)))
    lazs = [q for q in paths if q.endswith(".laz") and "tree_" in q]
    assert len(lazs) == 2
    import laspy

    las = laspy.read(lazs[0])
    tid = ids[0]
    total = sum(int((np.asarray(p.labels(t)) == tid).sum()) for t in range(len(p.tiles)))
    assert len(las.x) == total
    p.close()
