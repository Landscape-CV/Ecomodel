"""Export an island project back to the per-tile layout plus per-tree clouds.

Per tile: `<name>_scan.laz` is the original source file (copied, lossless, same point order),
`<name>_instances.npy` holds island-wide tree IDs, `<name>_meta.json` the usual tile meta. These
load in single-file mode like any other tile. `trees.csv` lists every tree; per-tree LAZ files
are rebuilt from the octree (positions within ~1 mm, 8-bit intensity).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .stats import build_sample, label_counts, island_trees
from .store import Project, ProjectError

Progress = Optional[Callable[[float, float, str], None]]


def _write_laz(path: Path, xyz_abs: np.ndarray, inten_u8: np.ndarray, crs_wkt: str = "") -> None:
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.floor(xyz_abs.min(axis=0)) if len(xyz_abs) else np.zeros(3)
    header.scales = np.array([0.001, 0.001, 0.001])
    if crs_wkt:
        try:
            import pyproj

            header.add_crs(pyproj.CRS.from_wkt(crs_wkt))
        except Exception:
            pass
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz_abs[:, 0], xyz_abs[:, 1], xyz_abs[:, 2]
    las.intensity = inten_u8.astype(np.uint16) << 8
    las.write(str(path))


def tree_points(proj: Project, tree_id: int, tiles: Optional[Sequence[int]] = None):
    """(xyz local float32, intensity uint8) of one tree, gathered via the node index."""
    xs, ins = [], []
    for t in (range(len(proj.tiles)) if tiles is None else tiles):
        lab = np.asarray(proj.labels(t))
        idx = np.flatnonzero(lab == tree_id)
        if len(idx) == 0:
            continue
        nodes = np.unique(np.asarray(proj.nodeof(t))[idx])
        for loc in nodes.tolist():
            g = int(proj.tile_base[t]) + int(loc)
            xyz, inten, nidx = proj.read_node(g)
            m = lab[nidx] == tree_id
            xs.append(xyz[m])
            ins.append(inten[m])
    if not xs:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8)
    return np.concatenate(xs), np.concatenate(ins)


def export_project(project: Project | str, out: str | Path, *, tile_names: Optional[Sequence[str]] = None,
                   tree_ids: Optional[Sequence[int]] = None, all_trees: bool = False, write_tiles: bool = True,
                   progress: Progress = None, log: Callable[[str], None] = print) -> List[str]:
    proj = project if isinstance(project, Project) else Project(project, writable=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prog = progress or (lambda d, t, m: None)
    names = [t["name"] for t in proj.tiles]
    tiles = [names.index(n) for n in tile_names] if tile_names else list(range(len(names)))
    if tile_names and len(tiles) != len(tile_names):
        raise ProjectError("Unknown tile name")
    proj.flush()
    paths: List[str] = []
    reviewed = sorted(int(v) for v in proj.meta.get("reviewed", []))
    crs = proj.meta.get("crs", "")

    counts = label_counts(proj)
    trees = island_trees(proj, counts, build_sample(proj))
    if tree_ids is None and all_trees:
        tree_ids = [t["id"] for t in trees]
    tree_ids = list(tree_ids or [])
    steps = (len(tiles) if write_tiles else 0) + len(tree_ids) + 1
    step = 0

    if write_tiles:
        for t in tiles:
            name = names[t]
            prog(step, steps, f"Writing tile {name}")
            src = Path(proj.tiles[t]["path"])
            if not src.exists():
                raise ProjectError(f"Source file for {name} is missing: {src}")
            dst = out / f"{name}_scan{src.suffix.lower()}"
            if dst.resolve() != src.resolve():
                shutil.copyfile(src, dst)
            lab = np.asarray(proj.labels(t), dtype=np.int32)
            np.save(out / f"{name}_instances.npy", lab)
            present = np.unique(lab[lab >= 0])
            meta = {
                "tile_name": name, "num_points": int(len(lab)), "num_trees": int(len(present)),
                "label_schema": {"ground": -1, "trees": ">=0"}, "edited": True,
                "island_project": str(proj.root), "island_ids": True,
                "reviewed_trees": [r for r in reviewed if r in set(present.tolist())],
                "source": str(src),
            }
            (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            paths += [str(dst), str(out / f"{name}_instances.npy"), str(out / f"{name}_meta.json")]
            step += 1

    o = proj.origin
    csv_path = out / "trees.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tree_id", "points", "tiles", "reviewed", "base_x", "base_y", "zmin", "height",
                    "xmin", "ymin", "xmax", "ymax"])
        for tr in trees:
            base = tr["base"] or [np.nan, np.nan]
            lo = tr["bmin"] or [np.nan] * 3
            hi = tr["bmax"] or [np.nan] * 3
            w.writerow([tr["id"], tr["count"], ";".join(names[t] for t in tr["tiles"]), int(tr["reviewed"]),
                        f"{base[0] + o[0]:.3f}", f"{base[1] + o[1]:.3f}", f"{lo[2] + o[2]:.3f}", tr["height"],
                        f"{lo[0] + o[0]:.3f}", f"{lo[1] + o[1]:.3f}", f"{hi[0] + o[0]:.3f}", f"{hi[1] + o[1]:.3f}"])
    paths.append(str(csv_path))

    if tree_ids:
        tdir = out / "trees"
        tdir.mkdir(exist_ok=True)
        tiles_of = {tr["id"]: tr["tiles"] for tr in trees}
        for tid in tree_ids:
            prog(step, steps, f"Writing tree {tid}")
            xyz, inten = tree_points(proj, int(tid), tiles_of.get(int(tid)))
            step += 1
            if len(xyz) == 0:
                log(f"[export] tree {tid}: no points")
                continue
            p = tdir / f"tree_{int(tid):06d}.laz"
            _write_laz(p, xyz.astype(np.float64) + o, inten, crs)
            paths.append(str(p))
    prog(steps, steps, "Done")
    log(f"[export] {len(paths)} files in {out}")
    return paths
