"""Island project CLI.

  python -m instance_annotator.project build  SRC_DIR --out PROJECT [--workers 2]
  python -m instance_annotator.project import PROJECT LABELS_DIR
  python -m instance_annotator.project segment PROJECT --method treelearn [--buffer 10] [--voxel 0.05]
  python -m instance_annotator.project stitch PROJECT [--apply 0.5]
  python -m instance_annotator.project export PROJECT --out DIR [--trees 3,7 | --all-trees]
  python -m instance_annotator.project info PROJECT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SP = Path(__file__).resolve().parents[2]
if str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m instance_annotator.project")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build octree + label store from a folder of tiles")
    b.add_argument("src")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--grid", type=int, default=128)
    b.add_argument("--leaf_max", type=int, default=150_000)
    b.add_argument("--force", action="store_true", help="Rebuild every tile")

    im = sub.add_parser("import", help="Import per-tile *_instances.npy labels")
    im.add_argument("project")
    im.add_argument("labels_dir")
    im.add_argument("--keep_ids", action="store_true", help="Do not offset ids per tile")

    s = sub.add_parser("segment", help="Buffered per-tile segmentation stitched into island ids")
    s.add_argument("project")
    s.add_argument("--method", default="treelearn")
    s.add_argument("--buffer", type=float, default=10.0)
    s.add_argument("--voxel", type=float, default=None, help="Seed voxel (default 0.05, treex 0.03)")
    s.add_argument("--max_seeds", type=int, default=None, help="Seed cap per chunk (default 12M, treex 40M)")
    s.add_argument("--tiles", default="", help="Comma-separated tile names (default all)")
    s.add_argument("--leaf_removal", action="store_true")
    s.add_argument("--reuse_cache", action="store_true", help="Reuse per-tile results in seg_cache/")
    s.add_argument("--pointsam_ckpt", default="", help="Point-SAM weights (.safetensors) for --method pointsam")

    st = sub.add_parser("stitch", help="Find (and optionally apply) merges across tile borders")
    st.add_argument("project")
    st.add_argument("--apply", type=float, default=None, help="Apply candidates with score >= this")

    e = sub.add_parser("export", help="Export tiles (global ids), trees.csv, per-tree LAZ")
    e.add_argument("project")
    e.add_argument("--out", required=True)
    e.add_argument("--tiles", default="")
    e.add_argument("--trees", default="", help="Comma-separated tree ids to export as LAZ")
    e.add_argument("--all-trees", action="store_true")
    e.add_argument("--no_tiles", action="store_true")

    i = sub.add_parser("info")
    i.add_argument("project")

    a = ap.parse_args(argv)
    if a.cmd == "build":
        from .build import build_project, default_workers

        build_project(a.src, a.out, grid=a.grid, leaf_max=a.leaf_max,
                      workers=a.workers or default_workers(), force=a.force)
    elif a.cmd == "import":
        from .build import import_labels

        import_labels(a.project, a.labels_dir, offset_per_tile=not a.keep_ids)
    elif a.cmd == "segment":
        from .segment_island import segment_island

        tiles = [t for t in a.tiles.split(",") if t] or None
        segment_island(a.project, a.method, buffer=a.buffer, voxel=a.voxel, max_seeds=a.max_seeds,
                       tile_names=tiles, leaf_removal=a.leaf_removal, reuse_cache=a.reuse_cache,
                       pointsam_ckpt=str(Path(a.pointsam_ckpt).resolve()) if a.pointsam_ckpt else None)
    elif a.cmd == "stitch":
        from .stitch import apply_candidates, find_candidates
        from .store import Project

        proj = Project(a.project)
        cands = find_candidates(proj)
        print(json.dumps(cands[:50], indent=1))
        print(f"{len(cands)} candidates written to {proj.root / 'stitch_candidates.json'}")
        if a.apply is not None:
            n = apply_candidates(proj, [c for c in cands if c["score"] >= a.apply])
            print(f"applied {n} merges")
        proj.close()
    elif a.cmd == "export":
        from .export import export_project

        export_project(a.project, a.out,
                       tile_names=[t for t in a.tiles.split(",") if t] or None,
                       tree_ids=[int(t) for t in a.trees.split(",") if t] or None,
                       all_trees=a.all_trees, write_tiles=not a.no_tiles)
    elif a.cmd == "info":
        from .store import Project

        p = Project(a.project, writable=False)
        print(json.dumps({k: v for k, v in p.meta.items() if k != "tiles"}, indent=1))
        print(f"{len(p.tiles)} tiles, {p.num_points:,} points, {p.num_nodes:,} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
