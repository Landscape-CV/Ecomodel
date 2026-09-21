"""
Headless demos for the TLS instance annotator.

  python -m instance_annotator.cli_demo
  python SyntheticPipeline/instance_annotator/cli_demo.py --tile ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SP_DIR = Path(__file__).resolve().parents[1]
_ROOT = _SP_DIR.parent
for p in (str(_ROOT), str(_SP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from instance_annotator.io import load_tile_prefix, save_tile
from instance_annotator.labels import LabelEditor
from instance_annotator.segment import run_method
from instance_annotator.viz import (
    build_plotly_figure,
    downsample_indices,
    save_plotly_html,
    selection_from_click,
)


def _demo_dir() -> Path:
    d = _SP_DIR / "output" / "annotator_demos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def demo_treelearn(tile_prefix: Path, out: Path) -> None:
    print("=== Demo 1: TreeLearn on real tile ===")
    tile = load_tile_prefix(tile_prefix)
    lab, info = run_method(
        tile["xyz"],
        tile["intensity"],
        "treelearn",
        treex_stock_tls=True,
        results_folder=str(_SP_DIR / "output" / "annotator_workdir"),
    )
    print("  ", info)
    meta = dict(tile["meta"])
    meta["source_method"] = "treelearn"
    meta["demo"] = "demo_treelearn"
    paths = save_tile(
        out,
        "demo_treelearn",
        tile["xyz"],
        tile["intensity"],
        lab,
        meta=meta,
    )
    fig, _ = build_plotly_figure(
        tile["xyz"],
        lab,
        downsample_indices(len(tile["xyz"]), 120_000),
        title="Demo 1 TreeLearn",
    )
    html = save_plotly_html(fig, out / "demo_treelearn.html")
    print("  wrote", paths)
    print("  html", html)
    # stash labels for demo 3
    np.save(out / "demo_treelearn_work_labels.npy", lab)
    np.save(out / "demo_treelearn_xyz.npy", tile["xyz"])
    np.save(out / "demo_treelearn_inten.npy", tile["intensity"])
    (out / "demo_treelearn_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def demo_manual(tile_prefix: Path, out: Path) -> None:
    print("=== Demo 2: Manual paint two blobs ===")
    tile = load_tile_prefix(tile_prefix)
    ed = LabelEditor(n_points=len(tile["xyz"]))
    xyz = tile["xyz"]
    # Seed near two XY percentiles to create fake trees
    xs, ys = xyz[:, 0], xyz[:, 1]
    p1 = np.array([np.percentile(xs, 30), np.percentile(ys, 30), np.percentile(xyz[:, 2], 40)])
    p2 = np.array([np.percentile(xs, 70), np.percentile(ys, 70), np.percentile(xyz[:, 2], 55)])
    i1 = int(np.argmin(np.sum((xyz - p1) ** 2, axis=1)))
    i2 = int(np.argmin(np.sum((xyz - p2) ** 2, axis=1)))
    m1 = selection_from_click(xyz, i1, radius_m=2.0)
    m2 = selection_from_click(xyz, i2, radius_m=2.0)
    # Avoid overlap preference for blob 1
    m2 = m2 & ~m1
    ed.paint_new(m1)
    ed.paint_new(m2)
    meta = {"source_method": "manual", "demo": "demo_manual"}
    paths = save_tile(
        out,
        "demo_manual",
        xyz,
        tile["intensity"],
        ed.labels,
        meta=meta,
    )
    fig, _ = build_plotly_figure(
        xyz,
        ed.labels,
        downsample_indices(len(xyz), 120_000),
        title="Demo 2 Manual paint",
    )
    html = save_plotly_html(fig, out / "demo_manual.html")
    print(f"  trees={ed.num_trees} pts_t0={(ed.labels==0).sum()} pts_t1={(ed.labels==1).sum()}")
    print("  wrote", paths)
    print("  html", html)


def demo_correct(out: Path) -> None:
    print("=== Demo 3: Merge + mark non-tree on TreeLearn pred ===")
    lab_path = out / "demo_treelearn_work_labels.npy"
    xyz_path = out / "demo_treelearn_xyz.npy"
    inten_path = out / "demo_treelearn_inten.npy"
    if not lab_path.exists():
        raise FileNotFoundError("Run demo 1 first (missing demo_treelearn_work_labels.npy)")
    lab = np.load(lab_path)
    xyz = np.load(xyz_path)
    inten = np.load(inten_path)
    ed = LabelEditor(lab)
    ids = ed.tree_ids()
    if len(ids) >= 2:
        # Merge two largest instances
        counts = [(int(i), int((ed.labels == i).sum())) for i in ids]
        counts.sort(key=lambda t: -t[1])
        a, b = counts[0][0], counts[1][0]
        n = ed.merge(b, a)
        print(f"  merged tree {b} → {a} ({n} pts)")
    # Mark a small ball near centroid as non-tree
    c = xyz.mean(axis=0)
    i0 = int(np.argmin(np.sum((xyz - c) ** 2, axis=1)))
    mask = selection_from_click(xyz, i0, radius_m=0.5)
    n_nt = ed.mark_nontree(mask)
    print(f"  marked {n_nt} pts non-tree near centroid")
    # undo smoke
    ed.undo()
    ed.redo()
    meta = {"source_method": "treelearn", "demo": "demo_corrected", "edited": True}
    paths = save_tile(
        out,
        "demo_corrected",
        xyz,
        inten,
        ed.labels,
        meta=meta,
    )
    fig, _ = build_plotly_figure(
        xyz,
        ed.labels,
        downsample_indices(len(xyz), 120_000),
        title="Demo 3 Corrected GT",
    )
    html = save_plotly_html(fig, out / "demo_corrected.html")
    print("  wrote", paths)
    print("  html", html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run instance annotator demos")
    parser.add_argument(
        "--tile",
        type=str,
        default=str(_SP_DIR / "testdataset" / "real_instance" / "l1w_t00_03"),
        help="Tile prefix under real_instance (or similar)",
    )
    parser.add_argument(
        "--skip_treelearn",
        action="store_true",
        help="Skip GPU TreeLearn demo (still runs manual + needs prior treelearn artifacts for demo3)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="all | treelearn | manual | correct",
    )
    args = parser.parse_args()
    tile = Path(args.tile)
    out = _demo_dir()
    which = args.only.lower()
    if which in ("all", "treelearn") and not args.skip_treelearn:
        demo_treelearn(tile, out)
    if which in ("all", "manual"):
        demo_manual(tile, out)
    if which in ("all", "correct"):
        demo_correct(out)
    print("\nDemos complete →", out)


if __name__ == "__main__":
    main()
