"""
E2a: sweep detect_stems parameters on the benchmark's cached (ground-removed) tiles and report
click quality (recall / precision / duplicates / misses). CPU only.

Usage (from SyntheticPipeline/, after benchmark_pointsam.py has cached the tiles):
  python pointsam_training/tune_stems.py
  python pointsam_training/tune_stems.py --cache output/pointsam_bench/_cache/real_instance
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SP = _HERE.parent
for p in (_SP.parent, _SP, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from promptable_instance import BaseSurface, stem_click_prompts, voxel_downsample_with_index  # noqa: E402
from benchmark_pointsam import click_quality  # noqa: E402

GRID = {
    "eps": [0.1, 0.2],
    "merge_dist": [0.5, 1.0, 1.5],
    "min_points": [15, 40],
    "max_xy_extent": [1.0, 2.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_SP / "output" / "pointsam_bench" / "_cache" / "synthetic"))
    ap.add_argument("--max_tiles", type=int, default=0)
    args = ap.parse_args()
    files = sorted(Path(args.cache).glob("*.npz"))[: args.max_tiles or None]
    if not files:
        raise SystemExit(f"no cached tiles in {args.cache}")
    tiles = []
    for f in files:
        z = np.load(f)
        xyz = z["work"][:, :3].astype(np.float64)
        _, keep = voxel_downsample_with_index(xyz, 0.05)
        tiles.append((f.stem, xyz[keep], z["work"][keep], z["gt"][keep]))
    surfaces = [BaseSurface(t[1]) for t in tiles]

    keys = list(GRID)
    rows = []
    for vals in itertools.product(*GRID.values()):
        kw = dict(zip(keys, vals))
        q = [click_quality(work, gt, stem_click_prompts(xyz, surface=s, **kw))
             for (_, xyz, work, gt), s in zip(tiles, surfaces)]
        mean = {k: float(np.mean([r[k] for r in q])) for k in ("click_recall", "click_precision", "click_dup", "click_miss")}
        rows.append((kw, mean))
        print(kw, {k: round(v, 3) for k, v in mean.items()}, flush=True)

    rows.sort(key=lambda r: -(2 * r[1]["click_recall"] * r[1]["click_precision"]
                              / max(1e-9, r[1]["click_recall"] + r[1]["click_precision"])))
    print("\nbest by click F1:")
    for kw, m in rows[:5]:
        print(" ", kw, {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    main()
