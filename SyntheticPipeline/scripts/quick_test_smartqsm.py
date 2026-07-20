"""Run a one-tile SmartQSM smoke test with synthetic oracle wood labels."""

import argparse
import os
import sys
from pathlib import Path

import laspy
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gui.smartqsm_runner import run_smartqsm_on_segments
from benchmark_qsm import MockConfig, evaluate_cylinders, build_parser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", help="Synthetic *_scan.laz file")
    parser.add_argument("--sq-dir", default=str(ROOT_DIR / "thirdparty" / "SmartQSM"))
    parser.add_argument("--sq-py", default=sys.executable)
    parser.add_argument(
        "--sq-cfg",
        default=str(ROOT_DIR / "thirdparty" / "SmartQSM" / "configs" / "space-colonization.yaml"),
    )
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "SyntheticPipeline" / "output" / "smartqsm_smoke"))
    args = parser.parse_args()

    base = args.scan.removesuffix("_scan.laz")
    labels = np.load(base + "_labels.npy")
    gt_cylinders = np.atleast_2d(np.loadtxt(base + "_gt.txt"))
    las = laspy.read(args.scan)
    xyz = np.column_stack((las.x, las.y, las.z))
    if len(labels) != len(xyz):
        raise ValueError("Point/label count mismatch")
    oracle = xyz[labels == 1]
    point_data = np.column_stack((oracle, np.ones(len(oracle))))

    os.makedirs(args.out_dir, exist_ok=True)
    config = MockConfig(args.sq_dir, args.sq_py, args.sq_cfg, args.timeout, 15.0)
    cylinders = run_smartqsm_on_segments(
        point_data,
        np.zeros(len(point_data), dtype=np.int32),
        args.out_dir,
        config,
        log=lambda message: print(message, end=""),
    )
    if len(cylinders) == 0:
        raise RuntimeError("SmartQSM returned no cylinders")

    metric_args = build_parser().parse_args([])
    metrics = evaluate_cylinders(gt_cylinders, cylinders, seed=42, args=metric_args)
    print(f"SmartQSM returned {len(cylinders)} Cx9 cylinders.")
    print(f"Whole-tree F1={metrics['Whole_F1']:.4f}, IoU={metrics['Whole_IoU']:.4f}")


if __name__ == "__main__":
    main()
