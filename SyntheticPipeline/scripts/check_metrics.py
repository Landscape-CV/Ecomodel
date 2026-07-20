"""Smoke-test the production-lite TreeQSM cylinder contract on one LAZ file."""

import argparse
import os
import sys
from pathlib import Path

import laspy
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecomodel_lite import EcomodelLite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", help="Input LAS/LAZ containing one tree")
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "SyntheticPipeline" / "output" / "qsm_smoke"))
    args = parser.parse_args()

    las = laspy.read(args.scan)
    points = np.column_stack((las.x, las.y, las.z))
    os.makedirs(args.out_dir, exist_ok=True)
    model = EcomodelLite(results_folder=args.out_dir)
    cylinders, metrics = model.get_cylinders_for_tree(points, tree_instance=0)
    if cylinders.shape[1:] != (9,) or len(cylinders) == 0:
        raise RuntimeError(f"TreeQSM returned invalid cylinder array {cylinders.shape}")
    print(f"TreeQSM returned {len(cylinders)} Cx9 cylinders.")
    if metrics:
        print(f"Tree metrics fields: {', '.join(sorted(metrics))}")


if __name__ == "__main__":
    main()
