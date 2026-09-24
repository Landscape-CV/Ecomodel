"""
Blender entrypoint for tree generation.

Launched via ``blender -b [-blend] -P entrypoint.py -- ...``.
Dispatches to Mangrove Geometry Nodes when a 'Mangrove tree' object is present,
otherwise falls back to the native L-system generator.
"""

import argparse
import sys
from pathlib import Path

import bpy

# Allow sibling imports when Blender runs this file with -P.
_RUNTIME_DIR = Path(__file__).resolve().parent
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

from lsystem import generate_tree_native  # noqa: E402
from mangrove import generate_from_geometry_nodes  # noqa: E402


def main(argv=None):
    if argv is None:
        argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=float, default=10.0)
    parser.add_argument("--out_obj", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args, _ = parser.parse_known_args(argv)

    if "Mangrove tree" in bpy.data.objects:
        print("Found 'Mangrove tree' object! Using Track A: Geometry Nodes generator.")
        generate_from_geometry_nodes(args.seed, args.out_obj, args.out_json)
    else:
        print("No GN file loaded. Falling back to native L-System generator.")
        generate_tree_native(args.seed, args.height, args.out_obj, args.out_json)


if __name__ == "__main__":
    main()
