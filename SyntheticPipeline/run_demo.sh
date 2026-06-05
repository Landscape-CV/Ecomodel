#!/bin/bash

echo "=== 0. Cleaning up old demo files ==="
rm -f SyntheticPipeline/output/assets/*.obj
rm -f SyntheticPipeline/output/assets/*.json
rm -f SyntheticPipeline/output/scenes/*

echo "=== 1. Generating Assets ==="
python SyntheticPipeline/asset_manager.py --num_trees 5 --blender_path /Applications/Blender.app/Contents/MacOS/Blender --blend_file SyntheticPipeline/output/assets/MangroveGen.blend --workers 2

echo "=== 2. Assembling Forest Scene ==="
python SyntheticPipeline/forest_assembler.py --scene_name demo_forest --num_trees 5 --area_size 15.0 --wind_x 0.2

echo "=== 3. Simulating LiDAR Scan ==="
python SyntheticPipeline/simulator_open3d.py --mesh_path SyntheticPipeline/output/scenes/demo_forest.obj --out_name demo_forest_scan

echo "=== 4. Parsing Ground Truth ==="
python SyntheticPipeline/gt_parser.py --json_path SyntheticPipeline/output/scenes/demo_forest_gt.json --out_name demo_forest_gt

echo "=== Demo Run Complete ==="
ls -lh SyntheticPipeline/output/*
