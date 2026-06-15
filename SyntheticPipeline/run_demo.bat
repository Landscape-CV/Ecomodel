@echo off
echo === 0. Cleaning up old demo files ===
if exist "SyntheticPipeline\output\assets\*.obj" del /q /f "SyntheticPipeline\output\assets\*.obj"
if exist "SyntheticPipeline\output\assets\*.json" del /q /f "SyntheticPipeline\output\assets\*.json"
if exist "SyntheticPipeline\output\scenes\*" del /q /f "SyntheticPipeline\output\scenes\*"

echo === 1. Generating Assets ===
rem Note: You may need to update the --blender_path if Blender is installed in a different directory or version folder
.\.venv\Scripts\python.exe SyntheticPipeline\asset_manager.py --num_trees 5 --blender_path "D:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --blend_file "D:\pointclouds\Mangrove Gen.blend" --workers 2

echo === 2. Assembling Forest Scene ===
.\.venv\Scripts\python.exe SyntheticPipeline\forest_assembler.py --scene_name demo_forest --num_trees 5 --area_size 15.0 --wind_x 0.2

echo === 3. Simulating LiDAR Scan ===
.\.venv\Scripts\python.exe SyntheticPipeline\simulator_open3d.py --mesh_path "SyntheticPipeline\output\scenes\demo_forest.obj" --out_name demo_forest_scan

echo === 4. Parsing Ground Truth ===
.\.venv\Scripts\python.exe SyntheticPipeline\gt_parser.py --json_path "SyntheticPipeline\output\scenes\demo_forest_gt.json" --out_name demo_forest_gt

echo === Demo Run Complete ===
dir /s "SyntheticPipeline\output\*"
