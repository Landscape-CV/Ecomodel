Write-Host "=== 0. Cleaning up old demo files ==="
Remove-Item -Path "SyntheticPipeline\output\assets\*.obj" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "SyntheticPipeline\output\pointclouds\*.ply" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "SyntheticPipeline\output\pointclouds\*.xyz" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "SyntheticPipeline\output\assets\*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "SyntheticPipeline\output\scenes\*" -Force -Recurse -ErrorAction SilentlyContinue

Write-Host "=== 1. Generating Assets ==="
# Note: You may need to update the --blender_path if Blender is installed in a different directory or version folder
.\.venv\Scripts\python.exe SyntheticPipeline\asset_manager.py --num_trees 5 --blender_path "D:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --blend_file "D:\pointclouds\Mangrove Gen.blend" --workers 2

Write-Host "=== 2. Assembling Forest Scene ==="
.\.venv\Scripts\python.exe SyntheticPipeline\forest_assembler.py --scene_name demo_forest --num_trees 5 --area_size 200.0
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline aborted during Forest Assembly (Exit Code: $LASTEXITCODE). Check the logs for MemoryError."
    exit $LASTEXITCODE
}

Write-Host "=== 3. Simulating LiDAR Scan ==="
.\.venv\Scripts\python.exe SyntheticPipeline\simulator_open3d.py --mesh_path "SyntheticPipeline\output\scenes\demo_forest.ply" --out_name demo_forest_scan --num_scans 5 --area_size 200.0

Write-Host "=== 4. Parsing Ground Truth ==="
.\.venv\Scripts\python.exe SyntheticPipeline\gt_parser.py --json_path "SyntheticPipeline\output\scenes\demo_forest_gt.json" --out_name demo_forest_gt

Write-Host "=== Demo Run Complete ==="
Get-ChildItem -Path "SyntheticPipeline\output\*" -Recurse
