"""End-to-end TLS tile orchestration: assets → scene → scan → GT."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from synthetic_tls.assemble.asset_manager import AssetManager
from synthetic_tls.assemble.forest_assembler import ForestAssembler
from synthetic_tls.factory import create_generator
from synthetic_tls.gt.parser import GTParser
from synthetic_tls.simulate.open3d import Open3DSimulator


def run_tile(
    config: Dict[str, Any],
    *,
    generate_assets: bool = True,
    write_instances: bool = False,
    scan_positions: Optional[List[List[float]]] = None,
    species: Optional[str] = None,
    min_spacing: Optional[float] = None,
    max_spacing: Optional[float] = None,
    placement_mode: str = "random",
) -> Dict[str, Any]:
    """
    Run the four-stage synthetic TLS tile pipeline.

    Args:
        config: Full pipeline config (``output_dirs``, ``simulation``, ``generator``, ``noise``).
        generate_assets: If False, skip AssetManager and reuse existing assets.
        write_instances: If True, pass face_tree_ids into the simulator.
        scan_positions: Optional explicit scanner origins; otherwise sampled randomly.
        species / min_spacing / max_spacing / placement_mode: Forwarded to ForestAssembler.

    Returns:
        Dict of artifact paths and scene meta.
    """
    out_dirs = config.get("output_dirs", {})
    base_dir = out_dirs.get("base", "SyntheticPipeline/output")
    assets_dir = out_dirs.get("assets", f"{base_dir}/assets")
    scenes_dir = out_dirs.get("scenes", f"{base_dir}/scenes")
    pc_dir = out_dirs.get("pointclouds", f"{base_dir}/pointclouds")

    for d in [assets_dir, scenes_dir, pc_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    sim_config = config.get("simulation", {})
    gen_config = config.get("generator", {})
    num_trees = sim_config.get("num_trees", 5)
    scene_name = sim_config.get("scene_name", "demo_forest")
    area_size = sim_config.get("area_size", 20.0)
    wind_x = sim_config.get("wind_x", 0.0)
    wind_y = sim_config.get("wind_y", 0.0)

    if generate_assets:
        print("=== 1. Generating Assets ===")
        generator = create_generator(gen_config)
        asset_manager = AssetManager(generator=generator, output_dir=assets_dir)
        workers = gen_config.get("workers", 2)
        asset_manager.generate_trees(num_trees=num_trees, num_workers=workers)
    else:
        print("=== 1. Skipping Asset Generation ===")

    print("\n=== 2. Assembling Forest Scene ===")
    assembler = ForestAssembler(asset_dir=assets_dir, output_dir=scenes_dir)
    meta = assembler.generate_scene(
        scene_name=scene_name,
        area_size=(area_size, area_size),
        num_trees=num_trees,
        wind_vector=(wind_x, wind_y, 0),
        species=species,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        placement_mode=placement_mode,
    )
    if meta is None:
        print("Error: Scene assembly failed (no assets?).")
        sys.exit(1)

    scene_mesh_path = os.path.join(scenes_dir, f"{scene_name}.ply")
    if not os.path.exists(scene_mesh_path):
        print(f"Error: Scene mesh not found at {scene_mesh_path}. Assembly might have failed.")
        sys.exit(1)

    print("\n=== 3. Simulating LiDAR Scan ===")
    sim = Open3DSimulator(output_dir=pc_dir)

    if scan_positions is None:
        positions: List[List[float]] = []
        num_scans = sim_config.get("num_scans", 5)
        for _ in range(num_scans):
            x = random.uniform(-area_size / 2, area_size / 2)
            y = random.uniform(-area_size / 2, area_size / 2)
            positions.append([x, y, 1.5])
    else:
        positions = scan_positions

    noise_params = dict(config.get("noise", {}))
    noise_params["wind_x"] = wind_x
    noise_params["wind_y"] = wind_y

    face_tree_ids = None
    instances_output_path = None
    if write_instances:
        face_tree_ids = os.path.join(scenes_dir, f"{scene_name}_face_tree_ids.npy")
        instances_output_path = os.path.join(pc_dir, f"{scene_name}_scan_instances.npy")

    scan_path = sim.scan(
        scene_mesh_path,
        positions,
        noise_params,
        f"{scene_name}_scan",
        face_tree_ids=face_tree_ids,
        instances_output_path=instances_output_path,
    )

    print("\n=== 4. Parsing Ground Truth ===")
    parser = GTParser(output_dir=scenes_dir)
    scene_gt_json = os.path.join(scenes_dir, f"{scene_name}_gt.json")
    if not os.path.exists(scene_gt_json):
        print(f"Error: GT JSON not found at {scene_gt_json}.")
        sys.exit(1)

    gt_txt = parser.parse_to_txt(scene_gt_json, f"{scene_name}_gt")

    print("\n=== Demo Run Complete ===")
    return {
        "assets_dir": assets_dir,
        "scenes_dir": scenes_dir,
        "pc_dir": pc_dir,
        "scene_name": scene_name,
        "scene_ply": scene_mesh_path,
        "scene_gt_json": scene_gt_json,
        "gt_txt": str(gt_txt) if gt_txt else None,
        "scan_laz": str(scan_path) if scan_path else None,
        "instances_npy": instances_output_path if write_instances else None,
        "meta": meta,
    }
