"""
Generate a multi-tree instance-segmentation benchmark dataset.

Factorial design (default config):
  4 densities × 2 compositions (mixed/mono) × 8 replicates = 64 tiles
  20 × 20 m plots with per-point instance labels from ray hit → face_tree_ids.

Usage:
  python scripts/generate_instance_benchmark.py
  python scripts/generate_instance_benchmark.py --config configs/instance_benchmark_smoke.example.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.arbaro_generator import ArbaroGenerator
from pipeline.asset_manager import AssetManager
from pipeline.forest_assembler import ForestAssembler
from pipeline.gt_parser import GTParser
from pipeline.simulator_open3d import Open3DSimulator


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _resolve_path(base_dir: str, maybe_relative: str) -> str:
    p = Path(maybe_relative)
    if p.is_absolute():
        return str(p)
    return str(Path(base_dir) / p)


def build_species_pool(
    base_dir: str,
    assets_dir: str,
    jar_path: str,
    xml_files: List[str],
    pool_size: int,
    num_workers: int,
    height_range: tuple,
) -> List[str]:
    """Generate Arbaro assets for every species into a shared pool directory."""
    os.makedirs(assets_dir, exist_ok=True)
    species_names = []
    for xml_path in xml_files:
        veg_name = Path(xml_path).stem
        species_names.append(veg_name)
        print(f"\n=== Asset pool: {veg_name} ({pool_size} variants) ===")
        generator = ArbaroGenerator(
            jar_path=jar_path,
            xml_template=xml_path,
            java_bin="java",
        )
        manager = AssetManager(generator=generator, output_dir=assets_dir)
        manager.generate_trees(
            num_trees=pool_size,
            height_range=tuple(height_range),
            num_workers=num_workers,
            name_prefix=veg_name,
            start_index=0,
        )
    return species_names


def generate_tile(
    *,
    tile_name: str,
    density_class: str,
    composition: str,
    density_cfg: Dict[str, Any],
    species_names: List[str],
    mono_species: Optional[str],
    area_size: float,
    assets_dir: str,
    scenes_dir: str,
    pc_dir: str,
    dataset_dir: str,
    noise_params: Dict[str, float],
    num_scans_range: List[int],
    label_schema: Dict[str, Any],
    seed: int,
) -> bool:
    random.seed(seed)
    np.random.seed(seed)

    if os.path.exists(scenes_dir):
        shutil.rmtree(scenes_dir)
    os.makedirs(scenes_dir, exist_ok=True)
    if os.path.exists(pc_dir):
        shutil.rmtree(pc_dir)
    os.makedirs(pc_dir, exist_ok=True)

    num_trees = int(density_cfg["num_trees"])
    placement_mode = density_cfg.get("placement_mode", "random")
    min_spacing = density_cfg.get("min_spacing")
    max_spacing = density_cfg.get("max_spacing")

    scene_name = f"{tile_name}_scene"
    assembler = ForestAssembler(asset_dir=assets_dir, output_dir=scenes_dir)
    assembly_meta = assembler.generate_scene(
        scene_name=scene_name,
        area_size=(area_size, area_size),
        num_trees=num_trees,
        wind_vector=(0.0, 0.0, 0.0),
        species=mono_species if composition == "mono" else None,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        placement_mode=placement_mode,
    )
    if assembly_meta is None:
        print(f"  [Error] Assembly failed for {tile_name}")
        return False

    scene_mesh_path = os.path.join(scenes_dir, f"{scene_name}.ply")
    face_ids_path = os.path.join(scenes_dir, f"{scene_name}_face_tree_ids.npy")
    if not os.path.exists(scene_mesh_path) or not os.path.exists(face_ids_path):
        print(f"  [Error] Missing scene mesh or face_tree_ids for {tile_name}")
        return False

    n_lo, n_hi = num_scans_range[0], num_scans_range[1]
    num_scans = random.randint(n_lo, n_hi)
    positions = []
    for _ in range(num_scans):
        x = random.uniform(-area_size / 2, area_size / 2)
        y = random.uniform(-area_size / 2, area_size / 2)
        positions.append([x, y, 1.5])

    sim = Open3DSimulator(output_dir=pc_dir)
    out_scan_name = f"{scene_name}_scan"
    laz_path = sim.scan(
        scene_mesh_path,
        positions,
        noise_params,
        out_scan_name,
        face_tree_ids=face_ids_path,
    )
    if laz_path is None:
        print(f"  [Error] Scan failed for {tile_name}")
        return False

    parser = GTParser(output_dir=scenes_dir)
    scene_gt_json = os.path.join(scenes_dir, f"{scene_name}_gt.json")
    if os.path.exists(scene_gt_json):
        parser.parse_to_txt(scene_gt_json, f"{scene_name}_gt")

    # Copy artifacts into dataset dir with stable tile prefix
    os.makedirs(dataset_dir, exist_ok=True)
    copies = {
        os.path.join(pc_dir, f"{out_scan_name}.laz"): os.path.join(dataset_dir, f"{tile_name}_scan.laz"),
        os.path.join(pc_dir, f"{out_scan_name}_instances.npy"): os.path.join(
            dataset_dir, f"{tile_name}_instances.npy"
        ),
        os.path.join(scenes_dir, f"{scene_name}_gt_mesh.ply"): os.path.join(
            dataset_dir, f"{tile_name}_trunk.ply"
        ),
        os.path.join(scenes_dir, f"{scene_name}_gt.json"): os.path.join(
            dataset_dir, f"{tile_name}_gt.json"
        ),
        os.path.join(scenes_dir, f"{scene_name}_gt.txt"): os.path.join(
            dataset_dir, f"{tile_name}_gt.txt"
        ),
        face_ids_path: os.path.join(dataset_dir, f"{tile_name}_face_tree_ids.npy"),
        scene_mesh_path: os.path.join(dataset_dir, f"{tile_name}_scene.ply"),
    }
    for src, dst in copies.items():
        if os.path.exists(src):
            shutil.copy(src, dst)

    instances_path = os.path.join(dataset_dir, f"{tile_name}_instances.npy")
    instance_stats = {}
    if os.path.exists(instances_path):
        inst = np.load(instances_path)
        tree_ids = np.unique(inst[inst >= 0])
        instance_stats = {
            "num_points": int(len(inst)),
            "num_ground_points": int(np.sum(inst == label_schema.get("ground", -1))),
            "num_tree_points": int(np.sum(inst >= 0)),
            "unique_tree_ids": tree_ids.astype(int).tolist(),
        }

    meta = {
        "tile_name": tile_name,
        "density_class": density_class,
        "composition": composition,
        "num_trees": num_trees,
        "area_size": area_size,
        "placement_mode": placement_mode,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "species_list": assembly_meta.get("species_list", []),
        "mono_species": mono_species,
        "available_species": species_names,
        "nn_distance_stats": assembly_meta.get("nn_distance_stats", {}),
        "placements": assembly_meta.get("placements", []),
        "num_scans": num_scans,
        "scan_positions": positions,
        "noise_params": noise_params,
        "label_schema": label_schema,
        "seed": seed,
        "instance_stats": instance_stats,
    }
    with open(os.path.join(dataset_dir, f"{tile_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"  [OK] {tile_name} -> {instance_stats.get('num_points', 0)} points")
    return True


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(base_dir, "configs", "instance_benchmark.example.json")

    parser = argparse.ArgumentParser(description="Generate multi-tree instance segmentation benchmark tiles")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--skip_asset_gen", action="store_true",
                        help="Reuse existing assets in output/instance_assets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)

    arbaro_trees_dir = os.path.join(base_dir, "lib", "arbaro", "trees")
    jar_path = os.path.join(base_dir, "lib", "arbaro", "arbaro_cmd.jar")
    assets_dir = os.path.join(base_dir, "output", "instance_assets")
    scenes_dir = os.path.join(base_dir, "output", "instance_temp_scenes")
    pc_dir = os.path.join(base_dir, "output", "instance_temp_pointclouds")
    dataset_dir = _resolve_path(base_dir, cfg.get("dataset_dir", "testdataset/instance"))

    if not os.path.exists(jar_path):
        print(f"Arbaro JAR not found at {jar_path}")
        print("Run: python scripts/setup_arbaro.py")
        sys.exit(1)

    xml_files = sorted(glob.glob(os.path.join(arbaro_trees_dir, "*.xml")))
    if not xml_files:
        print(f"No Arbaro species XMLs in {arbaro_trees_dir}")
        print("Run: python scripts/setup_arbaro.py")
        sys.exit(1)

    max_species = cfg.get("max_species")
    if max_species is not None:
        xml_files = xml_files[: int(max_species)]

    pool_size = int(cfg.get("pool_size_per_species", 5))
    num_workers = int(cfg.get("num_workers", 4))
    height_range = cfg.get("height_range", [8.0, 15.0])

    if args.skip_asset_gen and os.path.isdir(assets_dir) and any(Path(assets_dir).glob("*.obj")):
        species_names = sorted({
            p.stem.rsplit("_", 1)[0]
            for p in Path(assets_dir).glob("*.obj")
            if "_noleaf" not in p.name and p.with_suffix(".json").exists()
        })
        print(f"Reusing {len(species_names)} species from {assets_dir}")
    else:
        if os.path.exists(assets_dir) and not args.skip_asset_gen:
            # Keep pool across runs only when skipping; otherwise rebuild cleanly
            pass
        species_names = build_species_pool(
            base_dir=base_dir,
            assets_dir=assets_dir,
            jar_path=jar_path,
            xml_files=xml_files,
            pool_size=pool_size,
            num_workers=num_workers,
            height_range=height_range,
        )

    if not species_names:
        print("No species available in asset pool.")
        sys.exit(1)

    area_size = float(cfg.get("area_size", 20.0))
    replicates = int(cfg.get("replicates", 8))
    compositions = cfg.get("compositions", ["mixed", "mono"])
    densities = cfg.get("densities", {})
    noise_params = cfg.get("noise", {})
    num_scans_range = cfg.get("num_scans_range", [3, 5])
    label_schema = cfg.get("label_schema", {"ground": -1, "clutter": -2})

    os.makedirs(dataset_dir, exist_ok=True)
    tile_index = 0
    ok = 0
    fail = 0

    for density_class, density_cfg in densities.items():
        for composition in compositions:
            for rep in range(replicates):
                if composition == "mono":
                    mono_species = species_names[tile_index % len(species_names)]
                    tile_name = f"{density_class}_mono_{mono_species}_{rep:02d}"
                else:
                    mono_species = None
                    tile_name = f"{density_class}_mixed_{rep:02d}"

                print(f"\n>>> Tile {tile_name}")
                success = generate_tile(
                    tile_name=tile_name,
                    density_class=density_class,
                    composition=composition,
                    density_cfg=density_cfg,
                    species_names=species_names,
                    mono_species=mono_species,
                    area_size=area_size,
                    assets_dir=assets_dir,
                    scenes_dir=scenes_dir,
                    pc_dir=pc_dir,
                    dataset_dir=dataset_dir,
                    noise_params=noise_params,
                    num_scans_range=num_scans_range,
                    label_schema=label_schema,
                    seed=args.seed + tile_index * 997,
                )
                tile_index += 1
                if success:
                    ok += 1
                else:
                    fail += 1

    manifest = {
        "config": args.config,
        "dataset_dir": dataset_dir,
        "tiles_ok": ok,
        "tiles_failed": fail,
        "species": species_names,
        "densities": list(densities.keys()),
        "compositions": compositions,
        "replicates": replicates,
    }
    with open(os.path.join(dataset_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nDone. {ok} tiles ok, {fail} failed -> {dataset_dir}")


if __name__ == "__main__":
    main()
