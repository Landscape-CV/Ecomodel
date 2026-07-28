import json
import os
import random
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import psutil
import sys

try:
    import trimesh
except ImportError:
    warnings.warn("trimesh is not installed. Please 'pip install trimesh' to run ForestAssembler.")


GROUND_INSTANCE_ID = -1
CLUTTER_INSTANCE_ID = -2  # reserved for Phase-2 non-veg props


def _species_from_asset_stem(stem: str) -> str:
    """Parse species from asset stem like 'quaking_aspen_0003' or 'tree_0001'."""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


class ForestAssembler:
    """
    Assembles individual tree assets into a combined forest scene.
    It randomly places trees, applies wind tilt and scale variations,
    and accurately transforms ground truth skeleton parameters.

    Exports a per-face ``{scene}_face_tree_ids.npy`` map so LiDAR hits can be
    labeled with tree instance IDs (ground faces are ``-1``).
    """
    def __init__(self, asset_dir="SyntheticPipeline/output/assets", output_dir="SyntheticPipeline/output/scenes"):
        self.asset_dir = Path(asset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_tree_assets(self, species: Optional[str] = None):
        """
        Collect leaf-on tree assets. Searches recursively under asset_dir.

        Args:
            species: If set, only keep assets whose stem species prefix matches.
        """
        assets = []
        for obj_path in sorted(self.asset_dir.rglob("*.obj")):
            if "_noleaf" in obj_path.name:
                continue
            json_path = obj_path.with_suffix(".json")
            noleaf_path = obj_path.with_name(obj_path.stem + "_noleaf.obj")
            if not json_path.exists():
                continue
            asset_species = _species_from_asset_stem(obj_path.stem)
            if species is not None and asset_species != species:
                continue
            assets.append({
                "obj_path": obj_path,
                "json_path": json_path,
                "noleaf_path": noleaf_path if noleaf_path.exists() else None,
                "species": asset_species,
            })
        return assets

    def _sample_xy(
        self,
        area_size: Tuple[float, float],
        existing_xy: List[Tuple[float, float]],
        min_spacing: Optional[float],
        max_spacing: Optional[float],
        max_tries: int = 500,
    ) -> Tuple[float, float]:
        """Sample a tree XY position with optional spacing constraints."""
        half_x = area_size[0] / 2.0
        half_y = area_size[1] / 2.0

        def _clip(x: float, y: float) -> Tuple[float, float]:
            return (
                float(np.clip(x, -half_x, half_x)),
                float(np.clip(y, -half_y, half_y)),
            )

        for _ in range(max_tries):
            if existing_xy and max_spacing is not None:
                sx, sy = random.choice(existing_xy)
                angle = random.uniform(0.0, 2.0 * np.pi)
                r_min = float(min_spacing) if min_spacing is not None else 0.1
                r_max = float(max_spacing)
                if r_min >= r_max:
                    r_min = max(0.05, r_max * 0.5)
                r = random.uniform(r_min, r_max)
                x, y = _clip(sx + r * np.cos(angle), sy + r * np.sin(angle))
            else:
                x = random.uniform(-half_x, half_x)
                y = random.uniform(-half_y, half_y)

            if existing_xy:
                dists = np.sqrt(
                    (np.array([p[0] for p in existing_xy]) - x) ** 2
                    + (np.array([p[1] for p in existing_xy]) - y) ** 2
                )
                if min_spacing is not None and dists.min() < min_spacing:
                    continue
                if max_spacing is not None and dists.min() > max_spacing:
                    continue
            return x, y

        # Fallback: unconstrained random
        return random.uniform(-half_x, half_x), random.uniform(-half_y, half_y)

    @staticmethod
    def _nn_distance_stats(positions_xy: List[Tuple[float, float]]) -> Dict[str, float]:
        if len(positions_xy) < 2:
            return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "max": float("nan")}
        pts = np.asarray(positions_xy, dtype=float)
        nn = []
        for i in range(len(pts)):
            d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
            d[i] = np.inf
            nn.append(float(d.min()))
        nn = np.asarray(nn)
        return {
            "min": float(nn.min()),
            "mean": float(nn.mean()),
            "median": float(np.median(nn)),
            "max": float(nn.max()),
        }

    def generate_scene(
        self,
        scene_name: str = "scene_001",
        area_size: Tuple[float, float] = (50.0, 50.0),
        num_trees: int = 50,
        wind_vector: Tuple[float, float, float] = (0.1, 0.0, 0.0),
        species: Optional[str] = None,
        min_spacing: Optional[float] = None,
        max_spacing: Optional[float] = None,
        placement_mode: str = "random",
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a forest scene mesh and global ground truth JSON.

        Args:
            scene_name: Name of the generated scene files.
            area_size: Width and height of the generation area (meters).
            num_trees: Number of trees to place.
            wind_vector: Kept for API compatibility (wind applied in LiDAR sim).
            species: If set, only sample assets of this species (mono tiles).
            min_spacing: Minimum XY distance between tree stems (meters).
            max_spacing: Maximum XY distance to nearest existing stem (cluster mode).
            placement_mode: Label written to meta (`random`, `spread`, `cluster`).

        Returns:
            Placement/meta dict, or None if no assets were found.
        """
        assert area_size[0] > 0 and area_size[1] > 0, "Area size dimensions must be positive."
        assert num_trees > 0, "Number of trees must be greater than zero."
        _ = wind_vector  # wind sway is applied dynamically during LiDAR simulation

        assets = self._get_tree_assets(species=species)
        if not assets:
            warnings.warn("No tree assets found. Please run AssetManager first.")
            return None

        print(f"Found {len(assets)} tree assets. Generating forest...")

        import scipy.ndimage

        resolution = 0.5  # half-meter vertices
        nx = max(2, int(area_size[0] / resolution))
        ny = max(2, int(area_size[1] / resolution))

        xs = np.linspace(-area_size[0] / 2, area_size[0] / 2, nx)
        ys = np.linspace(-area_size[1] / 2, area_size[1] / 2, ny)
        X, Y = np.meshgrid(xs, ys)

        noise = np.random.normal(0, 0.5, X.shape)
        smoothed_noise = scipy.ndimage.gaussian_filter(noise, sigma=2.0)

        vertices = np.column_stack([X.ravel(), Y.ravel(), smoothed_noise.ravel()])

        faces = []
        for i in range(ny - 1):
            for j in range(nx - 1):
                v0 = i * nx + j
                v1 = v0 + 1
                v2 = v0 + nx
                v3 = v0 + nx + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

        grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        mesh_parts: List[Any] = [grid_mesh]
        noleaf_parts: List[Any] = []
        face_tree_ids: List[int] = [GROUND_INSTANCE_ID] * len(grid_mesh.faces)
        global_gt = []
        placements = []
        species_list = []
        existing_xy: List[Tuple[float, float]] = []

        for tree_id in range(num_trees):
            mem = psutil.virtual_memory()
            if mem.percent > 90.0:
                sys.stderr.write(
                    f"\nCRITICAL MEMORY ERROR: System RAM usage reached {mem.percent}%. "
                    "Aborting assembly to prevent OS freeze/swap thrashing!\n"
                )
                sys.exit(1)

            asset = random.choice(assets)
            obj_path = asset["obj_path"]
            json_path = asset["json_path"]
            noleaf_path = asset["noleaf_path"]
            asset_species = asset["species"]

            x, y = self._sample_xy(area_size, existing_xy, min_spacing, max_spacing)
            existing_xy.append((x, y))

            yaw = random.uniform(0, 2 * np.pi)
            yaw_transform = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])

            scale = random.uniform(0.7, 1.5)
            scale_transform = trimesh.transformations.scale_matrix(scale)
            rot_scale_transform = yaw_transform @ scale_transform

            tree_mesh = trimesh.load(str(obj_path), force="mesh", process=False)
            if not isinstance(tree_mesh, trimesh.Trimesh):
                tree_mesh = tree_mesh.dump(concatenate=True)
            tree_mesh.apply_transform(rot_scale_transform)

            min_z = tree_mesh.bounds[0][2]
            z_translation = -min_z

            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = [x, y, z_translation]

            tree_mesh.apply_transform(translation_matrix)
            mesh_parts.append(tree_mesh)
            face_tree_ids.extend([tree_id] * len(tree_mesh.faces))

            if noleaf_path is not None:
                noleaf_mesh = trimesh.load(str(noleaf_path), force="mesh", process=False)
                if not isinstance(noleaf_mesh, trimesh.Trimesh):
                    noleaf_mesh = noleaf_mesh.dump(concatenate=True)
                noleaf_mesh.apply_transform(rot_scale_transform)
                noleaf_mesh.apply_transform(translation_matrix)
                noleaf_parts.append(noleaf_mesh)

            transform = translation_matrix @ rot_scale_transform

            with open(json_path, "r") as f:
                skeleton = json.load(f)

            rot_matrix = transform[:3, :3]
            translation = transform[:3, 3]

            for cyl in skeleton:
                assert "start" in cyl and "end" in cyl and "axis" in cyl, "Invalid skeleton data format."

                start_new = rot_matrix @ np.array(cyl["start"]) + translation
                end_new = rot_matrix @ np.array(cyl["end"]) + translation
                axis_new = rot_matrix @ np.array(cyl["axis"])
                axis_norm = np.linalg.norm(axis_new)
                if axis_norm > 0:
                    axis_new = axis_new / axis_norm

                global_gt.append({
                    "tree_instance_id": tree_id,
                    "branch_id": cyl["branch_id"],
                    "start": start_new.tolist(),
                    "end": end_new.tolist(),
                    "radius": cyl["radius"] * scale,
                    "length": cyl["length"] * scale,
                    "axis": axis_new.tolist(),
                })

            placements.append({
                "tree_instance_id": tree_id,
                "species": asset_species,
                "x": x,
                "y": y,
                "yaw": yaw,
                "scale": scale,
                "asset": obj_path.name,
            })
            species_list.append(asset_species)

        merged_mesh = trimesh.util.concatenate(mesh_parts) if len(mesh_parts) > 1 else mesh_parts[0]
        face_tree_ids_arr = np.asarray(face_tree_ids, dtype=np.int32)
        if len(face_tree_ids_arr) != len(merged_mesh.faces):
            raise RuntimeError(
                f"face_tree_ids length ({len(face_tree_ids_arr)}) != "
                f"merged face count ({len(merged_mesh.faces)})"
            )

        out_ply = self.output_dir / f"{scene_name}.ply"
        out_noleaf_ply = self.output_dir / f"{scene_name}_gt_mesh.ply"
        out_json = self.output_dir / f"{scene_name}_gt.json"
        out_face_ids = self.output_dir / f"{scene_name}_face_tree_ids.npy"

        merged_mesh.export(str(out_ply))
        if noleaf_parts:
            merged_noleaf = trimesh.util.concatenate(noleaf_parts) if len(noleaf_parts) > 1 else noleaf_parts[0]
            merged_noleaf.export(str(out_noleaf_ply))

        with open(out_json, "w") as f:
            json.dump(global_gt, f, indent=4)

        np.save(str(out_face_ids), face_tree_ids_arr)

        meta = {
            "scene_name": scene_name,
            "num_trees": num_trees,
            "area_size": [float(area_size[0]), float(area_size[1])],
            "placement_mode": placement_mode,
            "min_spacing": min_spacing,
            "max_spacing": max_spacing,
            "species_filter": species,
            "species_list": species_list,
            "placements": placements,
            "nn_distance_stats": self._nn_distance_stats(existing_xy),
            "num_faces": int(len(merged_mesh.faces)),
            "num_ground_faces": int(len(grid_mesh.faces)),
            "face_tree_ids_path": str(out_face_ids),
        }

        print(f"Generated scene {scene_name} with {num_trees} trees.")
        print(f"Mesh: {out_ply}")
        print(f"GT  : {out_json}")
        print(f"Face IDs: {out_face_ids}")
        return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble Forest Scene from Tree Assets")
    parser.add_argument("--scene_name", type=str, default="forest_001")
    parser.add_argument("--num_trees", type=int, default=10)
    parser.add_argument("--area_size", type=float, default=200.0)
    parser.add_argument("--wind_x", type=float, default=0.0)
    parser.add_argument("--wind_y", type=float, default=0.0)
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--min_spacing", type=float, default=None)
    parser.add_argument("--max_spacing", type=float, default=None)
    parser.add_argument("--placement_mode", type=str, default="random")

    args = parser.parse_args()

    assembler = ForestAssembler()
    assembler.generate_scene(
        scene_name=args.scene_name,
        area_size=(args.area_size, args.area_size),
        num_trees=args.num_trees,
        wind_vector=(args.wind_x, args.wind_y, 0),
        species=args.species,
        min_spacing=args.min_spacing,
        max_spacing=args.max_spacing,
        placement_mode=args.placement_mode,
    )
