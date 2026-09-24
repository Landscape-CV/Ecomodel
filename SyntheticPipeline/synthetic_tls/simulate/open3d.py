import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import open3d as o3d
except ImportError:
    warnings.warn("open3d is not installed. Please 'pip install open3d' to run Open3DSimulator.")

try:
    import laspy
except ImportError:
    warnings.warn("laspy is not installed. Please 'pip install laspy lazrs' for .laz support.")

from synthetic_tls.constants import GROUND_INSTANCE_ID
from synthetic_tls.simulate.base import BaseLiDARSimulator


class Open3DSimulator(BaseLiDARSimulator):
    """
    Simulates a Terrestrial Laser Scanner (TLS) using Open3D's raycasting engine.
    Supports distance noise, beam divergence proxy (spatial jitter), and wind sway.

    When ``face_tree_ids`` is provided, also writes per-hit instance labels
    (``{output_filename}_instances.npy``) by mapping ray ``primitive_ids``.
    """

    def __init__(self, output_dir: str = "SyntheticPipeline/output/pointclouds"):
        super().__init__(output_dir)

    def _create_spherical_rays(self, origin: List[float], resolution_theta: float = 0.1, resolution_phi: float = 0.1):
        assert resolution_theta > 0 and resolution_phi > 0, "Resolutions must be strictly positive."
        thetas = np.arange(0, 360, resolution_theta)
        phis = np.arange(0, 180, resolution_phi)

        theta_grid, phi_grid = np.meshgrid(np.radians(thetas), np.radians(phis))

        theta_grid += np.random.uniform(
            -np.radians(resolution_theta) / 2, np.radians(resolution_theta) / 2, theta_grid.shape
        )
        phi_grid += np.random.uniform(
            -np.radians(resolution_phi) / 2, np.radians(resolution_phi) / 2, phi_grid.shape
        )

        dx = np.sin(phi_grid) * np.cos(theta_grid)
        dy = np.sin(phi_grid) * np.sin(theta_grid)
        dz = np.cos(phi_grid)

        directions = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
        origins = np.tile(origin, (directions.shape[0], 1))

        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        return rays, directions

    def scan(
        self,
        mesh_path: str,
        scan_positions: List[List[float]],
        noise_params: Dict[str, float],
        output_filename: str = "simulated_scan",
        face_tree_ids: Optional[Union[np.ndarray, str, Path]] = None,
        instances_output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[str]:
        assert len(scan_positions) > 0, "Must provide at least one scan position."
        if not Path(mesh_path).exists():
            warnings.warn(f"Mesh file not found: {mesh_path}")
            return None

        face_ids_arr = None
        if face_tree_ids is not None:
            if isinstance(face_tree_ids, (str, Path)):
                face_ids_arr = np.load(str(face_tree_ids))
            else:
                face_ids_arr = np.asarray(face_tree_ids)
            face_ids_arr = face_ids_arr.astype(np.int32)

        print(f"Loading mesh: {mesh_path}")
        base_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        num_triangles = len(base_mesh.triangles)
        if face_ids_arr is not None and len(face_ids_arr) != num_triangles:
            raise ValueError(
                f"face_tree_ids length ({len(face_ids_arr)}) does not match "
                f"mesh triangle count ({num_triangles})"
            )

        wind_x = noise_params.get("wind_x", 0.0)
        wind_y = noise_params.get("wind_y", 0.0)
        wind_sway_std = noise_params.get("wind_sway_std", 0.0)

        bounds_max = base_mesh.get_max_bound()
        max_height = bounds_max[2] if bounds_max[2] > 1.0 else 1.0

        all_points = []
        all_instance_ids = []

        for pos_idx, origin in enumerate(scan_positions):
            print(f"Scanning from position {pos_idx + 1}/{len(scan_positions)}: {origin}")

            import copy

            scan_mesh = copy.deepcopy(base_mesh)
            vertices = np.asarray(scan_mesh.vertices).copy()

            current_wind_x = wind_x + np.random.normal(0, wind_sway_std)
            current_wind_y = wind_y + np.random.normal(0, wind_sway_std)

            if abs(current_wind_x) > 1e-4 or abs(current_wind_y) > 1e-4:
                z = vertices[:, 2]
                z_factor = np.clip(z / max_height, 0, 1) ** 2
                vertices[:, 0] += current_wind_x * z_factor
                vertices[:, 1] += current_wind_y * z_factor
                scan_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                scan_mesh.compute_vertex_normals()

            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(scan_mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh_t)

            res_th = noise_params.get("resolution_theta_deg", 0.1)
            res_ph = noise_params.get("resolution_phi_deg", 0.1)
            rays, directions = self._create_spherical_rays(origin, res_th, res_ph)

            rays_tensor = o3d.core.Tensor(rays)
            ans = scene.cast_rays(rays_tensor)

            t_hit = ans["t_hit"].numpy()
            hit_mask = np.isfinite(t_hit)

            hit_distances = t_hit[hit_mask]
            hit_directions = directions[hit_mask]

            distance_std = noise_params.get("distance_noise_std", 0.002)
            if distance_std > 0:
                noise = np.random.normal(0, distance_std, hit_distances.shape)
                hit_distances += noise

            beam_div_std = noise_params.get("beam_divergence_noise_std", 0.0)

            origins_hit = rays[hit_mask, :3]
            points = origins_hit + hit_directions * hit_distances[:, np.newaxis]

            if beam_div_std > 0:
                spatial_noise = np.random.normal(0, beam_div_std, points.shape)
                points += spatial_noise

            primitive_normals = ans["primitive_normals"].numpy()[hit_mask]
            incidence_dot = np.abs(np.sum(hit_directions * primitive_normals, axis=1))
            distance_decay = np.clip((1.0 / (hit_distances ** 2)) * 100, 0, 1)
            intensities = np.clip(incidence_dot * distance_decay, 0, 1)

            points_with_intensity = np.hstack([points, intensities[:, np.newaxis]])
            all_points.append(points_with_intensity)

            if face_ids_arr is not None:
                prim_ids = ans["primitive_ids"].numpy()[hit_mask].astype(np.int64)
                valid_prim = (prim_ids >= 0) & (prim_ids < len(face_ids_arr))
                inst = np.full(prim_ids.shape, GROUND_INSTANCE_ID, dtype=np.int32)
                inst[valid_prim] = face_ids_arr[prim_ids[valid_prim]]
                all_instance_ids.append(inst)

        if not all_points:
            print("Warning: No points generated.")
            return None

        merged_points = np.vstack(all_points)
        coords = merged_points[:, :3]
        intensities = merged_points[:, 3]

        header = laspy.LasHeader(point_format=1, version="1.2")
        las = laspy.LasData(header)
        las.x = coords[:, 0]
        las.y = coords[:, 1]
        las.z = coords[:, 2]
        las.intensity = (intensities * 65535).astype(np.uint16)

        out_path = self.output_dir / f"{output_filename}.laz"
        las.write(str(out_path))
        print(f"Saved simulated point cloud ({len(las.x)} points) to: {out_path}")

        if face_ids_arr is not None and all_instance_ids:
            instances = np.concatenate(all_instance_ids).astype(np.int32)
            if len(instances) != len(coords):
                raise RuntimeError(
                    f"Instance label count ({len(instances)}) != point count ({len(coords)})"
                )
            if instances_output_path is None:
                instances_path = self.output_dir / f"{output_filename}_instances.npy"
            else:
                instances_path = Path(instances_output_path)
            instances_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(instances_path), instances)
            print(
                f"Saved instance labels ({len(instances)} pts, "
                f"{len(np.unique(instances[instances >= 0]))} trees) to: {instances_path}"
            )

        return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--out_name", type=str, default="simulated_scan")
    parser.add_argument("--num_scans", type=int, default=5)
    parser.add_argument("--area_size", type=float, default=200.0)
    parser.add_argument(
        "--face_tree_ids",
        type=str,
        default=None,
        help="Optional path to {scene}_face_tree_ids.npy",
    )
    args = parser.parse_args()

    sim = Open3DSimulator()

    import random

    positions = []
    for _ in range(args.num_scans):
        x = random.uniform(-args.area_size / 2, args.area_size / 2)
        y = random.uniform(-args.area_size / 2, args.area_size / 2)
        positions.append([x, y, 1.5])

    noise = {
        "resolution_theta_deg": 0.05,
        "resolution_phi_deg": 0.05,
        "distance_noise_std": 0.005,
        "wind_sway_std": 0.02,
        "beam_divergence_noise_std": 0.005,
        "voxel_downsample_size": 0.01,
    }

    sim.scan(
        args.mesh_path,
        positions,
        noise,
        args.out_name,
        face_tree_ids=args.face_tree_ids,
    )
