import numpy as np
import argparse
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("Warning: open3d is not installed. Please 'pip install open3d' to run Open3DSimulator.")

from simulator_base import BaseLiDARSimulator

class Open3DSimulator(BaseLiDARSimulator):
    def __init__(self, output_dir="SyntheticPipeline/output/pointclouds"):
        super().__init__(output_dir)
        
    def _create_spherical_rays(self, origin, resolution_theta=0.1, resolution_phi=0.1):
        """
        Create rays mimicking a TLS spherical scan.
        theta: azimuthal angle [0, 360]
        phi: polar angle [0, 180]
        resolutions are in degrees.
        """
        thetas = np.arange(0, 360, resolution_theta)
        phis = np.arange(0, 180, resolution_phi)
        
        theta_grid, phi_grid = np.meshgrid(np.radians(thetas), np.radians(phis))
        
        # Add random jitter to break perfect concentric circle artifacts
        theta_grid += np.random.uniform(-np.radians(resolution_theta)/2, np.radians(resolution_theta)/2, theta_grid.shape)
        phi_grid += np.random.uniform(-np.radians(resolution_phi)/2, np.radians(resolution_phi)/2, phi_grid.shape)
        
        # Spherical to Cartesian directions
        dx = np.sin(phi_grid) * np.cos(theta_grid)
        dy = np.sin(phi_grid) * np.sin(theta_grid)
        dz = np.cos(phi_grid)
        
        directions = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
        origins = np.tile(origin, (directions.shape[0], 1))
        
        # Open3D expects rays as [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        return rays, directions

    def scan(self, mesh_path, scan_positions, noise_params, output_filename="simulated_scan"):
        print(f"Loading mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        
        all_points = []
        
        for pos_idx, origin in enumerate(scan_positions):
            print(f"Scanning from position {pos_idx+1}/{len(scan_positions)}: {origin}")
            
            # 1. Create Rays
            res_th = noise_params.get("resolution_theta_deg", 0.1)
            res_ph = noise_params.get("resolution_phi_deg", 0.1)
            rays, directions = self._create_spherical_rays(origin, res_th, res_ph)
            
            # 2. Cast Rays
            rays_tensor = o3d.core.Tensor(rays)
            ans = scene.cast_rays(rays_tensor)
            
            # Filter misses (distance == inf)
            t_hit = ans['t_hit'].numpy()
            hit_mask = np.isfinite(t_hit)
            
            hit_distances = t_hit[hit_mask]
            hit_directions = directions[hit_mask]
            
            # 3. Apply Noise Models
            
            # A. Distance Noise (Sensor Noise)
            distance_std = noise_params.get("distance_noise_std", 0.002) # e.g. 2mm
            if distance_std > 0:
                noise = np.random.normal(0, distance_std, hit_distances.shape)
                hit_distances += noise
                
            # B. Angular/Beam Divergence Proxy (Spatial Jitter)
            # Simulating edge scattering by adding tangential noise
            beam_div_std = noise_params.get("beam_divergence_noise_std", 0.0) 
            
            # Calculate final 3D coordinates
            origins_hit = rays[hit_mask, :3]
            points = origins_hit + hit_directions * hit_distances[:, np.newaxis]
            
            if beam_div_std > 0:
                spatial_noise = np.random.normal(0, beam_div_std, points.shape)
                points += spatial_noise
                
            # C. Wind Sway Noise
            wind_sway_std = noise_params.get("wind_sway_std", 0.0)
            if wind_sway_std > 0:
                heights = points[:, 2]
                max_height = np.max(heights) if len(heights)>0 else 1.0
                sway_factor = np.clip(heights / max_height, 0, 1)
                sway_noise = np.random.normal(0, wind_sway_std, points.shape) * sway_factor[:, np.newaxis]
                points += sway_noise
                
            # 4. Calculate Intensity based on incidence angle
            primitive_normals = ans['primitive_normals'].numpy()[hit_mask]
            
            # Incidence is dot product between ray direction and surface normal.
            # We use absolute value since rays can hit from either side of a flat leaf polygon
            incidence_dot = np.abs(np.sum(hit_directions * primitive_normals, axis=1))
            
            # Simple distance decay approximation (normalized roughly for 10-20m)
            distance_decay = np.clip((1.0 / (hit_distances ** 2)) * 100, 0, 1)
            
            # Leaves (chaotic normals, glancing hits) naturally get lower/noisier intensity
            # Trunks (vertical, direct hits) get bright reflection.
            intensities = incidence_dot * distance_decay
            intensities = np.clip(intensities, 0, 1)
            
            # Combine coordinates and intensity into 4-column array
            points_with_intensity = np.hstack([points, intensities[:, np.newaxis]])
            all_points.append(points_with_intensity)
            
        # Merge all scan positions
        if not all_points:
            print("Warning: No points generated.")
            return None
            
        merged_points = np.vstack(all_points)
        
        # Downsample to simulate realistic point density if needed
        pcd = o3d.geometry.PointCloud()
        
        coords = merged_points[:, :3]
        intensities = merged_points[:, 3]
        
        # Map intensity to grayscale colors so .ply format naturally retains it
        colors = np.zeros((len(coords), 3))
        colors[:, 0] = intensities
        colors[:, 1] = intensities
        colors[:, 2] = intensities
        
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        voxel_size = noise_params.get("voxel_downsample_size", 0.0)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
            
        out_path = self.output_dir / f"{output_filename}.ply"
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"Saved simulated point cloud ({len(pcd.points)} points) to: {out_path}")
        return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--out_name", type=str, default="simulated_scan")
    parser.add_argument("--num_scans", type=int, default=5)
    parser.add_argument("--area_size", type=float, default=200.0)
    args = parser.parse_args()
    
    sim = Open3DSimulator()
    
    import random
    positions = []
    for _ in range(args.num_scans):
        x = random.uniform(-args.area_size/2, args.area_size/2)
        y = random.uniform(-args.area_size/2, args.area_size/2)
        positions.append([x, y, 1.5])
    
    noise = {
        "resolution_theta_deg": 0.05,
        "resolution_phi_deg": 0.05,
        "distance_noise_std": 0.005, # 5mm
        "wind_sway_std": 0.02, # 2cm sway at top
        "beam_divergence_noise_std": 0.005,
        "voxel_downsample_size": 0.01 # 1cm grid
    }
    
    sim.scan(args.mesh_path, positions, noise, args.out_name)
