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
            # Very simplistic model: add random offset to points that are higher up (leaves/thin branches)
            wind_sway_std = noise_params.get("wind_sway_std", 0.0)
            if wind_sway_std > 0:
                # Assuming z=0 is ground. More sway at higher Z.
                heights = points[:, 2]
                max_height = np.max(heights) if len(heights)>0 else 1.0
                sway_factor = np.clip(heights / max_height, 0, 1)
                sway_noise = np.random.normal(0, wind_sway_std, points.shape) * sway_factor[:, np.newaxis]
                points += sway_noise
                
            all_points.append(points)
            
        # Merge all scan positions
        if not all_points:
            print("Warning: No points generated.")
            return None
            
        merged_points = np.vstack(all_points)
        
        # Downsample to simulate realistic point density if needed
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        
        voxel_size = noise_params.get("voxel_downsample_size", 0.0)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
            
        out_path = self.output_dir / f"{output_filename}.xyz"
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"Saved simulated point cloud ({len(pcd.points)} points) to: {out_path}")
        return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--out_name", type=str, default="simulated_scan")
    args = parser.parse_args()
    
    sim = Open3DSimulator()
    
    # Example setup: 3 scan positions around the forest
    positions = [
        [0, 0, 1.5],     # Center
        [10, 10, 1.5],   # Corner
        [-10, -10, 1.5]  # Opposite Corner
    ]
    
    noise = {
        "resolution_theta_deg": 0.05,
        "resolution_phi_deg": 0.05,
        "distance_noise_std": 0.005, # 5mm
        "wind_sway_std": 0.02, # 2cm sway at top
        "beam_divergence_noise_std": 0.005,
        "voxel_downsample_size": 0.01 # 1cm grid
    }
    
    sim.scan(args.mesh_path, positions, noise, args.out_name)
