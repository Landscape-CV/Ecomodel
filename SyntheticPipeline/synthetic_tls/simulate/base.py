from abc import ABC, abstractmethod
from pathlib import Path


class BaseLiDARSimulator(ABC):
    """
    Abstract Base Class for LiDAR simulators.
    This interface ensures that we can easily swap between a custom
    Python raycaster (e.g. Open3D) and an external physical simulator like HELIOS++.
    """

    def __init__(self, output_dir="SyntheticPipeline/output/pointclouds"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def scan(self, mesh_path, scan_positions, noise_params, output_filename):
        """
        Executes a virtual LiDAR scan on the provided mesh.

        Args:
            mesh_path (str): Path to the scene mesh file.
            scan_positions (list of np.ndarray): Scanner origin locations.
            noise_params (dict): Noise configuration.
            output_filename (str): Output point cloud name (without extension).

        Returns:
            str: Path to the generated point cloud file (.las or .xyz)
        """
        pass
