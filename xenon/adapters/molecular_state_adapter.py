"""Adapter for converting MolecularState to visualization format."""

from __future__ import annotations

import numpy as np

from qubic.visualization.core.data_model import VisualizationData
from xenon.core.mechanism import MolecularState


class MolecularStateAdapter:
    """Adapter to convert MolecularState to visualization format.

    Converts individual molecular states into 3D visualizations,
    optionally representing protein structure or energy landscapes.
    """

    def __init__(self, state: MolecularState):
        """Initialize adapter with a molecular state.

        Args:
            state: MolecularState instance to adapt
        """
        self.state = state

    def to_viz_model(self) -> dict[str, any]:
        """Convert MolecularState into visualization-ready format.

        Returns:
            Dictionary with state properties for visualization
        """
        return {
            "id": self.state.state_id,
            "protein": self.state.protein_name,
            "free_energy": self.state.free_energy,
            "concentration": self.state.concentration,
            "metadata": self.state.metadata,
        }

    def to_energy_surface(
        self, resolution: int = 50, energy_scale: float = 1.0
    ) -> VisualizationData:
        """Create energy surface visualization for the molecular state.

        Generates a 3D surface representing the energy landscape around
        the molecular state.

        Args:
            resolution: Grid resolution for the surface
            energy_scale: Scaling factor for energy values

        Returns:
            VisualizationData with energy surface mesh
        """
        # Create a grid for the energy surface
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        mesh_x, mesh_y = np.meshgrid(x, y)

        # Generate energy landscape (Gaussian well centered at state)
        center_energy = self.state.free_energy
        mesh_z = center_energy + energy_scale * (mesh_x**2 + mesh_y**2)

        # Convert grid to vertices and faces
        vertices = []
        for i in range(resolution):
            for j in range(resolution):
                vertices.append([mesh_x[i, j], mesh_y[i, j], mesh_z[i, j]])
        vertices = np.array(vertices)

        # Create triangle faces for the mesh
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Two triangles per grid cell
                v0 = i * resolution + j
                v1 = i * resolution + (j + 1)
                v2 = (i + 1) * resolution + j
                v3 = (i + 1) * resolution + (j + 1)

                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = np.array(faces)

        # Create scalar field for energy values
        scalar_fields = {
            "energy": vertices[:, 2],  # Z coordinate is energy
            "concentration": np.full(len(vertices), self.state.concentration),
        }

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=scalar_fields,
            metadata={
                "state_id": self.state.state_id,
                "protein_name": self.state.protein_name,
                "center_energy": center_energy,
            },
        )

    def to_point_cloud(self, num_points: int = 1000) -> VisualizationData:
        """Create point cloud visualization for the molecular state.

        Generates a point cloud representing the molecular configuration.

        Args:
            num_points: Number of points in the cloud

        Returns:
            VisualizationData with point cloud
        """
        # Generate random points in 3D space (simplified molecular representation)
        np.random.seed(hash(self.state.state_id) % (2**32))
        positions = np.random.randn(num_points, 3)

        # Create dummy faces (point clouds don't have faces, but VisualizationData requires them)
        # Use degenerate triangles
        num_faces = max(1, num_points // 3)
        faces = np.zeros((num_faces, 3), dtype=int)
        for i in range(num_faces):
            idx = min(i * 3, num_points - 1)
            faces[i] = [idx, idx, idx]

        # Add scalar fields
        scalar_fields = {
            "free_energy": np.full(num_points, self.state.free_energy),
            "concentration": np.full(num_points, self.state.concentration),
        }

        return VisualizationData(
            vertices=positions,
            faces=faces,
            scalar_fields=scalar_fields,
            metadata={
                "state_id": self.state.state_id,
                "protein_name": self.state.protein_name,
                "representation": "point_cloud",
            },
        )
