"""Generic mesh adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qubic.visualization.adapters.base import SimulationAdapter
from qubic.visualization.core.data_model import VisualizationData


class MeshAdapter(SimulationAdapter):
    """Generic adapter for arbitrary mesh and field data.

    Supports:
    - Custom mesh formats
    - Generic scalar/vector field data
    - Procedural mesh generation
    """

    def load_data(self, source: str | Path | dict[str, Any]) -> VisualizationData:
        """Load generic mesh data.

        Args:
            source: File path or dictionary with mesh data

        Returns:
            VisualizationData object

        Raises:
            ValueError: If source format is invalid
        """

        if isinstance(source, (str, Path)):
            return self._load_from_file(Path(source))
        elif isinstance(source, dict):
            return self._load_from_dict(source)
        else:
            raise ValueError(
                f"Unsupported source type: {type(source)}. Expected file path or dictionary."
            )

    def _load_from_file(self, path: Path) -> VisualizationData:
        """Load mesh from file.

        Args:
            path: Path to mesh file

        Returns:
            VisualizationData object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Placeholder for file format detection and parsing
        raise NotImplementedError("File-based loading not yet implemented. Use dictionary input.")

    def _load_from_dict(self, data: dict[str, Any]) -> VisualizationData:
        """Load mesh from dictionary.

        Args:
            data: Dictionary with 'vertices', 'faces', and optional fields

        Returns:
            VisualizationData object

        Raises:
            ValueError: If required keys are missing
        """

        if "vertices" not in data or "faces" not in data:
            raise ValueError("Data must contain 'vertices' and 'faces' keys")

        vertices = np.asarray(data["vertices"])
        faces = np.asarray(data["faces"])

        # Extract fields
        scalar_fields = {}
        vector_fields = {}

        for key, value in data.items():
            if key in ("vertices", "faces", "normals"):
                continue

            field_data = np.asarray(value)

            if field_data.ndim == 1:
                scalar_fields[key] = field_data
            elif field_data.ndim == 2 and field_data.shape[1] == 3:
                vector_fields[key] = field_data

        metadata = {
            "adapter_type": "mesh",
            "num_vertices": len(vertices),
            "num_faces": len(faces),
        }

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
            metadata=metadata,
        )

    def validate_source(self, source: Any) -> bool:
        """Validate mesh data source.

        Args:
            source: Data source to validate

        Returns:
            True if source is valid
        """

        if isinstance(source, (str, Path)):
            path = Path(source)
            return path.exists()
        elif isinstance(source, dict):
            return "vertices" in source and "faces" in source
        return False

    def create_test_mesh(
        self, mesh_type: str = "sphere", resolution: int = 20
    ) -> VisualizationData:
        """Create test mesh for validation.

        Args:
            mesh_type: Type of mesh ('sphere', 'cube', 'cylinder')
            resolution: Mesh resolution

        Returns:
            VisualizationData with test mesh

        Raises:
            ValueError: If mesh type is unknown
        """

        if mesh_type == "sphere":
            return self._create_sphere(resolution)
        elif mesh_type == "cube":
            return self._create_cube()
        elif mesh_type == "cylinder":
            return self._create_cylinder(resolution)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")

    def _create_sphere(self, resolution: int = 20) -> VisualizationData:
        """Create UV sphere mesh."""

        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # Generate faces
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v0 = i * resolution + j
                v1 = i * resolution + (j + 1)
                v2 = (i + 1) * resolution + (j + 1)
                v3 = (i + 1) * resolution + j

                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces = np.array(faces)

        # Add synthetic scalar field
        distance_from_origin = np.linalg.norm(vertices, axis=1)

        return self._load_from_dict(
            {
                "vertices": vertices,
                "faces": faces,
                "distance": distance_from_origin,
            }
        )

    def _create_cube(self) -> VisualizationData:
        """Create cube mesh."""

        vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Back
                [4, 6, 5],
                [4, 7, 6],  # Front
                [0, 4, 5],
                [0, 5, 1],  # Bottom
                [3, 2, 6],
                [3, 6, 7],  # Top
                [0, 3, 7],
                [0, 7, 4],  # Left
                [1, 5, 6],
                [1, 6, 2],  # Right
            ]
        )

        return self._load_from_dict({"vertices": vertices, "faces": faces})

    def _create_cylinder(self, resolution: int = 20) -> VisualizationData:
        """Create cylinder mesh."""

        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(-1, 1, resolution)
        theta, z = np.meshgrid(theta, z)

        x = np.cos(theta)
        y = np.sin(theta)

        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # Generate faces
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v0 = i * resolution + j
                v1 = i * resolution + (j + 1) % resolution
                v2 = (i + 1) * resolution + (j + 1) % resolution
                v3 = (i + 1) * resolution + j

                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces = np.array(faces)

        return self._load_from_dict({"vertices": vertices, "faces": faces})
