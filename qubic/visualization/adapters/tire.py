"""Tire simulation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qubic.visualization.adapters.base import SimulationAdapter
from qubic.visualization.core.data_model import VisualizationData


class TireSimulationAdapter(SimulationAdapter):
    """Adapter for Goodyear tire simulation data.

    Supports loading tire mesh geometry and associated fields:
    - Thermal: Temperature distribution
    - Stress: Von Mises stress, principal stresses
    - Wear: Surface wear depth
    - Deformation: Displacement fields
    """

    def __init__(self) -> None:
        """Initialize tire simulation adapter."""

        self.supported_fields = {
            "temperature",
            "stress_von_mises",
            "stress_principal_1",
            "stress_principal_2",
            "stress_principal_3",
            "wear_depth",
            "displacement",
        }

    def load_data(self, source: str | Path | dict[str, Any]) -> VisualizationData:
        """Load tire simulation data.

        Args:
            source: Either a file path to simulation results or a dictionary
                   containing mesh and field data

        Returns:
            VisualizationData with tire mesh and fields

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
        """Load tire data from file.

        Args:
            path: Path to simulation output file

        Returns:
            VisualizationData object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Placeholder for actual file parsing
        # In production, this would parse Ansys CDB, VTK, or custom formats
        raise NotImplementedError("File-based loading not yet implemented. Use dictionary input.")

    def _load_from_dict(self, data: dict[str, Any]) -> VisualizationData:
        """Load tire data from dictionary.

        Args:
            data: Dictionary with keys 'vertices', 'faces', and optional field data

        Returns:
            VisualizationData object

        Raises:
            ValueError: If required keys are missing
        """

        if "vertices" not in data or "faces" not in data:
            raise ValueError("Data must contain 'vertices' and 'faces' keys")

        vertices = np.asarray(data["vertices"])
        faces = np.asarray(data["faces"])

        # Extract scalar and vector fields
        scalar_fields = {}
        vector_fields = {}

        for key, value in data.items():
            if key in ("vertices", "faces", "normals"):
                continue

            field_data = np.asarray(value)

            # Determine if scalar or vector field
            if field_data.ndim == 1:
                scalar_fields[key] = field_data
            elif field_data.ndim == 2 and field_data.shape[1] == 3:
                vector_fields[key] = field_data

        metadata = {
            "adapter_type": "tire",
            "num_vertices": len(vertices),
            "num_faces": len(faces),
            "fields": list(scalar_fields.keys()) + list(vector_fields.keys()),
        }

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
            metadata=metadata,
        )

    def validate_source(self, source: Any) -> bool:
        """Validate tire simulation data source.

        Args:
            source: Data source to validate

        Returns:
            True if source is valid for this adapter
        """

        if isinstance(source, (str, Path)):
            path = Path(source)
            return path.exists() and path.suffix in (".cdb", ".vtk", ".vtu", ".json")
        elif isinstance(source, dict):
            return "vertices" in source and "faces" in source
        return False

    def create_synthetic_tire(
        self, resolution: int = 32, include_fields: bool = True
    ) -> VisualizationData:
        """Create synthetic tire mesh for testing.

        Args:
            resolution: Angular resolution for tire geometry
            include_fields: Whether to generate synthetic field data

        Returns:
            VisualizationData with synthetic tire mesh
        """

        # Create torus mesh (simplified tire)
        major_radius = 0.3  # Distance from tire center to tube center
        minor_radius = 0.1  # Tube radius

        theta = np.linspace(0, 2 * np.pi, resolution)
        phi = np.linspace(0, 2 * np.pi, resolution)
        theta, phi = np.meshgrid(theta, phi)

        # Parametric torus equations
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)

        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # Generate faces (triangulate quad mesh)
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v0 = i * resolution + j
                v1 = i * resolution + (j + 1)
                v2 = (i + 1) * resolution + (j + 1)
                v3 = (i + 1) * resolution + j

                # Two triangles per quad
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces = np.array(faces)

        data_dict = {"vertices": vertices, "faces": faces}

        # Add synthetic field data
        if include_fields:
            # Temperature: hotter at bottom (road contact)
            temperature = 293.15 + 50 * (1 - vertices[:, 2] / vertices[:, 2].max())
            data_dict["temperature"] = temperature

            # Stress: higher at contact patch
            stress = 1e6 * np.exp(-5 * (vertices[:, 2] + minor_radius) ** 2)
            data_dict["stress_von_mises"] = stress

            # Wear: proportional to stress
            data_dict["wear_depth"] = stress / 1e7

        return self._load_from_dict(data_dict)
