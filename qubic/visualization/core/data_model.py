"""Core data model for visualization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VisualizationData:
    """Unified data structure for visualization.

    This class provides a standardized container for mesh data and associated
    scalar/vector fields used in visualization.

    Attributes:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of triangle face indices
        normals: Nx3 array of vertex normals (auto-computed if None)
        scalar_fields: Dictionary of scalar fields (e.g., temperature, stress)
        vector_fields: Dictionary of vector fields (e.g., displacement, velocity)
        metadata: Additional metadata for the visualization
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    scalar_fields: dict[str, np.ndarray] = field(default_factory=dict)
    vector_fields: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data and compute normals if needed."""
        self._validate()
        if self.normals is None:
            self.normals = self._compute_normals()

    def _validate(self) -> None:
        """Validate input data dimensions and consistency.

        Raises:
            ValueError: If data dimensions are inconsistent
        """
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be Nx3 array, got shape {self.vertices.shape}")

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be Mx3 array, got shape {self.faces.shape}")

        num_vertices = len(self.vertices)

        # Validate face indices
        if np.any(self.faces < 0) or np.any(self.faces >= num_vertices):
            raise ValueError(
                f"face indices must be in range [0, {num_vertices}), "
                f"got min={self.faces.min()}, max={self.faces.max()}"
            )

        # Validate scalar fields
        for name, field_data in self.scalar_fields.items():
            if len(field_data) != num_vertices:
                raise ValueError(
                    f"scalar field '{name}' has length {len(field_data)}, "
                    f"expected {num_vertices}"
                )

        # Validate vector fields
        for name, field_data in self.vector_fields.items():
            if field_data.shape != (num_vertices, 3):
                raise ValueError(
                    f"vector field '{name}' has shape {field_data.shape}, "
                    f"expected ({num_vertices}, 3)"
                )

    def _compute_normals(self) -> np.ndarray:
        """Compute vertex normals from face data.

        Returns:
            Nx3 array of normalized vertex normals
        """
        normals = np.zeros_like(self.vertices)

        # Compute face normals and accumulate at vertices
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            # Accumulate at each vertex
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        normals = normals / norms

        return normals

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute bounding box of the mesh.

        Returns:
            Tuple of (min_coords, max_coords) as 3D arrays
        """
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def add_scalar_field(self, name: str, data: np.ndarray) -> None:
        """Add or update a scalar field.

        Args:
            name: Field name
            data: Per-vertex scalar values

        Raises:
            ValueError: If data length doesn't match number of vertices
        """
        if len(data) != len(self.vertices):
            raise ValueError(
                f"scalar field length {len(data)} doesn't match "
                f"number of vertices {len(self.vertices)}"
            )
        self.scalar_fields[name] = data

    def add_vector_field(self, name: str, data: np.ndarray) -> None:
        """Add or update a vector field.

        Args:
            name: Field name
            data: Per-vertex 3D vectors

        Raises:
            ValueError: If data shape doesn't match (num_vertices, 3)
        """
        expected_shape = (len(self.vertices), 3)
        if data.shape != expected_shape:
            raise ValueError(
                f"vector field shape {data.shape} doesn't match " f"expected shape {expected_shape}"
            )
        self.vector_fields[name] = data

    def get_field_range(self, field_name: str) -> tuple[float, float]:
        """Get the min/max range of a scalar field.

        Args:
            field_name: Name of the scalar field

        Returns:
            Tuple of (min_value, max_value)

        Raises:
            KeyError: If field doesn't exist
        """
        if field_name not in self.scalar_fields:
            raise KeyError(f"scalar field '{field_name}' not found")
        data = self.scalar_fields[field_name]
        return float(np.min(data)), float(np.max(data))
