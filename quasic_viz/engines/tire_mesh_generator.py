"""Tire mesh generator for visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


class TireMeshGenerator:
    """Generate tire mesh geometry for visualization.

    Args:
        resolution: Mesh resolution (segments around circumference)
    """

    def __init__(self, resolution: int = 32) -> None:
        """Initialize tire mesh generator."""
        self.resolution = resolution

    def generate_tire_mesh(
        self,
        outer_diameter_mm: float = 700.0,
        width_mm: float = 225.0,
        rim_diameter_inch: float = 17.0,
    ) -> dict[str, Any]:
        """Generate tire mesh geometry.

        Args:
            outer_diameter_mm: Outer tire diameter in mm
            width_mm: Tire width in mm
            rim_diameter_inch: Rim diameter in inches

        Returns:
            Dictionary containing vertices, faces, normals
        """
        # Convert to meters
        outer_radius = outer_diameter_mm / 2000.0
        width = width_mm / 1000.0
        rim_radius = (rim_diameter_inch * 25.4) / 2000.0

        # Generate torus-like mesh
        theta = np.linspace(0, 2 * np.pi, self.resolution)
        phi = np.linspace(0, 2 * np.pi, self.resolution // 2)

        # Cross-section radius calculated from tire width and sidewall height
        cross_radius = min((outer_radius - rim_radius) / 2, width / 2)
        center_radius = rim_radius + cross_radius

        vertices = []
        for t in theta:
            for p in phi:
                x = (center_radius + cross_radius * np.cos(p)) * np.cos(t)
                y = (center_radius + cross_radius * np.cos(p)) * np.sin(t)
                z = cross_radius * np.sin(p)
                vertices.append([x, y, z])

        vertices = np.array(vertices, dtype=np.float32)

        # Generate faces
        faces = []
        n_theta = len(theta)
        n_phi = len(phi)
        for i in range(n_theta - 1):
            for j in range(n_phi - 1):
                v0 = i * n_phi + j
                v1 = i * n_phi + (j + 1)
                v2 = (i + 1) * n_phi + (j + 1)
                v3 = (i + 1) * n_phi + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces = np.array(faces, dtype=np.int32)

        # Compute normals
        normals = self._compute_normals(vertices, faces)

        return {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "num_vertices": len(vertices),
            "num_faces": len(faces),
        }

    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute vertex normals.

        Args:
            vertices: Vertex positions
            faces: Face indices

        Returns:
            Vertex normals
        """
        normals = np.zeros_like(vertices)

        for face in faces:
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normals[face] += normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms

        return normals.astype(np.float32)
