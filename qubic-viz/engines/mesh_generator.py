"""Tire mesh generation from geometry specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TireMesh:
    """3D tire mesh data.

    Attributes:
        vertices: Vertex positions as (N, 3) array
        faces: Triangle faces as (M, 3) array of vertex indices
        normals: Vertex normals as (N, 3) array
        uvs: Texture coordinates as (N, 2) array
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """Get number of faces."""
        return len(self.faces)

    def compute_face_normals(self) -> np.ndarray:
        """Compute face normals.

        Returns:
            Face normals as (M, 3) array
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Compute cross product
        normals = np.cross(v1 - v0, v2 - v0)

        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (lengths + 1e-10)

        return normals

    def recalculate_normals(self) -> None:
        """Recalculate vertex normals from face normals."""
        # Initialize normals to zero
        vertex_normals = np.zeros_like(self.vertices)

        # Accumulate face normals at vertices
        face_normals = self.compute_face_normals()
        for i, face in enumerate(self.faces):
            vertex_normals[face] += face_normals[i]

        # Normalize
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        self.normals = vertex_normals / (lengths + 1e-10)


class TireMeshGenerator:
    """Generate tire meshes from geometry specifications.

    Args:
        resolution: Resolution factor (higher = more detail)
    """

    def __init__(self, resolution: int = 32) -> None:
        """Initialize mesh generator."""
        self.resolution = resolution

    def generate_tire_mesh(self, tire_geometry: Any) -> TireMesh:
        """Generate tire mesh from TireGeometry.

        Args:
            tire_geometry: TireGeometry object from quasim.domains.tire

        Returns:
            Generated tire mesh
        """
        # Extract geometry parameters
        outer_diameter = getattr(tire_geometry, "outer_diameter_mm", 700.0) / 1000.0  # Convert to m
        width = getattr(tire_geometry, "width_mm", 225.0) / 1000.0
        rim_diameter = getattr(tire_geometry, "rim_diameter_inch", 17.0) * 0.0254  # Convert to m

        # Compute torus parameters
        major_radius = outer_diameter / 2.0
        minor_radius = (outer_diameter - rim_diameter) / 4.0

        return self._generate_torus(major_radius, minor_radius, width)

    def _generate_torus(
        self, major_radius: float, minor_radius: float, width: float
    ) -> TireMesh:
        """Generate torus mesh for tire.

        Args:
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
            width: Tire width

        Returns:
            Generated mesh
        """
        # Resolution parameters
        u_segments = self.resolution * 2  # Around major circumference
        v_segments = self.resolution  # Around minor circumference

        # Generate parametric surface
        u = np.linspace(0, 2 * np.pi, u_segments, endpoint=False)
        v = np.linspace(0, 2 * np.pi, v_segments, endpoint=False)
        u_grid, v_grid = np.meshgrid(u, v)

        # Flatten for vertex generation
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()

        # Parametric equations for torus
        x = (major_radius + minor_radius * np.cos(v_flat)) * np.cos(u_flat)
        y = (major_radius + minor_radius * np.cos(v_flat)) * np.sin(u_flat)
        z = minor_radius * np.sin(v_flat)

        # Apply width scaling
        z = z * (width / (2 * minor_radius))

        vertices = np.column_stack([x, y, z])

        # Generate faces (two triangles per quad)
        faces = []
        for i in range(v_segments):
            for j in range(u_segments):
                # Current quad vertices
                v0 = i * u_segments + j
                v1 = i * u_segments + ((j + 1) % u_segments)
                v2 = ((i + 1) % v_segments) * u_segments + ((j + 1) % u_segments)
                v3 = ((i + 1) % v_segments) * u_segments + j

                # Two triangles
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces = np.array(faces)

        # Generate UVs
        uvs = np.column_stack([
            u_flat / (2 * np.pi),
            v_flat / (2 * np.pi),
        ])

        # Create mesh
        mesh = TireMesh(
            vertices=vertices,
            faces=faces,
            normals=np.zeros_like(vertices),
            uvs=uvs,
        )

        # Compute normals
        mesh.recalculate_normals()

        return mesh

    def add_tread_detail(self, mesh: TireMesh, tread_design: Optional[Any] = None) -> TireMesh:
        """Add tread pattern detail to tire mesh.

        Args:
            mesh: Base tire mesh
            tread_design: TreadDesign object (optional)

        Returns:
            Mesh with tread detail
        """
        # For now, return mesh unchanged
        # TODO: Implement tread pattern displacement
        return mesh
