"""Deformation engine for contact patch simulation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .mesh_generator import TireMesh

# Default pressure in kPa for rolling deformation
DEFAULT_PRESSURE_KPA = 200.0


class DeformationEngine:
    """Engine for computing tire deformation.

    Args:
        stiffness: Material stiffness factor
    """

    def __init__(self, stiffness: float = 1.0) -> None:
        """Initialize deformation engine."""
        self.stiffness = stiffness

    def compute_contact_patch(
        self, mesh: TireMesh, load_kg: float, pressure_kpa: float, ground_plane_z: float = 0.0
    ) -> tuple[TireMesh, np.ndarray]:
        """Compute contact patch deformation.

        Args:
            mesh: Original tire mesh
            load_kg: Applied load in kg
            pressure_kpa: Inflation pressure in kPa
            ground_plane_z: Ground plane Z coordinate

        Returns:
            Tuple of (deformed_mesh, contact_forces)
        """
        # Create deformed copy
        deformed_vertices = mesh.vertices.copy()

        # Find vertices near contact patch
        contact_threshold = 0.05  # 5cm threshold
        contact_mask = mesh.vertices[:, 2] < (ground_plane_z + contact_threshold)

        # Compute deformation based on load and pressure
        load_n = load_kg * 9.81  # Convert to Newtons
        deformation_factor = load_n / (pressure_kpa * 1000 * self.stiffness)

        # Apply deformation to contact vertices
        contact_forces = np.zeros(len(mesh.vertices))
        for i, in_contact in enumerate(contact_mask):
            if in_contact:
                # Compute distance from ground
                distance = mesh.vertices[i, 2] - ground_plane_z

                # Apply deformation (flatten against ground)
                deformation = deformation_factor * np.exp(-distance / contact_threshold)
                deformed_vertices[i, 2] = max(ground_plane_z, mesh.vertices[i, 2] - deformation)

                # Compute contact force
                contact_forces[i] = load_n * np.exp(-distance / contact_threshold)

        # Create deformed mesh
        deformed_mesh = TireMesh(
            vertices=deformed_vertices,
            faces=mesh.faces.copy(),
            normals=mesh.normals.copy(),
            uvs=mesh.uvs.copy(),
        )

        # Recalculate normals
        deformed_mesh.recalculate_normals()

        return deformed_mesh, contact_forces

    def compute_rolling_deformation(
        self, mesh: TireMesh, rotation_angle: float, load_kg: float
    ) -> TireMesh:
        """Compute deformation during rolling.

        Args:
            mesh: Original tire mesh
            rotation_angle: Rotation angle in radians
            load_kg: Applied load in kg

        Returns:
            Deformed mesh
        """
        # Create deformed copy
        deformed_vertices = mesh.vertices.copy()

        # Rotate mesh
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        deformed_vertices = deformed_vertices @ rotation_matrix.T

        # Apply contact patch deformation
        contact_mesh, _ = self.compute_contact_patch(
            TireMesh(
                vertices=deformed_vertices,
                faces=mesh.faces,
                normals=mesh.normals,
                uvs=mesh.uvs,
            ),
            load_kg,
            DEFAULT_PRESSURE_KPA,
        )

        return contact_mesh
