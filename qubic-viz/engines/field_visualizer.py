"""Field visualization for thermal, stress, and other scalar fields."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .mesh_generator import TireMesh


class FieldVisualizer:
    """Visualize scalar fields on tire mesh.

    Args:
        mesh: Tire mesh to visualize on
    """

    def __init__(self, mesh: TireMesh) -> None:
        """Initialize field visualizer."""
        self.mesh = mesh

    def visualize_scalar_field(
        self,
        field_data: np.ndarray,
        field_name: str = "Field",
        colormap: str = "viridis",
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Visualize a scalar field on the mesh.

        Args:
            field_data: Scalar values at each vertex
            field_name: Name of the field for labeling
            colormap: Matplotlib colormap name
            output_path: Optional path to save image

        Returns:
            Rendered image as RGB array
        """
        if len(field_data) != self.mesh.num_vertices:
            raise ValueError(
                f"Field data size ({len(field_data)}) must match vertex count ({self.mesh.num_vertices})"
            )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Normalize field data
        vmin = field_data.min()
        vmax = field_data.max()

        # Plot vertices with field colors
        scatter = ax.scatter(
            self.mesh.vertices[:, 0],
            self.mesh.vertices[:, 1],
            self.mesh.vertices[:, 2],
            c=field_data,
            cmap=colormap,
            s=5,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label=field_name, shrink=0.5)

        # Set labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{field_name} Distribution")

        # Set viewing angle
        ax.view_init(elev=20, azim=45)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return image

    def visualize_vector_field(
        self,
        vector_data: np.ndarray,
        field_name: str = "Vector Field",
        output_path: Optional[Path] = None,
        subsample: int = 10,
    ) -> np.ndarray:
        """Visualize a vector field on the mesh.

        Args:
            vector_data: Vector values at each vertex (N, 3)
            field_name: Name of the field for labeling
            output_path: Optional path to save image
            subsample: Subsample factor for arrow density

        Returns:
            Rendered image as RGB array
        """
        if vector_data.shape[0] != self.mesh.num_vertices:
            raise ValueError("Vector data size must match vertex count")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Subsample for visualization
        indices = np.arange(0, self.mesh.num_vertices, subsample)
        positions = self.mesh.vertices[indices]
        vectors = vector_data[indices]

        # Plot mesh surface (lightly)
        ax.scatter(
            self.mesh.vertices[:, 0],
            self.mesh.vertices[:, 1],
            self.mesh.vertices[:, 2],
            c="lightgray",
            s=1,
            alpha=0.3,
        )

        # Plot vector field
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            length=0.1,
            normalize=True,
            color="red",
            alpha=0.7,
        )

        # Set labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{field_name} Distribution")

        # Set viewing angle
        ax.view_init(elev=20, azim=45)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return image

    def create_heatmap_2d(
        self,
        field_data: np.ndarray,
        field_name: str = "Field",
        projection: str = "top",
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Create 2D heatmap projection of field.

        Args:
            field_data: Scalar values at each vertex
            field_name: Name of the field for labeling
            projection: Projection direction ("top", "side", "front")
            output_path: Optional path to save image

        Returns:
            Rendered heatmap image
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Select projection coordinates
        if projection == "top":
            x, y = self.mesh.vertices[:, 0], self.mesh.vertices[:, 1]
            xlabel, ylabel = "X (m)", "Y (m)"
        elif projection == "side":
            x, y = self.mesh.vertices[:, 1], self.mesh.vertices[:, 2]
            xlabel, ylabel = "Y (m)", "Z (m)"
        else:  # front
            x, y = self.mesh.vertices[:, 0], self.mesh.vertices[:, 2]
            xlabel, ylabel = "X (m)", "Z (m)"

        # Create scatter plot heatmap
        scatter = ax.scatter(x, y, c=field_data, cmap="hot", s=10, alpha=0.6)

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label=field_name)

        # Set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{field_name} - {projection.capitalize()} View")
        ax.set_aspect("equal")

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return image
