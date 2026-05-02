"""Matplotlib-based rendering backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from qubic.visualization.core.camera import Camera
from qubic.visualization.core.data_model import VisualizationData


class MatplotlibBackend:
    """CPU-based rendering using Matplotlib.

    Provides 3D mesh visualization with scalar field color mapping
    using Matplotlib's 3D plotting capabilities.
    """

    def __init__(self, figsize: tuple[int, int] = (10, 8), dpi: int = 100) -> None:
        """Initialize matplotlib backend.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution in dots per inch
        """

        self.figsize = figsize
        self.dpi = dpi
        self.fig: Figure | None = None
        self.ax: Axes3D | None = None

    def render(
        self,
        data: VisualizationData,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        show_edges: bool = False,
        alpha: float = 1.0,
    ) -> Figure:
        """Render visualization data.

        Args:
            data: Visualization data to render
            scalar_field: Name of scalar field for color mapping (None for uniform color)
            camera: Camera settings (uses default if None)
            colormap: Matplotlib colormap name
            show_edges: Whether to show mesh edges
            alpha: Transparency (0=transparent, 1=opaque)

        Returns:
            Matplotlib figure with rendered visualization
        """

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Get vertices and faces
        vertices = data.vertices
        faces = data.faces

        # Prepare face vertices for Poly3DCollection
        face_vertices = vertices[faces]

        # Determine colors
        if scalar_field and scalar_field in data.scalar_fields:
            # Map scalar field to colors
            field_data = data.scalar_fields[scalar_field]
            vmin, vmax = data.get_field_range(scalar_field)

            # Average field values for each face
            face_colors = field_data[faces].mean(axis=1)
            face_colors_norm = (face_colors - vmin) / (vmax - vmin + 1e-10)

            # Get colormap
            cmap = plt.get_cmap(colormap)
            colors = cmap(face_colors_norm)

            # Create collection with colors
            collection = Poly3DCollection(
                face_vertices,
                facecolors=colors,
                alpha=alpha,
                edgecolor="k" if show_edges else None,
                linewidths=0.1 if show_edges else 0,
            )

            # Add colorbar
            mappable = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
            )
            mappable.set_array(field_data)
            self.fig.colorbar(mappable, ax=self.ax, label=scalar_field, shrink=0.8)

        else:
            # Uniform color
            collection = Poly3DCollection(
                face_vertices,
                facecolors="cyan",
                alpha=alpha,
                edgecolor="k" if show_edges else None,
                linewidths=0.1 if show_edges else 0,
            )

        self.ax.add_collection3d(collection)

        # Set axis limits based on bounding box
        min_coords, max_coords = data.get_bounding_box()
        center = (min_coords + max_coords) / 2
        max_range = (max_coords - min_coords).max() / 2

        self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax.set_zlim(center[2] - max_range, center[2] + max_range)

        # Apply camera settings
        if camera:
            self._apply_camera(camera, center)

        # Labels
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Title
        title = data.metadata.get("title", "Visualization")
        self.ax.set_title(title)

        return self.fig

    def _apply_camera(self, camera: Camera, center: np.ndarray) -> None:
        """Apply camera settings to the 3D axes.

        Args:
            camera: Camera configuration
            center: Center point of the scene
        """

        # Calculate view angles
        view_vec = camera.position - camera.target
        distance = np.linalg.norm(view_vec)

        # Elevation and azimuth
        azim = np.degrees(np.arctan2(view_vec[1], view_vec[0]))
        elev = np.degrees(np.arcsin(view_vec[2] / (distance + 1e-10)))

        self.ax.view_init(elev=elev, azim=azim)
        self.ax.dist = 10 / (distance + 1)  # Zoom level

    def save(self, output_path: Path, **kwargs) -> None:
        """Save rendered figure to file.

        Args:
            output_path: Output file path
            **kwargs: Additional arguments for plt.savefig
        """

        if self.fig is None:
            raise RuntimeError("No figure to save. Call render() first.")

        # Default kwargs
        save_kwargs = {"bbox_inches": "tight", "dpi": self.dpi}
        save_kwargs.update(kwargs)

        self.fig.savefig(output_path, **save_kwargs)
        plt.close(self.fig)

    def show(self) -> None:
        """Display the rendered figure interactively."""

        if self.fig is None:
            raise RuntimeError("No figure to show. Call render() first.")

        plt.show()

    def close(self) -> None:
        """Close the figure and free resources."""

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
