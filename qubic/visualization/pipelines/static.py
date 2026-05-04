"""Static single-frame rendering pipeline."""

from __future__ import annotations

from pathlib import Path

from qubic.visualization.backends.gpu_backend import GPUBackend
from qubic.visualization.backends.headless_backend import HeadlessBackend
from qubic.visualization.backends.matplotlib_backend import MatplotlibBackend
from qubic.visualization.core.camera import Camera
from qubic.visualization.core.data_model import VisualizationData


class StaticPipeline:
    """Pipeline for single-frame static rendering.

    Renders a single visualization and exports to file or displays interactively.
    """

    def __init__(
        self,
        backend: str = "matplotlib",
        figsize: tuple[int, int] = (10, 8),
        dpi: int = 100,
    ) -> None:
        """Initialize static rendering pipeline.

        Args:
            backend: Rendering backend ('matplotlib', 'headless', 'gpu')
            figsize: Figure size in inches (width, height)
            dpi: Resolution in dots per inch

        Raises:
            ValueError: If backend is not supported
        """

        self.backend_name = backend
        self.figsize = figsize
        self.dpi = dpi

        # Initialize backend
        if backend == "matplotlib":
            self.backend = MatplotlibBackend(figsize=figsize, dpi=dpi)
        elif backend == "headless":
            self.backend = HeadlessBackend(figsize=figsize, dpi=dpi)
        elif backend == "gpu":
            self.backend = GPUBackend(figsize=figsize, dpi=dpi)
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Choose from: 'matplotlib', 'headless', 'gpu'"
            )

    def render(
        self,
        data: VisualizationData,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        show_edges: bool = False,
        alpha: float = 1.0,
    ):
        """Render visualization.

        Args:
            data: Visualization data to render
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings (uses default if None)
            colormap: Colormap name for scalar field
            show_edges: Whether to show mesh edges
            alpha: Transparency (0=transparent, 1=opaque)

        Returns:
            Rendered figure object
        """

        return self.backend.render(
            data=data,
            scalar_field=scalar_field,
            camera=camera,
            colormap=colormap,
            show_edges=show_edges,
            alpha=alpha,
        )

    def render_and_save(
        self,
        data: VisualizationData,
        output_path: str | Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        show_edges: bool = False,
        alpha: float = 1.0,
        **save_kwargs,
    ) -> None:
        """Render and save to file.

        Args:
            data: Visualization data to render
            output_path: Output file path
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            show_edges: Whether to show mesh edges
            alpha: Transparency
            **save_kwargs: Additional arguments for save operation
        """

        self.render(
            data=data,
            scalar_field=scalar_field,
            camera=camera,
            colormap=colormap,
            show_edges=show_edges,
            alpha=alpha,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.backend.save(output_path, **save_kwargs)

    def render_and_show(
        self,
        data: VisualizationData,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        show_edges: bool = False,
        alpha: float = 1.0,
    ) -> None:
        """Render and display interactively.

        Args:
            data: Visualization data to render
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            show_edges: Whether to show mesh edges
            alpha: Transparency

        Raises:
            RuntimeError: If using headless backend
        """

        self.render(
            data=data,
            scalar_field=scalar_field,
            camera=camera,
            colormap=colormap,
            show_edges=show_edges,
            alpha=alpha,
        )

        self.backend.show()

    def close(self) -> None:
        """Close backend and free resources."""

        self.backend.close()
