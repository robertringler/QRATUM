"""Image export functionality."""

from __future__ import annotations

from pathlib import Path

from qubic.visualization.backends.headless_backend import HeadlessBackend
from qubic.visualization.core.camera import Camera
from qubic.visualization.core.data_model import VisualizationData


class ImageExporter:
    """Export visualizations to image formats (PNG, JPEG).

    Provides high-quality image export with configurable resolution
    and compression settings.
    """

    def __init__(self, dpi: int = 300, figsize: tuple[int, int] = (12, 10)) -> None:
        """Initialize image exporter.

        Args:
            dpi: Resolution in dots per inch (higher = better quality)
            figsize: Figure size in inches
        """

        self.dpi = dpi
        self.figsize = figsize
        self.backend = HeadlessBackend(figsize=figsize, dpi=dpi)

    def export_png(
        self,
        data: VisualizationData,
        output_path: Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        transparent: bool = False,
        **kwargs,
    ) -> None:
        """Export visualization as PNG.

        Args:
            data: Visualization data to export
            output_path: Output PNG file path
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            transparent: Whether to use transparent background
            **kwargs: Additional arguments for savefig
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Render
        self.backend.render(
            data=data,
            scalar_field=scalar_field,
            camera=camera,
            colormap=colormap,
        )

        # Save as PNG
        save_kwargs = {
            "format": "png",
            "dpi": self.dpi,
            "bbox_inches": "tight",
            "transparent": transparent,
        }
        save_kwargs.update(kwargs)

        self.backend.save(output_path, **save_kwargs)

    def export_jpeg(
        self,
        data: VisualizationData,
        output_path: Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        quality: int = 95,
        **kwargs,
    ) -> None:
        """Export visualization as JPEG.

        Args:
            data: Visualization data to export
            output_path: Output JPEG file path
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            quality: JPEG quality (1-100, higher = better quality)
            **kwargs: Additional arguments for savefig
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Render
        self.backend.render(
            data=data,
            scalar_field=scalar_field,
            camera=camera,
            colormap=colormap,
        )

        # Save as JPEG
        save_kwargs = {
            "format": "jpg",
            "dpi": self.dpi,
            "bbox_inches": "tight",
            "pil_kwargs": {"quality": quality},
        }
        save_kwargs.update(kwargs)

        self.backend.save(output_path, **save_kwargs)

    def export(
        self,
        data: VisualizationData,
        output_path: Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        **kwargs,
    ) -> None:
        """Export visualization with format auto-detected from extension.

        Args:
            data: Visualization data to export
            output_path: Output file path (extension determines format)
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            **kwargs: Additional arguments for export

        Raises:
            ValueError: If file extension is not supported
        """

        output_path = Path(output_path)
        extension = output_path.suffix.lower()

        if extension == ".png":
            self.export_png(data, output_path, scalar_field, camera, colormap, **kwargs)
        elif extension in (".jpg", ".jpeg"):
            self.export_jpeg(data, output_path, scalar_field, camera, colormap, **kwargs)
        else:
            raise ValueError(
                f"Unsupported image format: {extension}. Supported formats: .png, .jpg, .jpeg"
            )
