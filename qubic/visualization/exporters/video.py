"""Video export functionality."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from qubic.visualization.adapters.timeseries import TimeSeriesAdapter
from qubic.visualization.backends.headless_backend import HeadlessBackend
from qubic.visualization.core.camera import Camera

logger = logging.getLogger(__name__)


class VideoExporter:
    """Export time-series visualizations as video (MP4, GIF).

    Provides high-quality video export with configurable codec,
    bitrate, and frame rate settings.
    """

    def __init__(self, dpi: int = 150, figsize: tuple[int, int] = (12, 10)) -> None:
        """Initialize video exporter.

        Args:
            dpi: Resolution in dots per inch
            figsize: Figure size in inches
        """
        self.dpi = dpi
        self.figsize = figsize
        self.backend = HeadlessBackend(figsize=figsize, dpi=dpi)

    def export_mp4(
        self,
        adapter: TimeSeriesAdapter,
        output_path: Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        fps: int = 30,
        bitrate: str = "5000k",
        codec: str = "libx264",
    ) -> None:
        """Export time-series as MP4 video.

        Args:
            adapter: TimeSeriesAdapter with loaded data
            output_path: Output MP4 file path
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            fps: Frames per second
            bitrate: Video bitrate (e.g., '5000k')
            codec: Video codec (default: libx264)

        Raises:
            ImportError: If required libraries are not available
            RuntimeError: If no timesteps are loaded
        """
        try:
            import imageio
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            raise ImportError(
                "imageio and imageio-ffmpeg are required for MP4 export. "
                "Install with: pip install imageio imageio-ffmpeg"
            ) from None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_timesteps = adapter.get_num_timesteps()
        if n_timesteps == 0:
            raise RuntimeError("No timesteps loaded in adapter")

        logger.info(f"Exporting {n_timesteps} frames to MP4 at {fps} FPS")

        # Create video writer
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            bitrate=bitrate,
            quality=None,  # Use bitrate instead
        )

        try:
            for i in range(n_timesteps):
                logger.debug(f"Rendering frame {i+1}/{n_timesteps}")

                data = adapter.get_timestep(i)

                # Render frame
                self.backend.render(
                    data=data,
                    scalar_field=scalar_field,
                    camera=camera,
                    colormap=colormap,
                )

                # Save frame to buffer
                buf = io.BytesIO()
                self.backend.fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)

                # Add frame to video
                image = imageio.imread(buf)
                writer.append_data(image)

                buf.close()

        finally:
            writer.close()

        logger.info(f"MP4 video saved to {output_path}")

    def export_gif(
        self,
        adapter: TimeSeriesAdapter,
        output_path: Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        fps: int = 10,
        loop: int = 0,
    ) -> None:
        """Export time-series as animated GIF.

        Args:
            adapter: TimeSeriesAdapter with loaded data
            output_path: Output GIF file path
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            fps: Frames per second
            loop: Number of loops (0 = infinite)

        Raises:
            ImportError: If imageio is not available
            RuntimeError: If no timesteps are loaded
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for GIF export. "
                "Install with: pip install imageio"
            ) from None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_timesteps = adapter.get_num_timesteps()
        if n_timesteps == 0:
            raise RuntimeError("No timesteps loaded in adapter")

        logger.info(f"Exporting {n_timesteps} frames to GIF at {fps} FPS")

        # Render all frames
        frames = []

        for i in range(n_timesteps):
            logger.debug(f"Rendering frame {i+1}/{n_timesteps}")

            data = adapter.get_timestep(i)

            # Render frame
            self.backend.render(
                data=data,
                scalar_field=scalar_field,
                camera=camera,
                colormap=colormap,
            )

            # Save frame to buffer
            buf = io.BytesIO()
            self.backend.fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)

            image = imageio.imread(buf)
            frames.append(image)

            buf.close()

        # Save as GIF
        imageio.mimsave(output_path, frames, fps=fps, loop=loop)

        logger.info(f"GIF animation saved to {output_path}")

    def export(
        self,
        adapter: TimeSeriesAdapter,
        output_path: Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        fps: int = 30,
        **kwargs,
    ) -> None:
        """Export with format auto-detected from extension.

        Args:
            adapter: TimeSeriesAdapter with loaded data
            output_path: Output file path (extension determines format)
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            fps: Frames per second
            **kwargs: Additional format-specific arguments

        Raises:
            ValueError: If file extension is not supported
        """
        output_path = Path(output_path)
        extension = output_path.suffix.lower()

        if extension == ".mp4":
            self.export_mp4(
                adapter, output_path, scalar_field, camera, colormap, fps, **kwargs
            )
        elif extension == ".gif":
            self.export_gif(
                adapter, output_path, scalar_field, camera, colormap, fps, **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported video format: {extension}. "
                "Supported formats: .mp4, .gif"
            )
