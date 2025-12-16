"""Time-series animation pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from qubic.visualization.adapters.timeseries import TimeSeriesAdapter
from qubic.visualization.backends.headless_backend import HeadlessBackend
from qubic.visualization.core.camera import Camera

logger = logging.getLogger(__name__)


class TimeSeriesPipeline:
    """Pipeline for rendering time-series animations.

    Renders sequences of visualization frames and exports as video or GIF.
    """

    def __init__(
        self,
        backend: str = "headless",
        figsize: tuple[int, int] = (10, 8),
        dpi: int = 100,
    ) -> None:
        """Initialize time-series pipeline.

        Args:
            backend: Rendering backend (defaults to 'headless' for batch processing)
            figsize: Figure size in inches
            dpi: Resolution in dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi

        # Use headless backend for animation rendering
        self.backend = HeadlessBackend(figsize=figsize, dpi=dpi)

    def render_animation(
        self,
        adapter: TimeSeriesAdapter,
        output_path: str | Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        fps: int = 10,
        format: str = "mp4",
    ) -> None:
        """Render time-series as animation.

        Args:
            adapter: TimeSeriesAdapter with loaded data
            output_path: Output file path for animation
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings (same for all frames)
            colormap: Colormap name
            fps: Frames per second
            format: Output format ('mp4' or 'gif')

        Raises:
            ValueError: If format is not supported
            RuntimeError: If no timesteps are loaded
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_timesteps = adapter.get_num_timesteps()
        if n_timesteps == 0:
            raise RuntimeError("No timesteps loaded in adapter")

        logger.info(f"Rendering animation with {n_timesteps} frames at {fps} FPS")

        # Render frames to temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="qubic_viz_"))
        frame_paths = []

        try:
            for i in range(n_timesteps):
                logger.info(f"Rendering frame {i+1}/{n_timesteps}")

                data = adapter.get_timestep(i)

                # Render frame
                self.backend.render(
                    data=data,
                    scalar_field=scalar_field,
                    camera=camera,
                    colormap=colormap,
                )

                # Save frame
                frame_path = temp_dir / f"frame_{i:06d}.png"
                self.backend.save(frame_path)
                frame_paths.append(frame_path)

            # Combine frames into video/GIF
            self._create_animation(frame_paths, output_path, fps, format)

            logger.info(f"Animation saved to {output_path}")

        finally:
            # Clean up temporary files
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_animation(
        self, frame_paths: list[Path], output_path: Path, fps: int, format: str
    ) -> None:
        """Create animation from frames.

        Args:
            frame_paths: List of frame image paths
            output_path: Output file path
            fps: Frames per second
            format: Output format ('mp4' or 'gif')

        Raises:
            ValueError: If format is not supported
            ImportError: If required libraries are not available
        """
        try:
            import imageio

        except ImportError:
            raise ImportError(
                "imageio is required for animation export. "
                "Install with: pip install imageio imageio-ffmpeg"
            ) from None

        if format == "mp4":
            # Use imageio-ffmpeg for MP4
            try:
                import imageio_ffmpeg  # noqa: F401
            except ImportError:
                raise ImportError(
                    "imageio-ffmpeg is required for MP4 export. "
                    "Install with: pip install imageio-ffmpeg"
                ) from None

            writer = imageio.get_writer(output_path, fps=fps, codec="libx264")

            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

            writer.close()

        elif format == "gif":
            images = [imageio.imread(str(frame_path)) for frame_path in frame_paths]
            imageio.mimsave(output_path, images, fps=fps)

        else:
            raise ValueError(f"Unsupported animation format: {format}. Choose 'mp4' or 'gif'")

    def render_frames(
        self,
        adapter: TimeSeriesAdapter,
        output_dir: str | Path,
        scalar_field: str | None = None,
        camera: Camera | None = None,
        colormap: str = "viridis",
        format: str = "png",
    ) -> list[Path]:
        """Render individual frames to directory.

        Args:
            adapter: TimeSeriesAdapter with loaded data
            output_dir: Output directory for frames
            scalar_field: Name of scalar field for color mapping
            camera: Camera settings
            colormap: Colormap name
            format: Image format for frames

        Returns:
            List of paths to rendered frames

        Raises:
            RuntimeError: If no timesteps are loaded
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_timesteps = adapter.get_num_timesteps()
        if n_timesteps == 0:
            raise RuntimeError("No timesteps loaded in adapter")

        logger.info(f"Rendering {n_timesteps} frames to {output_dir}")

        frame_paths = []

        for i in range(n_timesteps):
            data = adapter.get_timestep(i)

            self.backend.render(
                data=data,
                scalar_field=scalar_field,
                camera=camera,
                colormap=colormap,
            )

            frame_path = output_dir / f"frame_{i:06d}.{format}"
            self.backend.save(frame_path)
            frame_paths.append(frame_path)

        logger.info(f"Rendered {len(frame_paths)} frames")

        return frame_paths
