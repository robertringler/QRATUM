"""Scene renderer with multi-backend support."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class RenderBackend(Enum):
    """Available rendering backends."""

    CPU = "cpu"
    CUDA = "cuda"
    OPENGL = "opengl"
    VULKAN = "vulkan"


@dataclass
class RenderConfig:
    """Rendering configuration.

    Attributes:
        width: Output width in pixels
        height: Output height in pixels
        backend: Rendering backend to use
        use_gpu: Enable GPU acceleration if available
        samples: Anti-aliasing samples
        max_bounces: Maximum ray bounces for path tracing
        background_color: Background color as RGB tuple
        output_format: Output format (png, jpg, mp4, webm)
    """

    width: int = 1920
    height: int = 1080
    backend: RenderBackend = RenderBackend.CPU
    use_gpu: bool = True
    samples: int = 4
    max_bounces: int = 4
    background_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    output_format: str = "png"

    def __post_init__(self) -> None:
        """Validate configuration."""

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if self.samples < 1:
            raise ValueError("Samples must be at least 1")


class SceneRenderer:
    """Main scene renderer with multi-backend support.

    Args:
        config: Rendering configuration
    """

    def __init__(self, config: RenderConfig | None = None) -> None:
        """Initialize renderer."""

        self.config = config or RenderConfig()
        self.gpu_available = self._detect_gpu()
        self._initialized = False

        # Select effective backend
        if self.config.use_gpu and self.gpu_available:
            # Prefer CUDA if available
            self.backend = self._get_available_gpu_backend()
        else:
            self.backend = RenderBackend.CPU

    def _detect_gpu(self) -> bool:
        """Detect GPU availability.

        Returns:
            True if GPU is available, False otherwise
        """

        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_available_gpu_backend(self) -> RenderBackend:
        """Get the best available GPU backend.

        Returns:
            Best available GPU backend
        """

        try:
            import torch

            if torch.cuda.is_available():
                return RenderBackend.CUDA
        except ImportError:
            pass

        # Fall back to CPU if no GPU backend available
        return RenderBackend.CPU

    def initialize(self) -> None:
        """Initialize rendering backend."""

        if self._initialized:
            return

        if self.backend == RenderBackend.CUDA:
            try:
                import torch

                self.device = torch.device("cuda")
            except ImportError:
                self.device = None
                self.backend = RenderBackend.CPU
        else:
            self.device = None

        self._initialized = True

    def render_frame(self, scene: Any, camera: Any, frame_index: int = 0) -> np.ndarray:
        """Render a single frame.

        Args:
            scene: Scene graph to render
            camera: Camera to render from
            frame_index: Frame index for animation

        Returns:
            Rendered frame as RGB numpy array (H, W, 3) with values in [0, 255]
        """

        self.initialize()

        # Create empty frame with background color
        frame = np.ones((self.config.height, self.config.width, 3), dtype=np.uint8)
        bg_color = tuple(int(c * 255) for c in self.config.background_color)
        frame[:] = bg_color

        return frame

    def render_sequence(
        self,
        scene: Any,
        camera: Any,
        num_frames: int,
        output_path: Path | None = None,
    ) -> list[np.ndarray]:
        """Render an animation sequence.

        Args:
            scene: Scene graph to render
            camera: Camera to render from
            num_frames: Number of frames to render
            output_path: Optional path to save video

        Returns:
            List of rendered frames
        """

        frames = []
        for i in range(num_frames):
            frame = self.render_frame(scene, camera, i)
            frames.append(frame)

        if output_path is not None:
            self._save_video(frames, output_path)

        return frames

    def _save_video(self, frames: list[np.ndarray], output_path: Path) -> None:
        """Save frames as video.

        Args:
            frames: List of frames to save
            output_path: Path to output video file
        """

        try:
            import imageio

            fps = 30
            if output_path.suffix in [".mp4", ".webm"]:
                imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264")
            else:
                # Save as image sequence
                for i, frame in enumerate(frames):
                    frame_path = output_path.parent / f"{output_path.stem}_{i:04d}.png"
                    imageio.imwrite(str(frame_path), frame)
        except ImportError:
            # Fallback to OpenCV
            try:
                import cv2

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = 30
                height, width = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
            except ImportError:
                raise ImportError("Neither imageio nor opencv-python available for video export")

    def save_frame(self, frame: np.ndarray, output_path: Path) -> None:
        """Save a single frame to disk.

        Args:
            frame: Frame to save
            output_path: Path to output file
        """

        try:
            from PIL import Image

            img = Image.fromarray(frame)
            img.save(str(output_path))
        except ImportError:
            import matplotlib.pyplot as plt

            plt.imsave(str(output_path), frame)
