"""Headless PNG/MP4 run capture utility.

Captures simulation frames and encodes them as video artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from quasim.common.video import encode_gif, encode_video, save_frame


@dataclass
class RunCapture:
    """Capture simulation frames for video generation.

    Attributes:
        frames: List of captured frames
        metadata: Metadata for each frame
        output_dir: Output directory for artifacts
        frame_counter: Current frame number
    """

    frames: list[np.ndarray] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Optional[Path] = None
    frame_counter: int = 0

    def record(self, step_dict: dict[str, Any]) -> None:
        """Record a simulation step.

        Args:
            step_dict: Dictionary with step data including 'frame' key
        """
        if "frame" in step_dict:
            frame = step_dict["frame"]

            # Ensure frame is numpy array
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)

            # Ensure 3D RGB format (H, W, 3)
            if frame.ndim == 2:
                # Grayscale -> RGB
                frame = np.stack([frame, frame, frame], axis=-1)

            self.frames.append(frame)

            # Store metadata (excluding frame data)
            meta = {k: v for k, v in step_dict.items() if k != "frame"}
            meta["frame_index"] = self.frame_counter
            self.metadata.append(meta)

            self.frame_counter += 1

    def finalize(
        self,
        mp4_path: Optional[Union[str, Path]] = None,
        gif_path: Optional[Union[str, Path]] = None,
        save_pngs: bool = False,
        fps: int = 30,
    ) -> dict[str, Path]:
        """Finalize capture and encode videos.

        Args:
            mp4_path: Output path for MP4 video
            gif_path: Output path for GIF
            save_pngs: Whether to save individual PNG frames
            fps: Frames per second for video

        Returns:
            Dictionary of generated artifact paths
        """
        if not self.frames:
            return {}

        artifacts = {}

        # Set default paths if output_dir is specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            if mp4_path is None:
                mp4_path = self.output_dir / "capture.mp4"
            if gif_path is None:
                gif_path = self.output_dir / "capture.gif"

        # Encode MP4
        if mp4_path:
            mp4_path = Path(mp4_path)
            try:
                encode_video(self.frames, mp4_path, fps=fps)
                artifacts["mp4"] = mp4_path
            except Exception as e:
                print(f"Warning: Could not encode MP4: {e}")

        # Encode GIF (use subset of frames for smaller file)
        if gif_path:
            gif_path = Path(gif_path)
            try:
                # Downsample frames for GIF
                step = max(1, len(self.frames) // 50)
                gif_frames = self.frames[::step]
                encode_gif(gif_frames, gif_path, fps=10)
                artifacts["gif"] = gif_path
            except Exception as e:
                print(f"Warning: Could not encode GIF: {e}")

        # Save PNGs
        if save_pngs and self.output_dir:
            png_dir = self.output_dir / "frames"
            png_dir.mkdir(parents=True, exist_ok=True)

            for i, frame in enumerate(self.frames):
                png_path = png_dir / f"frame_{i:04d}.png"
                try:
                    save_frame(frame, png_path)
                except Exception as e:
                    print(f"Warning: Could not save frame {i}: {e}")

            artifacts["png_dir"] = png_dir

        return artifacts

    def clear(self) -> None:
        """Clear all captured frames and metadata."""
        self.frames.clear()
        self.metadata.clear()
        self.frame_counter = 0


def create_dummy_frame(
    width: int = 640,
    height: int = 480,
    step: int = 0,
    pattern: str = "checkerboard",
) -> np.ndarray:
    """Create a dummy visualization frame for testing.

    Args:
        width: Frame width
        height: Frame height
        step: Step number for animation
        pattern: Pattern type ('checkerboard', 'gradient', 'noise')

    Returns:
        RGB frame as numpy array
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    if pattern == "checkerboard":
        # Animated checkerboard
        square_size = 40
        offset = (step * 2) % (square_size * 2)
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i + j + offset) // square_size) % 2 == 0:
                    frame[i : i + square_size, j : j + square_size] = [255, 255, 255]

    elif pattern == "gradient":
        # Animated gradient
        phase = (step / 100.0) * 2 * np.pi
        x = np.linspace(0, 2 * np.pi + phase, width)
        y = np.linspace(0, 2 * np.pi + phase, height)
        X, Y = np.meshgrid(x, y)

        R = (np.sin(X) * 127 + 128).astype(np.uint8)
        G = (np.sin(Y) * 127 + 128).astype(np.uint8)
        B = (np.sin(X + Y) * 127 + 128).astype(np.uint8)

        frame = np.stack([R, G, B], axis=-1)

    elif pattern == "noise":
        # Random noise with seed based on step
        rng = np.random.RandomState(step)
        frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)

    return frame
