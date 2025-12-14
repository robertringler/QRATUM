"""Video encoding utilities using FFmpeg.

Provides functions to capture frames and encode them as MP4/GIF.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union

import numpy as np


def encode_video(
    frames: list[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = "libx264",
    quality: int = 23,
) -> None:
    """Encode list of frames as MP4 video.

    Args:
        frames: List of RGB frames as numpy arrays (H, W, 3)
        output_path: Output video path
        fps: Frames per second
        codec: Video codec (libx264 or libx265)
        quality: Quality parameter (lower is better, 0-51)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio

        # Use imageio with ffmpeg backend
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec=codec,
            quality=quality,
            macro_block_size=1,
        )

        for frame in frames:
            # Ensure uint8 format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

        writer.close()

    except ImportError:
        raise ImportError("imageio and imageio-ffmpeg required for video encoding")


def encode_gif(
    frames: list[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 10,
    loop: int = 0,
) -> None:
    """Encode list of frames as animated GIF.

    Args:
        frames: List of RGB frames as numpy arrays (H, W, 3)
        output_path: Output GIF path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio

        # Convert frames to uint8
        frames_uint8 = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frames_uint8.append(frame)

        imageio.mimsave(
            str(output_path),
            frames_uint8,
            duration=1000.0 / fps,
            loop=loop,
        )

    except ImportError:
        raise ImportError("imageio required for GIF encoding")


def save_frame(frame: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save single frame as PNG.

    Args:
        frame: RGB frame as numpy array (H, W, 3)
        output_path: Output PNG path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio

        # Ensure uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        imageio.imwrite(str(output_path), frame)

    except ImportError:
        raise ImportError("imageio required for image saving")


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available.

    Returns:
        True if FFmpeg is available
    """
    return shutil.which("ffmpeg") is not None
