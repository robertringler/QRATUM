# tests/test_video_autocapture.py
"""
Tests for automatic video capture functionality in QuASIM.
"""

import os
from pathlib import Path
import shutil
import pytest
import numpy as np

from quasim.control.optimizer import optimize_a
from quasim.viz.renderer import render_frame, FlowFrameSpec


@pytest.fixture
def temp_artifacts_dir(tmp_path):
    """Create a temporary artifacts directory for testing."""
    artifacts_dir = tmp_path / "artifacts" / "flows"
    artifacts_dir.mkdir(parents=True)
    
    # Change to temp directory for test
    original_dir = Path.cwd()
    os.chdir(tmp_path)
    
    yield artifacts_dir
    
    # Restore original directory
    os.chdir(original_dir)


def test_render_frame_creates_rgb_array():
    """Test that render_frame produces a valid RGB array."""
    spec = FlowFrameSpec(
        frame_idx=0,
        time=0.0,
        control=1.0,
        objective=-2.5,
        w2=0.1,
        fr_speed=0.2,
        bures_dist=0.15,
        qfi=0.95,
        fidelity=0.98,
        free_energy=-1.5,
    )
    
    frame = render_frame(spec, dpi=50)
    
    # Check that frame is an RGB array
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.min() >= 0
    assert frame.max() <= 255


def test_video_autocapture_creates_files(temp_artifacts_dir):
    """Test that video files are created during a simulation run."""
    # Import here to ensure we're in the temp directory
    from quasim.cli.run_flow import _create_video_artifacts, _generate_video_hash
    
    # Run a small optimization
    N = 30
    a_opt, hist, logs = optimize_a(steps=3, N=N, T=1.0, seed=42)
    
    # Generate video artifacts
    repro_hash = _generate_video_hash(42, 3, N, 1.0)
    _create_video_artifacts(a_opt, hist, logs, T=1.0, N=N, repro_hash=repro_hash, num_frames=10)
    
    # Check that files exist
    mp4_file = temp_artifacts_dir / f"quasim_run_{repro_hash}.mp4"
    gif_file = temp_artifacts_dir / f"quasim_run_{repro_hash}.gif"
    latest_mp4 = temp_artifacts_dir / "quasim_run_latest.mp4"
    latest_gif = temp_artifacts_dir / "quasim_run_latest.gif"
    
    assert mp4_file.exists(), "MP4 file should be created"
    assert gif_file.exists(), "GIF file should be created"
    assert latest_mp4.exists() or latest_mp4.is_symlink(), "Latest MP4 symlink should exist"
    assert latest_gif.exists() or latest_gif.is_symlink(), "Latest GIF symlink should exist"


def test_video_file_size_threshold(temp_artifacts_dir):
    """Test that MP4 file size exceeds minimum threshold (indicates video data written)."""
    from quasim.cli.run_flow import _create_video_artifacts, _generate_video_hash
    
    # Run a small optimization
    N = 40
    a_opt, hist, logs = optimize_a(steps=3, N=N, T=1.0, seed=123)
    
    # Generate video artifacts
    repro_hash = _generate_video_hash(123, 3, N, 1.0)
    _create_video_artifacts(a_opt, hist, logs, T=1.0, N=N, repro_hash=repro_hash, num_frames=15)
    
    # Check file sizes
    mp4_file = temp_artifacts_dir / f"quasim_run_{repro_hash}.mp4"
    gif_file = temp_artifacts_dir / f"quasim_run_{repro_hash}.gif"
    
    mp4_size_mb = mp4_file.stat().st_size / (1024 * 1024)
    gif_size_mb = gif_file.stat().st_size / (1024 * 1024)
    
    # MP4 should be at least 0.05 MB (50 KB) for a short video
    assert mp4_size_mb > 0.05, f"MP4 file size {mp4_size_mb:.3f} MB is too small"
    
    # GIF should be at least 0.1 MB (100 KB)
    assert gif_size_mb > 0.1, f"GIF file size {gif_size_mb:.3f} MB is too small"


def test_reproducible_hash_generation():
    """Test that hash generation is deterministic."""
    from quasim.cli.run_flow import _generate_video_hash
    
    hash1 = _generate_video_hash(42, 150, 300, 3.0)
    hash2 = _generate_video_hash(42, 150, 300, 3.0)
    
    assert hash1 == hash2, "Hash should be reproducible with same parameters"
    
    hash3 = _generate_video_hash(43, 150, 300, 3.0)
    assert hash1 != hash3, "Hash should differ with different seed"


def test_flow_frame_spec_dataclass():
    """Test that FlowFrameSpec dataclass works correctly."""
    spec = FlowFrameSpec(
        frame_idx=5,
        time=1.5,
        control=0.95,
        objective=-3.2,
        w2=0.12,
        fr_speed=0.25,
        bures_dist=0.18,
        qfi=0.92,
        fidelity=0.97,
        free_energy=-2.1,
    )
    
    assert spec.frame_idx == 5
    assert spec.time == 1.5
    assert spec.control == 0.95
    assert spec.objective == -3.2
