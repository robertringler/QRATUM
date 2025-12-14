"""Tests for core renderer."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add qubic-viz to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.renderer import RenderBackend, RenderConfig, SceneRenderer


def test_render_config_defaults():
    """Test RenderConfig default values."""
    config = RenderConfig()
    assert config.width == 1920
    assert config.height == 1080
    assert config.backend == RenderBackend.CPU
    assert config.use_gpu is True


def test_render_config_validation():
    """Test RenderConfig validation."""
    with pytest.raises(ValueError):
        RenderConfig(width=-1)

    with pytest.raises(ValueError):
        RenderConfig(height=0)

    with pytest.raises(ValueError):
        RenderConfig(samples=0)


def test_scene_renderer_initialization():
    """Test SceneRenderer initialization."""
    config = RenderConfig(width=800, height=600)
    renderer = SceneRenderer(config)

    assert renderer.config.width == 800
    assert renderer.config.height == 600
    assert renderer.backend in [RenderBackend.CPU, RenderBackend.CUDA]


def test_scene_renderer_gpu_detection():
    """Test GPU detection."""
    renderer = SceneRenderer()
    assert isinstance(renderer.gpu_available, bool)


def test_render_frame_basic():
    """Test basic frame rendering."""
    config = RenderConfig(width=100, height=100)
    renderer = SceneRenderer(config)

    # Render with minimal scene
    frame = renderer.render_frame(None, None, 0)

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (100, 100, 3)
    assert frame.dtype == np.uint8


def test_render_sequence():
    """Test sequence rendering."""
    config = RenderConfig(width=100, height=100)
    renderer = SceneRenderer(config)

    frames = renderer.render_sequence(None, None, num_frames=5)

    assert len(frames) == 5
    assert all(isinstance(f, np.ndarray) for f in frames)
    assert all(f.shape == (100, 100, 3) for f in frames)
