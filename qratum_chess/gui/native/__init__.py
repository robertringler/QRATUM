"""Native GPU Rendering Layer for QRATUM-Chess.

This module provides high-performance GPU-accelerated rendering using
OpenGL/Vulkan for the Bob visualization system. Targets 120+ FPS with
low-latency frame updates.

Components:
- GPURenderer: Core rendering engine
- ShaderManager: GPU shader compilation and management
- BufferManager: GPU buffer allocation and updates
- TextureManager: Texture atlas and heatmap rendering
"""

from __future__ import annotations

__all__ = [
    "GPURenderer",
    "GPUConfig",
    "RenderTarget",
    "ShaderProgram",
    "BufferManager",
    "TextureAtlas",
]

from qratum_chess.gui.native.renderer import GPURenderer, GPUConfig, RenderTarget
from qratum_chess.gui.native.shaders import ShaderProgram, ShaderManager
from qratum_chess.gui.native.buffers import BufferManager, GPUBuffer
from qratum_chess.gui.native.textures import TextureAtlas, TextureManager
