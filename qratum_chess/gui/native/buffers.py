"""GPU Buffer Management for QRATUM-Chess.

Handles GPU buffer allocation, updates, and memory management
for high-performance rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class BufferType(Enum):
    """GPU buffer types."""

    VERTEX = "vertex"
    INDEX = "index"
    UNIFORM = "uniform"
    STORAGE = "storage"
    TEXTURE = "texture"


class BufferUsage(Enum):
    """Buffer usage hints."""

    STATIC = "static"  # Data set once, used many times
    DYNAMIC = "dynamic"  # Data updated frequently
    STREAM = "stream"  # Data updated every frame


@dataclass
class GPUBuffer:
    """GPU buffer representation.

    Attributes:
        name: Buffer name
        buffer_type: Type of buffer
        usage: Usage pattern
        size: Buffer size in bytes
        handle: GPU buffer handle
        data: CPU-side data copy (for debugging)
    """

    name: str
    buffer_type: BufferType
    usage: BufferUsage
    size: int
    handle: int = 0
    data: np.ndarray | None = None

    @property
    def is_valid(self) -> bool:
        """Check if buffer has valid GPU handle."""
        return self.handle > 0


@dataclass
class VertexAttribute:
    """Vertex attribute specification.

    Attributes:
        name: Attribute name
        location: Shader location
        components: Number of components (1-4)
        dtype: Data type (float32, int32, etc.)
        normalized: Whether to normalize
        offset: Offset in vertex structure
    """

    name: str
    location: int
    components: int
    dtype: str = "float32"
    normalized: bool = False
    offset: int = 0


@dataclass
class VertexLayout:
    """Vertex buffer layout specification.

    Attributes:
        attributes: List of vertex attributes
        stride: Total vertex size in bytes
    """

    attributes: list[VertexAttribute]
    stride: int

    @classmethod
    def create_position_color(cls) -> VertexLayout:
        """Create layout for position + color vertices."""
        return cls(
            attributes=[
                VertexAttribute("position", 0, 3, "float32", offset=0),
                VertexAttribute("color", 1, 4, "float32", offset=12),
            ],
            stride=28,  # 3*4 + 4*4 bytes
        )

    @classmethod
    def create_position_texcoord_color(cls) -> VertexLayout:
        """Create layout for position + texcoord + color vertices."""
        return cls(
            attributes=[
                VertexAttribute("position", 0, 3, "float32", offset=0),
                VertexAttribute("texcoord", 1, 2, "float32", offset=12),
                VertexAttribute("color", 2, 4, "float32", offset=20),
            ],
            stride=36,  # 3*4 + 2*4 + 4*4 bytes
        )


class BufferManager:
    """Manages GPU buffer allocation and updates.

    Features:
    - Buffer pool management
    - Efficient batch updates
    - Memory defragmentation
    - Usage tracking
    """

    def __init__(self, max_memory_mb: int = 256) -> None:
        """Initialize buffer manager.

        Args:
            max_memory_mb: Maximum GPU memory to use in MB
        """
        self.max_memory = max_memory_mb * 1024 * 1024
        self._buffers: dict[str, GPUBuffer] = {}
        self._allocated_memory = 0
        self._next_handle = 1

    def create_buffer(
        self,
        name: str,
        buffer_type: BufferType,
        size: int,
        usage: BufferUsage = BufferUsage.DYNAMIC,
        data: np.ndarray | None = None,
    ) -> GPUBuffer:
        """Create a new GPU buffer.

        Args:
            name: Buffer name
            buffer_type: Type of buffer
            size: Size in bytes
            usage: Usage pattern
            data: Initial data

        Returns:
            Created GPU buffer
        """
        if name in self._buffers:
            self.delete_buffer(name)

        if self._allocated_memory + size > self.max_memory:
            raise MemoryError(
                f"GPU memory limit exceeded: {self._allocated_memory + size} > {self.max_memory}"
            )

        buffer = GPUBuffer(
            name=name,
            buffer_type=buffer_type,
            usage=usage,
            size=size,
            handle=self._next_handle,
            data=data.copy() if data is not None else None,
        )

        self._buffers[name] = buffer
        self._allocated_memory += size
        self._next_handle += 1

        return buffer

    def create_vertex_buffer(
        self,
        name: str,
        vertices: np.ndarray,
        layout: VertexLayout,
        usage: BufferUsage = BufferUsage.DYNAMIC,
    ) -> GPUBuffer:
        """Create a vertex buffer with specified layout.

        Args:
            name: Buffer name
            vertices: Vertex data
            layout: Vertex layout specification
            usage: Usage pattern

        Returns:
            Created vertex buffer
        """
        size = vertices.nbytes
        return self.create_buffer(name, BufferType.VERTEX, size, usage, vertices)

    def create_index_buffer(
        self,
        name: str,
        indices: np.ndarray,
        usage: BufferUsage = BufferUsage.STATIC,
    ) -> GPUBuffer:
        """Create an index buffer.

        Args:
            name: Buffer name
            indices: Index data (uint16 or uint32)
            usage: Usage pattern

        Returns:
            Created index buffer
        """
        size = indices.nbytes
        return self.create_buffer(name, BufferType.INDEX, size, usage, indices)

    def create_uniform_buffer(
        self,
        name: str,
        size: int,
        data: np.ndarray | None = None,
    ) -> GPUBuffer:
        """Create a uniform buffer.

        Args:
            name: Buffer name
            size: Size in bytes
            data: Initial data

        Returns:
            Created uniform buffer
        """
        return self.create_buffer(name, BufferType.UNIFORM, size, BufferUsage.DYNAMIC, data)

    def update_buffer(
        self,
        name: str,
        data: np.ndarray,
        offset: int = 0,
    ) -> bool:
        """Update buffer data.

        Args:
            name: Buffer name
            data: New data
            offset: Offset in bytes

        Returns:
            True if update successful
        """
        buffer = self._buffers.get(name)
        if not buffer:
            return False

        if offset + data.nbytes > buffer.size:
            return False

        # Update CPU-side copy
        if buffer.data is not None:
            # In production, would also update GPU memory
            if offset == 0 and data.nbytes == buffer.size:
                buffer.data = data.copy()
            else:
                flat_data = buffer.data.view(np.uint8)
                flat_new = data.view(np.uint8)
                flat_data[offset : offset + len(flat_new)] = flat_new

        return True

    def get_buffer(self, name: str) -> GPUBuffer | None:
        """Get a buffer by name.

        Args:
            name: Buffer name

        Returns:
            Buffer or None
        """
        return self._buffers.get(name)

    def delete_buffer(self, name: str) -> bool:
        """Delete a buffer.

        Args:
            name: Buffer name

        Returns:
            True if deleted
        """
        buffer = self._buffers.get(name)
        if buffer:
            self._allocated_memory -= buffer.size
            del self._buffers[name]
            return True
        return False

    def clear(self) -> None:
        """Delete all buffers."""
        self._buffers.clear()
        self._allocated_memory = 0

    def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory stats
        """
        return {
            "allocated_bytes": self._allocated_memory,
            "allocated_mb": self._allocated_memory / (1024 * 1024),
            "max_bytes": self.max_memory,
            "max_mb": self.max_memory / (1024 * 1024),
            "usage_percent": self._allocated_memory / self.max_memory * 100,
            "buffer_count": len(self._buffers),
            "buffers": {
                name: {
                    "type": buf.buffer_type.value,
                    "size": buf.size,
                    "usage": buf.usage.value,
                }
                for name, buf in self._buffers.items()
            },
        }


class RingBuffer:
    """Ring buffer for streaming vertex data.

    Used for dynamic updates without stalling the GPU.
    """

    def __init__(
        self,
        manager: BufferManager,
        name: str,
        size: int,
        num_frames: int = 3,
    ) -> None:
        """Initialize ring buffer.

        Args:
            manager: Buffer manager
            name: Base buffer name
            size: Size per frame
            num_frames: Number of frames to buffer
        """
        self.manager = manager
        self.name = name
        self.frame_size = size
        self.num_frames = num_frames

        self._current_frame = 0
        self._offset = 0

        # Create backing buffer
        total_size = size * num_frames
        self.buffer = manager.create_buffer(
            name,
            BufferType.VERTEX,
            total_size,
            BufferUsage.STREAM,
        )

    def begin_frame(self) -> None:
        """Begin a new frame."""
        self._current_frame = (self._current_frame + 1) % self.num_frames
        self._offset = 0

    def write(self, data: np.ndarray) -> int:
        """Write data to ring buffer.

        Args:
            data: Data to write

        Returns:
            Offset where data was written
        """
        frame_offset = self._current_frame * self.frame_size
        write_offset = frame_offset + self._offset

        if self._offset + data.nbytes > self.frame_size:
            raise OverflowError("Ring buffer frame overflow")

        # Update buffer
        self.manager.update_buffer(self.name, data, write_offset)

        result_offset = self._offset
        self._offset += data.nbytes

        return result_offset

    def get_frame_offset(self) -> int:
        """Get current frame's base offset."""
        return self._current_frame * self.frame_size
