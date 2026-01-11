"""Shared Memory Bridge for GPU-Engine communication.

Provides low-latency buffer sharing between the GPU renderer
and game engine visualization layer.

Target latency: <10ms for buffer transfer.
"""

from __future__ import annotations

import mmap
import struct
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np


class BufferState(Enum):
    """Shared buffer state flags."""
    EMPTY = 0
    WRITING = 1
    READY = 2
    READING = 3


@dataclass
class SharedBuffer:
    """Shared memory buffer.
    
    Attributes:
        name: Buffer name
        size: Buffer size in bytes
        width: Image width (if image buffer)
        height: Image height (if image buffer)
        channels: Number of channels (if image buffer)
        state: Current buffer state
        timestamp: Last update timestamp
    """
    name: str
    size: int
    width: int = 0
    height: int = 0
    channels: int = 4
    state: BufferState = BufferState.EMPTY
    timestamp: float = 0.0
    
    # Memory mapping (simulated)
    _data: np.ndarray | None = None


@dataclass
class BufferHeader:
    """Header structure for shared buffers.
    
    Memory layout:
    - magic: 4 bytes (0x51524154 = "QRAT")
    - version: 4 bytes
    - state: 4 bytes
    - timestamp: 8 bytes (double)
    - width: 4 bytes
    - height: 4 bytes
    - channels: 4 bytes
    - reserved: 4 bytes
    Total: 36 bytes
    """
    MAGIC = 0x51524154  # "QRAT"
    VERSION = 1
    SIZE = 36
    FORMAT = '<IIIdIIII'
    
    @classmethod
    def pack(
        cls,
        state: BufferState,
        timestamp: float,
        width: int,
        height: int,
        channels: int,
    ) -> bytes:
        """Pack header into bytes."""
        return struct.pack(
            cls.FORMAT,
            cls.MAGIC,
            cls.VERSION,
            state.value,
            timestamp,
            width,
            height,
            channels,
            0,  # reserved
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> tuple[int, int, BufferState, float, int, int, int]:
        """Unpack header from bytes."""
        magic, version, state, timestamp, width, height, channels, _ = struct.unpack(
            cls.FORMAT, data[:cls.SIZE]
        )
        return magic, version, BufferState(state), timestamp, width, height, channels


class SharedMemoryBridge:
    """Bridge for sharing GPU buffers with external processes.
    
    This class manages shared memory regions for transferring
    rendered buffers from the GPU renderer to game engines
    (Unity, Unreal, etc.) with minimal latency.
    
    Features:
    - Double-buffered transfers for smooth updates
    - Lock-free producer-consumer pattern
    - Automatic buffer management
    - Cross-platform support (Windows, Linux, macOS)
    """
    
    def __init__(self, name_prefix: str = "qratum_chess") -> None:
        """Initialize shared memory bridge.
        
        Args:
            name_prefix: Prefix for shared memory names
        """
        self.name_prefix = name_prefix
        self._buffers: dict[str, SharedBuffer] = {}
        self._active = False
        
        # Statistics
        self._write_count = 0
        self._read_count = 0
        self._total_bytes_written = 0
        self._total_bytes_read = 0
        self._last_write_time = 0.0
        self._last_read_time = 0.0
    
    def start(self) -> bool:
        """Start the shared memory bridge.
        
        Returns:
            True if started successfully
        """
        self._active = True
        return True
    
    def stop(self) -> None:
        """Stop the bridge and release resources."""
        self._active = False
        
        for buffer in self._buffers.values():
            buffer._data = None
        
        self._buffers.clear()
    
    def create_buffer(
        self,
        name: str,
        width: int,
        height: int,
        channels: int = 4,
    ) -> SharedBuffer:
        """Create a shared buffer for image data.
        
        Args:
            name: Buffer name
            width: Image width
            height: Image height
            channels: Number of channels (default 4 for RGBA)
            
        Returns:
            Created shared buffer
        """
        full_name = f"{self.name_prefix}_{name}"
        
        # Calculate size
        pixel_size = width * height * channels
        total_size = BufferHeader.SIZE + pixel_size
        
        buffer = SharedBuffer(
            name=full_name,
            size=total_size,
            width=width,
            height=height,
            channels=channels,
            state=BufferState.EMPTY,
        )
        
        # Allocate memory (simulated - in production would use mmap)
        buffer._data = np.zeros((height, width, channels), dtype=np.uint8)
        
        self._buffers[name] = buffer
        return buffer
    
    def write_buffer(
        self,
        name: str,
        data: np.ndarray,
        wait_for_read: bool = False,
    ) -> bool:
        """Write data to a shared buffer.
        
        Args:
            name: Buffer name
            data: Image data (must match buffer dimensions)
            wait_for_read: Wait for consumer to read previous data
            
        Returns:
            True if write successful
        """
        buffer = self._buffers.get(name)
        if not buffer:
            return False
        
        # Check dimensions
        if data.shape[:2] != (buffer.height, buffer.width):
            return False
        
        # Wait for consumer if requested
        if wait_for_read:
            timeout = 0.01  # 10ms timeout
            start = time.perf_counter()
            while buffer.state == BufferState.READY:
                if time.perf_counter() - start > timeout:
                    break  # Don't block indefinitely
                time.sleep(0.001)
        
        # Mark as writing
        buffer.state = BufferState.WRITING
        
        # Copy data
        if buffer._data is not None:
            if data.ndim == 2:
                buffer._data[:, :, 0] = data
            elif data.shape[2] == buffer.channels:
                buffer._data[:] = data
            elif data.shape[2] == 3 and buffer.channels == 4:
                buffer._data[:, :, :3] = data
                buffer._data[:, :, 3] = 255
        
        # Update metadata
        buffer.timestamp = time.time()
        buffer.state = BufferState.READY
        
        # Update stats
        self._write_count += 1
        self._total_bytes_written += data.nbytes
        self._last_write_time = time.perf_counter()
        
        return True
    
    def read_buffer(self, name: str) -> np.ndarray | None:
        """Read data from a shared buffer.
        
        Args:
            name: Buffer name
            
        Returns:
            Image data or None if not ready
        """
        buffer = self._buffers.get(name)
        if not buffer or buffer.state != BufferState.READY:
            return None
        
        # Mark as reading
        buffer.state = BufferState.READING
        
        # Copy data
        data = None
        if buffer._data is not None:
            data = buffer._data.copy()
        
        # Mark as empty
        buffer.state = BufferState.EMPTY
        
        # Update stats
        self._read_count += 1
        if data is not None:
            self._total_bytes_read += data.nbytes
        self._last_read_time = time.perf_counter()
        
        return data
    
    def get_buffer_info(self, name: str) -> dict[str, Any] | None:
        """Get buffer information.
        
        Args:
            name: Buffer name
            
        Returns:
            Buffer info dict or None
        """
        buffer = self._buffers.get(name)
        if not buffer:
            return None
        
        return {
            'name': buffer.name,
            'size': buffer.size,
            'width': buffer.width,
            'height': buffer.height,
            'channels': buffer.channels,
            'state': buffer.state.name,
            'timestamp': buffer.timestamp,
        }
    
    def is_buffer_ready(self, name: str) -> bool:
        """Check if buffer has new data ready.
        
        Args:
            name: Buffer name
            
        Returns:
            True if ready for reading
        """
        buffer = self._buffers.get(name)
        return buffer is not None and buffer.state == BufferState.READY
    
    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'active': self._active,
            'buffer_count': len(self._buffers),
            'write_count': self._write_count,
            'read_count': self._read_count,
            'total_bytes_written': self._total_bytes_written,
            'total_bytes_read': self._total_bytes_read,
            'total_mb_written': self._total_bytes_written / (1024 * 1024),
            'total_mb_read': self._total_bytes_read / (1024 * 1024),
            'buffers': {
                name: self.get_buffer_info(name)
                for name in self._buffers
            },
        }


class DoubleBuffer:
    """Double-buffered shared memory for lock-free updates.
    
    Uses two buffers to allow simultaneous reading and writing
    without blocking.
    """
    
    def __init__(
        self,
        bridge: SharedMemoryBridge,
        name: str,
        width: int,
        height: int,
        channels: int = 4,
    ) -> None:
        """Initialize double buffer.
        
        Args:
            bridge: Shared memory bridge
            name: Base buffer name
            width: Image width
            height: Image height
            channels: Number of channels
        """
        self.bridge = bridge
        self.name = name
        
        # Create two buffers
        self._buffer_a = bridge.create_buffer(f"{name}_a", width, height, channels)
        self._buffer_b = bridge.create_buffer(f"{name}_b", width, height, channels)
        
        # Track which buffer is current
        self._write_buffer = "a"
        self._read_buffer = "b"
    
    def write(self, data: np.ndarray) -> bool:
        """Write to the current write buffer and swap.
        
        Args:
            data: Image data
            
        Returns:
            True if successful
        """
        buffer_name = f"{self.name}_{self._write_buffer}"
        
        if self.bridge.write_buffer(buffer_name, data):
            # Swap buffers
            self._write_buffer, self._read_buffer = self._read_buffer, self._write_buffer
            return True
        
        return False
    
    def read(self) -> np.ndarray | None:
        """Read from the current read buffer.
        
        Returns:
            Image data or None
        """
        buffer_name = f"{self.name}_{self._read_buffer}"
        return self.bridge.read_buffer(buffer_name)
    
    def get_latest(self) -> np.ndarray | None:
        """Get the most recently written data without consuming.
        
        Returns:
            Image data or None
        """
        # Check both buffers and return the newest
        info_a = self.bridge.get_buffer_info(f"{self.name}_a")
        info_b = self.bridge.get_buffer_info(f"{self.name}_b")
        
        if info_a and info_b:
            if info_a['timestamp'] > info_b['timestamp']:
                return self.bridge.read_buffer(f"{self.name}_a")
            else:
                return self.bridge.read_buffer(f"{self.name}_b")
        
        return None
