"""Inter-Process Communication Layer for QRATUM-Chess.

Provides high-speed communication between the Core GPU Layer
and Game Engine Layer (Unity/Unreal).

Features:
- Shared memory for buffer transfer
- Low-latency messaging (<10ms target)
- Serialization of render data
- Vulkan interop support
"""

from __future__ import annotations

__all__ = [
    "SharedMemoryBridge",
    "IPCMessage",
    "MessageType",
    "EngineConnector",
]

from qratum_chess.gui.ipc.engine_connector import EngineConnector, EngineType
from qratum_chess.gui.ipc.messaging import IPCMessage, MessageQueue, MessageType
from qratum_chess.gui.ipc.shared_memory import SharedBuffer, SharedMemoryBridge
