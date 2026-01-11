"""Game Engine Connector for QRATUM-Chess.

Provides integration with Unity and Unreal Engine for
high-fidelity visualization of the Bob chess system.

Features:
- Unity C# interop via shared memory and sockets
- Unreal Engine C++ interop via shared memory
- Automatic buffer synchronization
- VR/AR support preparation
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import numpy as np

from qratum_chess.gui.ipc.shared_memory import SharedMemoryBridge, DoubleBuffer
from qratum_chess.gui.ipc.messaging import (
    IPCMessage, MessageType, MessageQueue, MessageRouter,
    create_board_update, create_cortex_heatmap, create_tree_update,
    create_telemetry, create_ah_metrics,
)


class EngineType(Enum):
    """Supported game engines."""
    UNITY = "unity"
    UNREAL = "unreal"
    GODOT = "godot"
    CUSTOM = "custom"


class ConnectionState(Enum):
    """Engine connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class EngineConfig:
    """Engine connector configuration.
    
    Attributes:
        engine_type: Target game engine
        host: Socket host for messaging
        port: Socket port
        shared_memory_prefix: Prefix for shared memory names
        buffer_width: Image buffer width
        buffer_height: Image buffer height
        target_fps: Target update rate
        use_shared_memory: Use shared memory for buffers
        use_sockets: Use sockets for messaging
    """
    engine_type: EngineType = EngineType.UNITY
    host: str = "127.0.0.1"
    port: int = 51888
    shared_memory_prefix: str = "qratum_bob"
    buffer_width: int = 1920
    buffer_height: int = 1080
    target_fps: int = 60
    use_shared_memory: bool = True
    use_sockets: bool = True


@dataclass
class EngineStats:
    """Engine connector statistics.
    
    Attributes:
        messages_sent: Total messages sent
        messages_received: Total messages received
        buffers_written: Total buffer writes
        last_message_time: Last message timestamp
        average_latency_ms: Average message latency
        connection_uptime: Connection uptime in seconds
    """
    messages_sent: int = 0
    messages_received: int = 0
    buffers_written: int = 0
    last_message_time: float = 0.0
    average_latency_ms: float = 0.0
    connection_uptime: float = 0.0


class EngineConnector:
    """Connector for integrating with game engines.
    
    This class manages communication between the QRATUM-Chess
    GPU layer and external game engines (Unity, Unreal, etc.)
    for high-fidelity visualization.
    
    Features:
    - Shared memory buffer transfer for rendered frames
    - Socket-based messaging for commands and events
    - Automatic reconnection
    - Multi-engine support
    """
    
    def __init__(self, config: EngineConfig | None = None) -> None:
        """Initialize engine connector.
        
        Args:
            config: Connector configuration
        """
        self.config = config or EngineConfig()
        
        # State
        self._state = ConnectionState.DISCONNECTED
        self._stats = EngineStats()
        self._running = False
        self._connected_time = 0.0
        
        # Shared memory
        self._memory_bridge: SharedMemoryBridge | None = None
        self._buffers: dict[str, DoubleBuffer] = {}
        
        # Sockets
        self._socket: socket.socket | None = None
        self._recv_thread: threading.Thread | None = None
        
        # Message handling
        self._message_router = MessageRouter()
        self._outgoing_queue = MessageQueue()
        self._send_thread: threading.Thread | None = None
        
        # Callbacks
        self._on_connect: Callable[[], None] | None = None
        self._on_disconnect: Callable[[], None] | None = None
        self._on_message: Callable[[IPCMessage], None] | None = None
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to engine."""
        return self._state == ConnectionState.CONNECTED
    
    def set_callbacks(
        self,
        on_connect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        on_message: Callable[[IPCMessage], None] | None = None,
    ) -> None:
        """Set connection callbacks.
        
        Args:
            on_connect: Called when connected
            on_disconnect: Called when disconnected
            on_message: Called for each received message
        """
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_message = on_message
    
    def connect(self) -> bool:
        """Connect to the game engine.
        
        Returns:
            True if connection successful
        """
        if self._state == ConnectionState.CONNECTED:
            return True
        
        self._state = ConnectionState.CONNECTING
        self._running = True
        
        try:
            # Initialize shared memory
            if self.config.use_shared_memory:
                self._memory_bridge = SharedMemoryBridge(self.config.shared_memory_prefix)
                self._memory_bridge.start()
                
                # Create standard buffers
                self._create_standard_buffers()
            
            # Connect socket
            if self.config.use_sockets:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(5.0)
                
                try:
                    self._socket.connect((self.config.host, self.config.port))
                    self._socket.settimeout(0.1)  # Non-blocking for recv
                except socket.error:
                    # Start as server if connect fails
                    self._socket.close()
                    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self._socket.bind((self.config.host, self.config.port))
                    self._socket.listen(1)
                    self._socket.settimeout(0.1)
                    print(f"Listening for engine connection on {self.config.host}:{self.config.port}")
                
                # Start receive thread
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
                
                # Start send thread
                self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
                self._send_thread.start()
            
            self._state = ConnectionState.CONNECTED
            self._connected_time = time.time()
            
            # Send connect message
            self.send_message(IPCMessage(
                msg_type=MessageType.CONNECT,
                payload={
                    'engine_type': self.config.engine_type.value,
                    'version': '1.0.0',
                },
            ))
            
            if self._on_connect:
                self._on_connect()
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            self._state = ConnectionState.ERROR
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the game engine."""
        self._running = False
        self._state = ConnectionState.DISCONNECTED
        
        # Send disconnect message
        try:
            self.send_message(IPCMessage(msg_type=MessageType.DISCONNECT))
        except Exception:
            pass
        
        # Stop threads
        if self._recv_thread:
            self._recv_thread.join(timeout=1.0)
            self._recv_thread = None
        
        if self._send_thread:
            self._send_thread.join(timeout=1.0)
            self._send_thread = None
        
        # Close socket
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        
        # Stop shared memory
        if self._memory_bridge:
            self._memory_bridge.stop()
            self._memory_bridge = None
        
        if self._on_disconnect:
            self._on_disconnect()
    
    def _create_standard_buffers(self) -> None:
        """Create standard shared memory buffers."""
        if not self._memory_bridge:
            return
        
        # Main board buffer
        self._buffers['board'] = DoubleBuffer(
            self._memory_bridge,
            'board',
            self.config.buffer_width,
            self.config.buffer_height,
        )
        
        # Cortex heatmap buffers
        for cortex in ['tactical', 'strategic', 'conceptual']:
            self._buffers[f'heatmap_{cortex}'] = DoubleBuffer(
                self._memory_bridge,
                f'heatmap_{cortex}',
                512, 512,  # Smaller for heatmaps
            )
        
        # Search tree buffer
        self._buffers['tree'] = DoubleBuffer(
            self._memory_bridge,
            'tree',
            800, 600,
        )
        
        # Quantum buffer
        self._buffers['quantum'] = DoubleBuffer(
            self._memory_bridge,
            'quantum',
            400, 300,
        )
    
    def send_message(self, message: IPCMessage) -> bool:
        """Send a message to the engine.
        
        Args:
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            return False
        
        return self._outgoing_queue.put(message)
    
    def write_buffer(self, name: str, data: np.ndarray) -> bool:
        """Write data to a shared buffer.
        
        Args:
            name: Buffer name
            data: Image data
            
        Returns:
            True if written successfully
        """
        if not self._memory_bridge or name not in self._buffers:
            return False
        
        success = self._buffers[name].write(data)
        if success:
            self._stats.buffers_written += 1
        
        return success
    
    def send_board_state(
        self,
        fen: str,
        buffer: np.ndarray | None = None,
        last_move: str | None = None,
        evaluation: float = 0.0,
    ) -> None:
        """Send board state update.
        
        Args:
            fen: Current position FEN
            buffer: Rendered board image
            last_move: Last move in UCI format
            evaluation: Position evaluation
        """
        # Send message
        self.send_message(create_board_update(fen, last_move, evaluation))
        
        # Write buffer if provided
        if buffer is not None:
            self.write_buffer('board', buffer)
    
    def send_cortex_heatmap(
        self,
        layer: str,
        values: np.ndarray,
        buffer: np.ndarray | None = None,
        contribution: float = 0.33,
    ) -> None:
        """Send cortex heatmap update.
        
        Args:
            layer: Cortex layer (tactical, strategic, conceptual)
            values: 8x8 heatmap values
            buffer: Rendered heatmap image
            contribution: Layer contribution weight
        """
        # Send message
        self.send_message(create_cortex_heatmap(layer, values.tolist(), contribution))
        
        # Write buffer if provided
        if buffer is not None:
            self.write_buffer(f'heatmap_{layer}', buffer)
    
    def send_tree_update(
        self,
        nodes: list[dict],
        edges: list[tuple[int, int]],
        pv_path: list[int],
        buffer: np.ndarray | None = None,
    ) -> None:
        """Send search tree update.
        
        Args:
            nodes: Node data list
            edges: Edge connections
            pv_path: Principal variation node IDs
            buffer: Rendered tree image
        """
        self.send_message(create_tree_update(nodes, edges, pv_path))
        
        if buffer is not None:
            self.write_buffer('tree', buffer)
    
    def send_telemetry(
        self,
        nps: float,
        hash_hit_rate: float,
        threads: int,
        latency_ms: float,
    ) -> None:
        """Send telemetry update.
        
        Args:
            nps: Nodes per second
            hash_hit_rate: Hash table hit rate
            threads: Active threads
            latency_ms: Evaluation latency
        """
        self.send_message(create_telemetry(nps, hash_hit_rate, threads, latency_ms))
    
    def send_anti_holographic(
        self,
        stochasticity: float,
        destabilization: float,
        anti_reconstructibility: float,
        model_breaking_index: float,
    ) -> None:
        """Send anti-holographic metrics.
        
        Args:
            stochasticity: Stochasticity score
            destabilization: Destabilization score
            anti_reconstructibility: Anti-reconstructibility score
            model_breaking_index: Model breaking index
        """
        self.send_message(create_ah_metrics(
            stochasticity, destabilization, anti_reconstructibility, model_breaking_index
        ))
    
    def register_handler(
        self,
        msg_type: MessageType | None,
        handler: Callable[[IPCMessage], None],
    ) -> None:
        """Register a message handler.
        
        Args:
            msg_type: Message type (None for all)
            handler: Handler function
        """
        self._message_router.register(msg_type, handler)
    
    def _recv_loop(self) -> None:
        """Receive messages from engine."""
        client_socket = self._socket
        buffer = b""
        
        while self._running:
            try:
                # Accept connection if server
                if client_socket and hasattr(client_socket, 'accept'):
                    try:
                        client_socket, _ = self._socket.accept()
                        client_socket.settimeout(0.1)
                        print("Engine connected")
                    except socket.timeout:
                        continue
                
                if not client_socket:
                    time.sleep(0.1)
                    continue
                
                # Receive data
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        time.sleep(0.01)
                        continue
                    
                    buffer += data
                    
                    # Process complete messages (newline delimited)
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        try:
                            message = IPCMessage.from_bytes(line)
                            self._stats.messages_received += 1
                            self._stats.last_message_time = time.time()
                            
                            # Dispatch message
                            self._message_router.dispatch(message)
                            
                            if self._on_message:
                                self._on_message(message)
                                
                        except Exception as e:
                            print(f"Message parse error: {e}")
                            
                except socket.timeout:
                    pass
                    
            except Exception as e:
                print(f"Recv error: {e}")
                time.sleep(0.1)
    
    def _send_loop(self) -> None:
        """Send messages to engine."""
        while self._running:
            message = self._outgoing_queue.get(timeout=0.1)
            if message and self._socket:
                try:
                    data = message.to_bytes() + b'\n'
                    self._socket.sendall(data)
                    self._stats.messages_sent += 1
                except Exception as e:
                    print(f"Send error: {e}")
    
    def get_stats(self) -> dict[str, Any]:
        """Get connector statistics.
        
        Returns:
            Statistics dictionary
        """
        if self._connected_time > 0:
            self._stats.connection_uptime = time.time() - self._connected_time
        
        return {
            'state': self._state.value,
            'engine_type': self.config.engine_type.value,
            'messages_sent': self._stats.messages_sent,
            'messages_received': self._stats.messages_received,
            'buffers_written': self._stats.buffers_written,
            'last_message_time': self._stats.last_message_time,
            'average_latency_ms': self._stats.average_latency_ms,
            'connection_uptime': self._stats.connection_uptime,
            'shared_memory': self._memory_bridge.get_stats() if self._memory_bridge else None,
        }
