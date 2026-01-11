"""IPC Messaging System for QRATUM-Chess.

Provides structured messaging between GPU layer and game engines
for commands, events, and state synchronization.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from collections import deque


class MessageType(Enum):
    """IPC message types."""
    # Control messages
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    
    # State updates
    BOARD_UPDATE = "board_update"
    EVALUATION_UPDATE = "evaluation_update"
    PV_UPDATE = "pv_update"
    CLOCK_UPDATE = "clock_update"
    
    # Cortex data
    CORTEX_HEATMAP = "cortex_heatmap"
    CORTEX_CONTRIBUTION = "cortex_contribution"
    CORTEX_ENTROPY = "cortex_entropy"
    
    # Search tree
    TREE_UPDATE = "tree_update"
    TREE_NODE_ADD = "tree_node_add"
    TREE_NODE_EXPAND = "tree_node_expand"
    TREE_PV_HIGHLIGHT = "tree_pv_highlight"
    
    # Motifs
    MOTIF_DETECTED = "motif_detected"
    NOVELTY_EVENT = "novelty_event"
    
    # Quantum
    QUANTUM_STATE = "quantum_state"
    QUANTUM_AMPLITUDE = "quantum_amplitude"
    QUANTUM_DECISION = "quantum_decision"
    
    # Anti-holographic
    AH_STOCHASTICITY = "ah_stochasticity"
    AH_DESTABILIZATION = "ah_destabilization"
    AH_METRICS = "ah_metrics"
    
    # Telemetry
    TELEMETRY_PERFORMANCE = "telemetry_performance"
    TELEMETRY_CACHE = "telemetry_cache"
    TELEMETRY_MEMORY = "telemetry_memory"
    
    # Commands (from engine to core)
    CMD_NEW_GAME = "cmd_new_game"
    CMD_STOP_GAME = "cmd_stop_game"
    CMD_MAKE_MOVE = "cmd_make_move"
    CMD_SET_POSITION = "cmd_set_position"
    CMD_START_SEARCH = "cmd_start_search"
    CMD_STOP_SEARCH = "cmd_stop_search"
    CMD_CAPTURE_SNAPSHOT = "cmd_capture_snapshot"
    
    # Responses
    RESPONSE_OK = "response_ok"
    RESPONSE_ERROR = "response_error"


@dataclass
class IPCMessage:
    """Inter-process communication message.
    
    Attributes:
        msg_type: Message type
        payload: Message payload (JSON-serializable)
        timestamp: Message timestamp
        msg_id: Unique message ID
        reply_to: ID of message this is replying to
        priority: Message priority (0 = normal, higher = more urgent)
    """
    msg_type: MessageType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    msg_id: str = ""
    reply_to: str = ""
    priority: int = 0
    
    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"{self.msg_type.value}_{int(self.timestamp * 1000)}"
    
    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps({
            'type': self.msg_type.value,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'id': self.msg_id,
            'reply_to': self.reply_to,
            'priority': self.priority,
        })
    
    @classmethod
    def from_json(cls, data: str) -> 'IPCMessage':
        """Deserialize message from JSON."""
        obj = json.loads(data)
        return cls(
            msg_type=MessageType(obj['type']),
            payload=obj.get('payload', {}),
            timestamp=obj.get('timestamp', time.time()),
            msg_id=obj.get('id', ''),
            reply_to=obj.get('reply_to', ''),
            priority=obj.get('priority', 0),
        )
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for network/IPC."""
        return self.to_json().encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'IPCMessage':
        """Deserialize from bytes."""
        return cls.from_json(data.decode('utf-8'))


class MessageQueue:
    """Thread-safe message queue with priority support.
    
    Messages with higher priority are processed first.
    """
    
    def __init__(self, max_size: int = 10000) -> None:
        """Initialize message queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queues: dict[int, deque[IPCMessage]] = {}  # Priority -> queue
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._count = 0
    
    def put(self, message: IPCMessage, block: bool = True, timeout: float | None = None) -> bool:
        """Add message to queue.
        
        Args:
            message: Message to add
            block: Block if queue is full
            timeout: Timeout for blocking
            
        Returns:
            True if added successfully
        """
        with self._lock:
            if self._count >= self.max_size:
                if not block:
                    return False
                # In production would wait for space
                return False
            
            priority = message.priority
            if priority not in self._queues:
                self._queues[priority] = deque()
            
            self._queues[priority].append(message)
            self._count += 1
            self._not_empty.notify()
            
            return True
    
    def get(self, block: bool = True, timeout: float | None = None) -> IPCMessage | None:
        """Get highest priority message from queue.
        
        Args:
            block: Block if queue is empty
            timeout: Timeout for blocking
            
        Returns:
            Message or None
        """
        with self._not_empty:
            if self._count == 0:
                if not block:
                    return None
                self._not_empty.wait(timeout)
                if self._count == 0:
                    return None
            
            # Get from highest priority queue
            for priority in sorted(self._queues.keys(), reverse=True):
                if self._queues[priority]:
                    message = self._queues[priority].popleft()
                    self._count -= 1
                    return message
            
            return None
    
    def get_nowait(self) -> IPCMessage | None:
        """Get message without blocking."""
        return self.get(block=False)
    
    def peek(self) -> IPCMessage | None:
        """Peek at highest priority message without removing."""
        with self._lock:
            for priority in sorted(self._queues.keys(), reverse=True):
                if self._queues[priority]:
                    return self._queues[priority][0]
            return None
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return self._count
    
    def clear(self) -> None:
        """Clear all messages."""
        with self._lock:
            self._queues.clear()
            self._count = 0


class MessageRouter:
    """Routes messages to registered handlers.
    
    Provides:
    - Handler registration by message type
    - Wildcard handlers
    - Async message processing
    """
    
    def __init__(self) -> None:
        """Initialize message router."""
        self._handlers: dict[MessageType, list[Callable[[IPCMessage], None]]] = {}
        self._wildcard_handlers: list[Callable[[IPCMessage], None]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._queue = MessageQueue()
    
    def register(
        self,
        msg_type: MessageType | None,
        handler: Callable[[IPCMessage], None],
    ) -> None:
        """Register a message handler.
        
        Args:
            msg_type: Message type to handle (None for all)
            handler: Handler function
        """
        if msg_type is None:
            self._wildcard_handlers.append(handler)
        else:
            if msg_type not in self._handlers:
                self._handlers[msg_type] = []
            self._handlers[msg_type].append(handler)
    
    def unregister(
        self,
        msg_type: MessageType | None,
        handler: Callable[[IPCMessage], None],
    ) -> None:
        """Unregister a message handler.
        
        Args:
            msg_type: Message type
            handler: Handler function
        """
        if msg_type is None:
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
        else:
            if msg_type in self._handlers and handler in self._handlers[msg_type]:
                self._handlers[msg_type].remove(handler)
    
    def dispatch(self, message: IPCMessage) -> None:
        """Dispatch a message to handlers.
        
        Args:
            message: Message to dispatch
        """
        # Call type-specific handlers
        if message.msg_type in self._handlers:
            for handler in self._handlers[message.msg_type]:
                try:
                    handler(message)
                except Exception as e:
                    print(f"Handler error for {message.msg_type}: {e}")
        
        # Call wildcard handlers
        for handler in self._wildcard_handlers:
            try:
                handler(message)
            except Exception as e:
                print(f"Wildcard handler error: {e}")
    
    def enqueue(self, message: IPCMessage) -> bool:
        """Enqueue a message for async processing.
        
        Args:
            message: Message to enqueue
            
        Returns:
            True if enqueued
        """
        return self._queue.put(message)
    
    def start(self) -> None:
        """Start async message processing."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop async message processing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def _process_loop(self) -> None:
        """Message processing loop."""
        while self._running:
            message = self._queue.get(timeout=0.1)
            if message:
                self.dispatch(message)


# Predefined message creators
def create_board_update(
    fen: str,
    last_move: str | None = None,
    evaluation: float = 0.0,
) -> IPCMessage:
    """Create a board update message."""
    return IPCMessage(
        msg_type=MessageType.BOARD_UPDATE,
        payload={
            'fen': fen,
            'last_move': last_move,
            'evaluation': evaluation,
        },
    )


def create_cortex_heatmap(
    layer: str,
    values: list[list[float]],
    contribution: float,
) -> IPCMessage:
    """Create a cortex heatmap message."""
    return IPCMessage(
        msg_type=MessageType.CORTEX_HEATMAP,
        payload={
            'layer': layer,
            'values': values,
            'contribution': contribution,
        },
    )


def create_tree_update(
    nodes: list[dict],
    edges: list[tuple[int, int]],
    pv_path: list[int],
) -> IPCMessage:
    """Create a search tree update message."""
    return IPCMessage(
        msg_type=MessageType.TREE_UPDATE,
        payload={
            'nodes': nodes,
            'edges': edges,
            'pv_path': pv_path,
        },
    )


def create_telemetry(
    nps: float,
    hash_hit_rate: float,
    threads: int,
    latency_ms: float,
) -> IPCMessage:
    """Create a telemetry update message."""
    return IPCMessage(
        msg_type=MessageType.TELEMETRY_PERFORMANCE,
        payload={
            'nps': nps,
            'hash_hit_rate': hash_hit_rate,
            'threads': threads,
            'latency_ms': latency_ms,
        },
    )


def create_ah_metrics(
    stochasticity: float,
    destabilization: float,
    anti_reconstructibility: float,
    model_breaking_index: float,
) -> IPCMessage:
    """Create an anti-holographic metrics message."""
    return IPCMessage(
        msg_type=MessageType.AH_METRICS,
        payload={
            'stochasticity': stochasticity,
            'destabilization': destabilization,
            'anti_reconstructibility': anti_reconstructibility,
            'model_breaking_index': model_breaking_index,
        },
    )
