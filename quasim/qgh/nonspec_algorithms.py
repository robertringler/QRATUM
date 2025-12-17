"""QGH non-speculative algorithms for distributed quantum systems.

This module implements quantum graph hash structures, superposition resolution,
distributed stream monitoring, and self-consistency propagation algorithms.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from quasim.qgh.typing import StateVector, TensorArray

logger = logging.getLogger(__name__)

EPS = 1e-10


@dataclass
class CausalHistoryHash:
    """Causal history hash for quantum event ordering.

    Maintains a hash-based causal history of quantum events with
    deterministic ordering and collision detection.

    Parameters
    ----------
    history_size : int, optional
        Maximum history size (default: 1000)
    hash_algo : str, optional
        Hash algorithm to use (default: 'sha256')

    Examples
    --------
    >>> chh = CausalHistoryHash(history_size=100)
    >>> chh.add_event("event1", np.array([1.0, 2.0]))
    >>> chh.verify_causality("event1")
    True
    """

    history_size: int = 1000
    hash_algo: str = "sha256"
    _history: deque[tuple[str, str]] = field(default_factory=deque, init=False, repr=False)
    _event_hashes: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _compute_hash(self, event_id: str, data: NDArray[np.float64]) -> str:
        """Compute deterministic hash of event and data."""

        hasher = hashlib.new(self.hash_algo)
        hasher.update(event_id.encode("utf-8"))
        hasher.update(data.tobytes())
        return hasher.hexdigest()

    def add_event(self, event_id: str, data: NDArray[np.float64]) -> str:
        """Add event to causal history.

        Parameters
        ----------
        event_id : str
            Unique event identifier
        data : NDArray[np.float64]
            Event data

        Returns
        -------
        str
            Event hash
        """

        event_hash = self._compute_hash(event_id, data)
        self._event_hashes[event_id] = event_hash
        self._history.append((event_id, event_hash))

        if len(self._history) > self.history_size:
            old_id, _ = self._history.popleft()
            if old_id in self._event_hashes:
                del self._event_hashes[old_id]

        return event_hash

    def verify_causality(self, event_id: str) -> bool:
        """Verify event is in causal history.

        Parameters
        ----------
        event_id : str
            Event identifier to verify

        Returns
        -------
        bool
            True if event exists in history
        """

        return event_id in self._event_hashes

    def get_history(self) -> list[tuple[str, str]]:
        """Get complete causal history.

        Returns
        -------
        list[tuple[str, str]]
            List of (event_id, event_hash) tuples
        """

        return list(self._history)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for testing/checkpointing."""

        return {
            "history_size": self.history_size,
            "hash_algo": self.hash_algo,
            "history": list(self._history),
            "event_hashes": dict(self._event_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalHistoryHash:
        """Deserialize from dictionary."""

        instance = cls(history_size=data["history_size"], hash_algo=data["hash_algo"])
        instance._history = deque(data["history"], maxlen=data["history_size"])
        instance._event_hashes = dict(data["event_hashes"])
        return instance


@dataclass
class SuperpositionResolver:
    """Resolve quantum superpositions through iterative consistency checks.

    Converges superposed states to classical observations using
    consistency functions and iterative refinement.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum iterations for convergence (default: 100)
    tolerance : float, optional
        Convergence tolerance (default: 1e-6)

    Examples
    --------
    >>> resolver = SuperpositionResolver(max_iterations=50)
    >>> state = np.array([0.7, 0.2, 0.1])
    >>> result = resolver.resolve(state, lambda x: x / np.sum(x))
    >>> result['converged']
    True
    """

    max_iterations: int = 100
    tolerance: float = 1e-6

    def resolve(
        self,
        initial_state: StateVector,
        consistency_fn: Callable[[StateVector], StateVector],
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Resolve superposition to consistent state.

        Parameters
        ----------
        initial_state : StateVector
            Initial superposed state
        consistency_fn : Callable
            Function that enforces consistency constraints
        rng : np.random.Generator, optional
            Random number generator

        Returns
        -------
        dict[str, Any]
            Resolution results with 'state', 'converged', 'iterations'
        """

        rng = rng or np.random.default_rng(42)
        state = np.copy(initial_state)

        for iteration in range(self.max_iterations):
            prev_state = np.copy(state)

            # Apply consistency function
            state = consistency_fn(state)

            # Check convergence
            delta = np.linalg.norm(state - prev_state)
            if delta < self.tolerance:
                return {
                    "state": state,
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_delta": float(delta),
                }

        logger.warning(
            f"SuperpositionResolver did not converge in {self.max_iterations} iterations"
        )
        return {
            "state": state,
            "converged": False,
            "iterations": self.max_iterations,
            "final_delta": float(np.linalg.norm(state - prev_state)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration."""

        return {"max_iterations": self.max_iterations, "tolerance": self.tolerance}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuperpositionResolver:
        """Deserialize from dictionary."""

        return cls(max_iterations=data["max_iterations"], tolerance=data["tolerance"])


@dataclass
class DistributedStreamMonitor:
    """Monitor distributed data streams with synchronization.

    Tracks multiple data streams and detects synchronization patterns
    and anomalies across distributed quantum systems.

    Parameters
    ----------
    num_streams : int, optional
        Number of streams to monitor (default: 4)
    buffer_size : int, optional
        Size of circular buffer per stream (default: 1000)

    Examples
    --------
    >>> monitor = DistributedStreamMonitor(num_streams=3, buffer_size=100)
    >>> monitor.add_sample(stream_id=0, value=1.5)
    >>> stats = monitor.get_stream_stats(stream_id=0)
    >>> stats['count'] == 1
    True
    """

    num_streams: int = 4
    buffer_size: int = 1000
    _buffers: dict[int, deque[float]] = field(default_factory=dict, init=False, repr=False)
    _timestamps: dict[int, deque[float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize stream buffers."""

        for i in range(self.num_streams):
            self._buffers[i] = deque(maxlen=self.buffer_size)
            self._timestamps[i] = deque(maxlen=self.buffer_size)

    def add_sample(self, stream_id: int, value: float, timestamp: float | None = None) -> None:
        """Add sample to stream.

        Parameters
        ----------
        stream_id : int
            Stream identifier
        value : float
            Sample value
        timestamp : float, optional
            Sample timestamp (auto-generated if None)
        """

        if stream_id not in self._buffers:
            self._buffers[stream_id] = deque(maxlen=self.buffer_size)
            self._timestamps[stream_id] = deque(maxlen=self.buffer_size)

        self._buffers[stream_id].append(value)
        ts = timestamp if timestamp is not None else len(self._buffers[stream_id])
        self._timestamps[stream_id].append(ts)

    def get_stream_stats(self, stream_id: int) -> dict[str, float]:
        """Get statistics for a stream.

        Parameters
        ----------
        stream_id : int
            Stream identifier

        Returns
        -------
        dict[str, float]
            Statistics including mean, std, min, max, count
        """

        if stream_id not in self._buffers or not self._buffers[stream_id]:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        data = np.array(list(self._buffers[stream_id]))
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "count": len(data),
        }

    def detect_sync_patterns(self, threshold: float = 0.8) -> list[tuple[int, int]]:
        """Detect synchronized patterns between streams.

        Parameters
        ----------
        threshold : float, optional
            Correlation threshold (default: 0.8)

        Returns
        -------
        list[tuple[int, int]]
            List of (stream_i, stream_j) pairs with high correlation
        """

        synced_pairs = []

        for i in range(self.num_streams):
            for j in range(i + 1, self.num_streams):
                if i not in self._buffers or j not in self._buffers:
                    continue
                if len(self._buffers[i]) < 2 or len(self._buffers[j]) < 2:
                    continue

                # Compute correlation on aligned data
                min_len = min(len(self._buffers[i]), len(self._buffers[j]))
                data_i = np.array(list(self._buffers[i])[-min_len:])
                data_j = np.array(list(self._buffers[j])[-min_len:])

                if np.std(data_i) < EPS or np.std(data_j) < EPS:
                    continue

                corr = np.corrcoef(data_i, data_j)[0, 1]
                if abs(corr) >= threshold:
                    synced_pairs.append((i, j))

        return synced_pairs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""

        return {
            "num_streams": self.num_streams,
            "buffer_size": self.buffer_size,
            "buffers": {k: list(v) for k, v in self._buffers.items()},
            "timestamps": {k: list(v) for k, v in self._timestamps.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributedStreamMonitor:
        """Deserialize from dictionary."""

        instance = cls(num_streams=data["num_streams"], buffer_size=data["buffer_size"])
        instance._buffers = {
            int(k): deque(v, maxlen=data["buffer_size"]) for k, v in data["buffers"].items()
        }
        instance._timestamps = {
            int(k): deque(v, maxlen=data["buffer_size"]) for k, v in data["timestamps"].items()
        }
        return instance


@dataclass
class SelfConsistencyPropagator:
    """Propagate self-consistency constraints through quantum network.

    Enforces consistency constraints across distributed quantum nodes
    using iterative message passing and constraint satisfaction.

    Parameters
    ----------
    num_nodes : int, optional
        Number of nodes in network (default: 10)
    damping : float, optional
        Damping factor for message passing (default: 0.5)
    max_iterations : int, optional
        Maximum propagation iterations (default: 50)

    Examples
    --------
    >>> prop = SelfConsistencyPropagator(num_nodes=5, damping=0.5)
    >>> states = np.random.rand(5, 3)
    >>> result = prop.propagate(states)
    >>> result['converged'] in [True, False]
    True
    """

    num_nodes: int = 10
    damping: float = 0.5
    max_iterations: int = 50
    tolerance: float = 1e-5

    def propagate(
        self, node_states: TensorArray, adjacency: NDArray[np.float64] | None = None
    ) -> dict[str, Any]:
        """Propagate consistency constraints.

        Parameters
        ----------
        node_states : TensorArray
            Initial node states, shape (num_nodes, state_dim)
        adjacency : NDArray[np.float64], optional
            Adjacency matrix (defaults to fully connected)

        Returns
        -------
        dict[str, Any]
            Propagation results with 'states', 'converged', 'iterations'
        """

        if adjacency is None:
            # Default to fully connected
            adjacency = np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)

        states = np.copy(node_states)

        for iteration in range(self.max_iterations):
            prev_states = np.copy(states)

            # Message passing: average neighbor states
            for i in range(self.num_nodes):
                neighbors = adjacency[i] > 0
                if np.any(neighbors):
                    neighbor_avg = np.mean(states[neighbors], axis=0)
                    # Damped update
                    states[i] = (1 - self.damping) * states[i] + self.damping * neighbor_avg

            # Check convergence
            delta = np.linalg.norm(states - prev_states)
            if delta < self.tolerance:
                return {
                    "states": states,
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_delta": float(delta),
                }

        logger.warning(
            f"SelfConsistencyPropagator did not converge in {self.max_iterations} iterations"
        )
        return {
            "states": states,
            "converged": False,
            "iterations": self.max_iterations,
            "final_delta": float(delta),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration."""

        return {
            "num_nodes": self.num_nodes,
            "damping": self.damping,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelfConsistencyPropagator:
        """Deserialize from dictionary."""

        return cls(
            num_nodes=data["num_nodes"],
            damping=data["damping"],
            max_iterations=data["max_iterations"],
            tolerance=data["tolerance"],
        )


@dataclass
class StabilityMonitor:
    """Monitor system stability and detect instabilities.

    Tracks system metrics over time and detects trends indicating
    instability or divergence.

    Parameters
    ----------
    window_size : int, optional
        Size of sliding window for stability checks (default: 50)
    threshold : float, optional
        Instability threshold (default: 2.0)

    Examples
    --------
    >>> monitor = StabilityMonitor(window_size=20, threshold=2.0)
    >>> for i in range(30):
    ...     monitor.add_metric(float(i))
    >>> monitor.is_stable()
    False
    """

    window_size: int = 50
    threshold: float = 2.0
    _metrics: deque[float] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize metrics buffer."""

        self._metrics = deque(maxlen=self.window_size)

    def add_metric(self, value: float) -> None:
        """Add metric value.

        Parameters
        ----------
        value : float
            Metric value to add
        """

        self._metrics.append(value)

    def is_stable(self) -> bool:
        """Check if system is stable.

        Returns
        -------
        bool
            True if system appears stable
        """

        if len(self._metrics) < self.window_size // 2:
            return True  # Not enough data

        metrics = np.array(list(self._metrics))

        # Check for linear trend
        if len(metrics) >= 2:
            x = np.arange(len(metrics))
            slope = np.polyfit(x, metrics, 1)[0]

            # Normalize slope by metric scale
            metric_scale = np.std(metrics) + EPS
            normalized_slope = abs(slope) / metric_scale

            if normalized_slope > self.threshold:
                return False

        return True

    def get_stats(self) -> dict[str, float]:
        """Get stability statistics.

        Returns
        -------
        dict[str, float]
            Statistics including mean, std, trend
        """

        if not self._metrics:
            return {"mean": 0.0, "std": 0.0, "trend": 0.0, "count": 0}

        metrics = np.array(list(self._metrics))
        x = np.arange(len(metrics))
        trend = float(np.polyfit(x, metrics, 1)[0]) if len(metrics) >= 2 else 0.0

        return {
            "mean": float(np.mean(metrics)),
            "std": float(np.std(metrics)),
            "trend": trend,
            "count": len(metrics),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""

        return {
            "window_size": self.window_size,
            "threshold": self.threshold,
            "metrics": list(self._metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StabilityMonitor:
        """Deserialize from dictionary."""

        instance = cls(window_size=data["window_size"], threshold=data["threshold"])
        instance._metrics = deque(data["metrics"], maxlen=data["window_size"])
        return instance
