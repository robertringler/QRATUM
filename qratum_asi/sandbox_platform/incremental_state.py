"""Incremental State Evaluation for Redundancy Reduction.

Implements incremental state evaluation to reduce redundant computation
by tracking state deltas and using checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain


class DeltaType(Enum):
    """Type of state delta."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    COMPOSITE = "composite"


@dataclass
class StateDelta:
    """Delta representing a change in state.

    Attributes:
        delta_id: Unique delta identifier
        delta_type: Type of change
        path: Path to affected state element
        old_value: Previous value (None for ADD)
        new_value: New value (None for DELETE)
        timestamp: When delta was created
    """

    delta_id: str
    delta_type: DeltaType
    path: str
    old_value: Any = None
    new_value: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute deterministic hash of delta."""
        content = {
            "delta_id": self.delta_id,
            "delta_type": self.delta_type.value,
            "path": self.path,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize delta."""
        return {
            "delta_id": self.delta_id,
            "delta_type": self.delta_type.value,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "delta_hash": self.compute_hash(),
        }


@dataclass
class StateCheckpoint:
    """Checkpoint of state at a point in time.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        state_snapshot: Copy of state at checkpoint
        state_hash: Hash of state
        delta_count: Number of deltas since last checkpoint
        created_at: Checkpoint creation time
    """

    checkpoint_id: str
    state_snapshot: dict[str, Any]
    state_hash: str
    delta_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_checkpoint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "state_hash": self.state_hash,
            "delta_count": self.delta_count,
            "created_at": self.created_at,
            "parent_checkpoint_id": self.parent_checkpoint_id,
        }


class IncrementalStateEvaluator:
    """Evaluator for incremental state changes.

    Provides:
    - State tracking with deltas
    - Checkpoint management
    - Incremental evaluation (only changed state)
    - Redundancy elimination
    """

    def __init__(
        self,
        evaluator_id: str = "incremental",
        checkpoint_interval: int = 100,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize incremental state evaluator.

        Args:
            evaluator_id: Unique evaluator identifier
            checkpoint_interval: Number of deltas between checkpoints
            merkle_chain: Merkle chain for audit trail
        """
        self.evaluator_id = evaluator_id
        self.checkpoint_interval = checkpoint_interval
        self.merkle_chain = merkle_chain or MerkleChain()

        # State management
        self._state: dict[str, Any] = {}
        self._deltas: list[StateDelta] = []
        self._checkpoints: dict[str, StateCheckpoint] = {}
        self._current_checkpoint_id: str | None = None

        self._delta_counter = 0
        self._checkpoint_counter = 0
        self._lock = threading.RLock()

        # Statistics
        self._total_evaluations = 0
        self._skipped_evaluations = 0
        self._total_time_saved_ms = 0.0

        # Cached evaluation results
        self._eval_cache: dict[str, Any] = {}

        # Create initial checkpoint
        self._create_checkpoint()

        # Log initialization
        self.merkle_chain.add_event(
            "incremental_evaluator_initialized",
            {
                "evaluator_id": evaluator_id,
                "checkpoint_interval": checkpoint_interval,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def set_value(self, path: str, value: Any) -> StateDelta:
        """Set a value in the state.

        Args:
            path: Path to state element
            value: New value

        Returns:
            StateDelta describing the change
        """
        with self._lock:
            self._delta_counter += 1
            delta_id = f"delta_{self.evaluator_id}_{self._delta_counter:010d}"

            old_value = self._get_nested(self._state, path)
            delta_type = DeltaType.ADD if old_value is None else DeltaType.UPDATE

            delta = StateDelta(
                delta_id=delta_id,
                delta_type=delta_type,
                path=path,
                old_value=old_value,
                new_value=value,
            )

            self._set_nested(self._state, path, value)
            self._deltas.append(delta)

            # Invalidate affected cache entries
            self._invalidate_cache(path)

            # Check if checkpoint needed
            if len(self._deltas) >= self.checkpoint_interval:
                self._create_checkpoint()

            return delta

    def delete_value(self, path: str) -> StateDelta | None:
        """Delete a value from the state.

        Args:
            path: Path to state element

        Returns:
            StateDelta if value existed, None otherwise
        """
        with self._lock:
            old_value = self._get_nested(self._state, path)
            if old_value is None:
                return None

            self._delta_counter += 1
            delta_id = f"delta_{self.evaluator_id}_{self._delta_counter:010d}"

            delta = StateDelta(
                delta_id=delta_id,
                delta_type=DeltaType.DELETE,
                path=path,
                old_value=old_value,
                new_value=None,
            )

            self._delete_nested(self._state, path)
            self._deltas.append(delta)

            # Invalidate affected cache entries
            self._invalidate_cache(path)

            return delta

    def get_value(self, path: str) -> Any:
        """Get a value from the state.

        Args:
            path: Path to state element

        Returns:
            Value at path, or None if not found
        """
        return self._get_nested(self._state, path)

    def evaluate(
        self,
        evaluator: Callable[[dict[str, Any], list[StateDelta]], Any],
        force_full: bool = False,
    ) -> tuple[Any, bool]:
        """Evaluate state incrementally.

        Args:
            evaluator: Function that evaluates state
            force_full: Force full evaluation (ignore cache)

        Returns:
            Tuple of (result, was_incremental)
        """
        self._total_evaluations += 1
        start_time = time.perf_counter()

        with self._lock:
            state_hash = self._compute_state_hash()

            # Check cache
            if not force_full and state_hash in self._eval_cache:
                self._skipped_evaluations += 1
                execution_time = (time.perf_counter() - start_time) * 1000
                self._total_time_saved_ms += execution_time * 10  # Estimated savings
                return self._eval_cache[state_hash], True

            # Get deltas since last checkpoint
            current_deltas = self._deltas.copy()

            # Evaluate
            result = evaluator(self._state, current_deltas)

            # Cache result
            self._eval_cache[state_hash] = result

            # Log evaluation
            self.merkle_chain.add_event(
                "incremental_evaluation",
                {
                    "evaluator_id": self.evaluator_id,
                    "delta_count": len(current_deltas),
                    "state_hash": state_hash,
                },
            )

            return result, False

    def evaluate_affected_only(
        self,
        evaluator: Callable[[dict[str, Any], str], Any],
        affected_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate only affected state paths.

        Args:
            evaluator: Function that evaluates a single path
            affected_paths: Paths to evaluate (None for all changed)

        Returns:
            Dictionary of path -> evaluation result
        """
        self._total_evaluations += 1

        with self._lock:
            # Determine affected paths
            if affected_paths is None:
                affected_paths = list({d.path for d in self._deltas})

            results: dict[str, Any] = {}
            for path in affected_paths:
                value = self._get_nested(self._state, path)
                if value is not None:
                    results[path] = evaluator(self._state, path)

            return results

    def _create_checkpoint(self) -> StateCheckpoint:
        """Create a new state checkpoint."""
        self._checkpoint_counter += 1
        checkpoint_id = f"checkpoint_{self.evaluator_id}_{self._checkpoint_counter:06d}"

        state_hash = self._compute_state_hash()
        checkpoint = StateCheckpoint(
            checkpoint_id=checkpoint_id,
            state_snapshot=copy.deepcopy(self._state),
            state_hash=state_hash,
            delta_count=len(self._deltas),
            parent_checkpoint_id=self._current_checkpoint_id,
        )

        self._checkpoints[checkpoint_id] = checkpoint
        self._current_checkpoint_id = checkpoint_id
        self._deltas.clear()

        # Log checkpoint
        self.merkle_chain.add_event(
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint_id,
                "state_hash": state_hash,
            },
        )

        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from a checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore

        Returns:
            True if restored successfully
        """
        with self._lock:
            if checkpoint_id not in self._checkpoints:
                return False

            checkpoint = self._checkpoints[checkpoint_id]
            self._state = copy.deepcopy(checkpoint.state_snapshot)
            self._current_checkpoint_id = checkpoint_id
            self._deltas.clear()
            self._eval_cache.clear()

            # Log restoration
            self.merkle_chain.add_event(
                "checkpoint_restored",
                {
                    "checkpoint_id": checkpoint_id,
                    "state_hash": checkpoint.state_hash,
                },
            )

            return True

    def _compute_state_hash(self) -> str:
        """Compute hash of current state."""
        return hashlib.sha3_256(
            json.dumps(self._state, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _get_nested(self, obj: dict[str, Any], path: str) -> Any:
        """Get nested value by path."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _set_nested(self, obj: dict[str, Any], path: str, value: Any) -> None:
        """Set nested value by path."""
        parts = path.split(".")
        current = obj
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _delete_nested(self, obj: dict[str, Any], path: str) -> bool:
        """Delete nested value by path."""
        parts = path.split(".")
        current = obj
        for part in parts[:-1]:
            if part not in current:
                return False
            current = current[part]
        if parts[-1] in current:
            del current[parts[-1]]
            return True
        return False

    def _invalidate_cache(self, path: str) -> None:
        """Invalidate cache entries affected by path change."""
        # Simple invalidation: clear all cache on any change
        # More sophisticated implementations could track dependencies
        self._eval_cache.clear()

    def get_deltas_since(self, checkpoint_id: str | None = None) -> list[StateDelta]:
        """Get deltas since a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID (None for current)

        Returns:
            List of deltas
        """
        if checkpoint_id is None:
            return self._deltas.copy()

        # Find deltas since checkpoint
        # For simplicity, return current deltas
        return self._deltas.copy()

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of current state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def get_evaluator_stats(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        skip_rate = (
            self._skipped_evaluations / self._total_evaluations
            if self._total_evaluations > 0
            else 0
        )

        return {
            "evaluator_id": self.evaluator_id,
            "checkpoint_interval": self.checkpoint_interval,
            "total_evaluations": self._total_evaluations,
            "skipped_evaluations": self._skipped_evaluations,
            "skip_rate": skip_rate,
            "total_time_saved_ms": self._total_time_saved_ms,
            "total_deltas": self._delta_counter,
            "pending_deltas": len(self._deltas),
            "total_checkpoints": len(self._checkpoints),
            "current_checkpoint": self._current_checkpoint_id,
            "cache_size": len(self._eval_cache),
            "state_hash": self._compute_state_hash(),
        }
