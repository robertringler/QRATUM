"""Tensor Network Precomputations for High-Performance Evaluation.

Implements tensor network precomputations and classical-quantum hybrid
evaluations running off the critical path.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain


class TensorComputationType(Enum):
    """Type of tensor computation."""

    CONTRACTION = "contraction"
    DECOMPOSITION = "decomposition"
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    HYBRID = "hybrid"


class ComputationStatus(Enum):
    """Status of tensor computation."""

    PENDING = "pending"
    COMPUTING = "computing"
    CACHED = "cached"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class TensorCache:
    """Cache entry for precomputed tensor results.

    Attributes:
        cache_id: Unique cache identifier
        computation_type: Type of computation
        input_hash: Hash of input parameters
        result: Cached result
        computation_time_ms: Time to compute
        expires_at: Cache expiration time
    """

    cache_id: str
    computation_type: TensorComputationType
    input_hash: str
    result: dict[str, Any]
    computation_time_ms: float = 0.0
    hit_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize cache entry."""
        return {
            "cache_id": self.cache_id,
            "computation_type": self.computation_type.value,
            "input_hash": self.input_hash,
            "computation_time_ms": self.computation_time_ms,
            "hit_count": self.hit_count,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass
class HybridComputation:
    """Hybrid classical-quantum computation task.

    Attributes:
        computation_id: Unique computation identifier
        classical_component: Classical computation description
        quantum_component: Quantum computation description
        status: Computation status
        classical_result: Classical computation result
        quantum_result: Quantum computation result
        combined_result: Combined final result
    """

    computation_id: str
    classical_component: dict[str, Any]
    quantum_component: dict[str, Any] | None = None
    status: ComputationStatus = ComputationStatus.PENDING
    classical_result: dict[str, Any] | None = None
    quantum_result: dict[str, Any] | None = None
    combined_result: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize hybrid computation."""
        return {
            "computation_id": self.computation_id,
            "status": self.status.value,
            "has_quantum": self.quantum_component is not None,
            "has_classical_result": self.classical_result is not None,
            "has_quantum_result": self.quantum_result is not None,
            "has_combined_result": self.combined_result is not None,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_time_ms": self.total_time_ms,
        }


class TensorNetworkPrecomputer:
    """Precomputer for tensor network operations.

    Provides:
    - Background precomputation of tensor operations
    - Caching of results for reuse
    - Off-critical-path execution
    - Hybrid classical-quantum support
    """

    def __init__(
        self,
        precomputer_id: str = "tensor_precomputer",
        cache_size: int = 1000,
        cache_ttl_seconds: int = 3600,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize tensor network precomputer.

        Args:
            precomputer_id: Unique precomputer identifier
            cache_size: Maximum cache entries
            cache_ttl_seconds: Cache time-to-live
            merkle_chain: Merkle chain for audit trail
        """
        self.precomputer_id = precomputer_id
        self.cache_size = cache_size
        self.cache_ttl_seconds = cache_ttl_seconds
        self.merkle_chain = merkle_chain or MerkleChain()

        # Cache storage
        self._cache: dict[str, TensorCache] = {}
        self._cache_counter = 0
        self._lock = threading.RLock()

        # Computation tracking
        self._pending_computations: dict[str, HybridComputation] = {}
        self._computation_counter = 0

        # Background workers
        self._background_tasks: list[threading.Thread] = []

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_computations = 0
        self._total_computation_time_ms = 0.0

        # Log initialization
        self.merkle_chain.add_event(
            "tensor_precomputer_initialized",
            {
                "precomputer_id": precomputer_id,
                "cache_size": cache_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def precompute(
        self,
        computation_type: TensorComputationType,
        parameters: dict[str, Any],
        compute_func: Callable[[dict[str, Any]], dict[str, Any]],
        background: bool = True,
    ) -> TensorCache | None:
        """Precompute a tensor operation.

        Args:
            computation_type: Type of computation
            parameters: Computation parameters
            compute_func: Function to perform computation
            background: Whether to compute in background

        Returns:
            TensorCache if computed immediately, None if background
        """
        input_hash = self._compute_input_hash(parameters)

        # Check cache
        cached = self._get_cached(input_hash)
        if cached:
            self._cache_hits += 1
            cached.hit_count += 1
            return cached

        self._cache_misses += 1

        if background:
            # Start background computation
            thread = threading.Thread(
                target=self._compute_and_cache,
                args=(computation_type, parameters, compute_func, input_hash),
                daemon=True,
            )
            self._background_tasks.append(thread)
            thread.start()
            return None

        # Compute synchronously
        return self._compute_and_cache(
            computation_type, parameters, compute_func, input_hash
        )

    def _compute_and_cache(
        self,
        computation_type: TensorComputationType,
        parameters: dict[str, Any],
        compute_func: Callable[[dict[str, Any]], dict[str, Any]],
        input_hash: str,
    ) -> TensorCache:
        """Compute and cache result."""
        start_time = time.perf_counter()

        try:
            result = compute_func(parameters)
            computation_time = (time.perf_counter() - start_time) * 1000

            with self._lock:
                self._cache_counter += 1
                cache_id = f"tcache_{self.precomputer_id}_{self._cache_counter:08d}"

                # Calculate expiration
                expires_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl_seconds)
                ).isoformat()

                cache_entry = TensorCache(
                    cache_id=cache_id,
                    computation_type=computation_type,
                    input_hash=input_hash,
                    result=result,
                    computation_time_ms=computation_time,
                    expires_at=expires_at,
                )

                # Evict if necessary
                if len(self._cache) >= self.cache_size:
                    self._evict_oldest()

                self._cache[input_hash] = cache_entry
                self._total_computations += 1
                self._total_computation_time_ms += computation_time

            # Log computation
            self.merkle_chain.add_event(
                "tensor_computation_cached",
                {
                    "cache_id": cache_id,
                    "computation_type": computation_type.value,
                    "computation_time_ms": computation_time,
                },
            )

            return cache_entry

        except Exception as e:
            # Log failure
            self.merkle_chain.add_event(
                "tensor_computation_failed",
                {
                    "input_hash": input_hash,
                    "error": str(e),
                },
            )
            raise

    def _get_cached(self, input_hash: str) -> TensorCache | None:
        """Get cached result if available and not expired."""
        if input_hash not in self._cache:
            return None

        entry = self._cache[input_hash]

        # Check expiration
        if entry.expires_at:
            expires = datetime.fromisoformat(entry.expires_at)
            if datetime.now(timezone.utc) > expires:
                del self._cache[input_hash]
                return None

        return entry

    def _compute_input_hash(self, parameters: dict[str, Any]) -> str:
        """Compute hash of input parameters."""
        return hashlib.sha3_256(
            json.dumps(parameters, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache:
            return

        oldest_hash = min(
            self._cache.keys(),
            key=lambda h: self._cache[h].created_at,
        )
        del self._cache[oldest_hash]

    def get_cached(self, input_hash: str) -> TensorCache | None:
        """Get cached result by input hash."""
        return self._get_cached(input_hash)

    def create_hybrid_computation(
        self,
        classical_component: dict[str, Any],
        quantum_component: dict[str, Any] | None = None,
    ) -> HybridComputation:
        """Create a hybrid computation task.

        Args:
            classical_component: Classical computation description
            quantum_component: Optional quantum component

        Returns:
            HybridComputation task
        """
        with self._lock:
            self._computation_counter += 1
            computation_id = f"hybrid_{self.precomputer_id}_{self._computation_counter:08d}"

            computation = HybridComputation(
                computation_id=computation_id,
                classical_component=classical_component,
                quantum_component=quantum_component,
            )

            self._pending_computations[computation_id] = computation
            return computation

    def execute_hybrid(
        self,
        computation: HybridComputation,
        classical_func: Callable[[dict[str, Any]], dict[str, Any]],
        quantum_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        combiner_func: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]] | None = None,
    ) -> HybridComputation:
        """Execute a hybrid computation.

        Args:
            computation: Computation to execute
            classical_func: Classical computation function
            quantum_func: Optional quantum computation function
            combiner_func: Function to combine results

        Returns:
            Completed HybridComputation
        """
        computation.status = ComputationStatus.COMPUTING
        start_time = time.perf_counter()

        try:
            # Execute classical component
            computation.classical_result = classical_func(computation.classical_component)

            # Execute quantum component if present
            if computation.quantum_component and quantum_func:
                computation.quantum_result = quantum_func(computation.quantum_component)

            # Combine results
            if combiner_func:
                computation.combined_result = combiner_func(
                    computation.classical_result,
                    computation.quantum_result,
                )
            else:
                computation.combined_result = {
                    "classical": computation.classical_result,
                    "quantum": computation.quantum_result,
                }

            computation.status = ComputationStatus.CACHED
            computation.completed_at = datetime.now(timezone.utc).isoformat()
            computation.total_time_ms = (time.perf_counter() - start_time) * 1000

            # Log completion
            self.merkle_chain.add_event(
                "hybrid_computation_completed",
                {
                    "computation_id": computation.computation_id,
                    "total_time_ms": computation.total_time_ms,
                },
            )

        except Exception as e:
            computation.status = ComputationStatus.FAILED
            computation.combined_result = {"error": str(e)}

        return computation

    # Standard tensor operations

    def contract_tensors(
        self,
        tensor_a: dict[str, Any],
        tensor_b: dict[str, Any],
        contraction_indices: list[tuple[int, int]],
    ) -> TensorCache | None:
        """Precompute tensor contraction.

        Args:
            tensor_a: First tensor description
            tensor_b: Second tensor description
            contraction_indices: Indices to contract

        Returns:
            Cached result if available
        """
        params = {
            "tensor_a": tensor_a,
            "tensor_b": tensor_b,
            "contraction_indices": contraction_indices,
        }

        def compute(p: dict[str, Any]) -> dict[str, Any]:
            # Simplified tensor contraction simulation
            return {
                "result_shape": [10, 10],  # Placeholder
                "contraction_cost": len(p["contraction_indices"]) * 100,
                "operation": "contraction",
            }

        return self.precompute(
            TensorComputationType.CONTRACTION,
            params,
            compute,
            background=True,
        )

    def decompose_tensor(
        self,
        tensor: dict[str, Any],
        decomposition_type: str = "svd",
        rank: int = 10,
    ) -> TensorCache | None:
        """Precompute tensor decomposition.

        Args:
            tensor: Tensor to decompose
            decomposition_type: Type of decomposition (svd, qr, etc.)
            rank: Target rank

        Returns:
            Cached result if available
        """
        params = {
            "tensor": tensor,
            "decomposition_type": decomposition_type,
            "rank": rank,
        }

        def compute(p: dict[str, Any]) -> dict[str, Any]:
            # Simplified decomposition simulation
            return {
                "factors": [{"shape": [p["rank"], p["rank"]]}],
                "singular_values": [1.0 / (i + 1) for i in range(p["rank"])],
                "truncation_error": 0.01,
                "operation": "decomposition",
            }

        return self.precompute(
            TensorComputationType.DECOMPOSITION,
            params,
            compute,
            background=True,
        )

    def clear_cache(self) -> int:
        """Clear all cached results.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_precomputer_stats(self) -> dict[str, Any]:
        """Get precomputer statistics."""
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0
        )

        return {
            "precomputer_id": self.precomputer_id,
            "cache_size": self.cache_size,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_entries": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "total_computations": self._total_computations,
            "total_computation_time_ms": self._total_computation_time_ms,
            "pending_computations": len(self._pending_computations),
            "background_tasks": len(self._background_tasks),
        }
