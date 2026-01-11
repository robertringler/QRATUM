"""Deterministic Stubs for Heavy Computations.

Provides deterministic stubs for heavy computations when full fidelity
isn't required, reducing evaluation overhead.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain


class StubFidelityLevel(Enum):
    """Fidelity level for computation stubs."""

    EXACT = "exact"  # Full computation (no stub)
    HIGH = "high"  # High fidelity approximation (~99% accuracy)
    MEDIUM = "medium"  # Medium fidelity (~95% accuracy)
    LOW = "low"  # Low fidelity (~90% accuracy)
    PLACEHOLDER = "placeholder"  # Placeholder value only


@dataclass
class ComputationStub:
    """Stub for a heavy computation.

    Attributes:
        stub_id: Unique stub identifier
        computation_name: Name of the computation
        fidelity_level: Fidelity level of the stub
        stub_function: Function that implements the stub
        expected_speedup: Expected speedup vs full computation
        accuracy_estimate: Estimated accuracy (0-1)
        use_count: Number of times stub has been used
    """

    stub_id: str
    computation_name: str
    fidelity_level: StubFidelityLevel
    stub_function: Callable[[dict[str, Any]], Any] | None = None
    expected_speedup: float = 10.0
    accuracy_estimate: float = 0.95
    use_count: int = 0
    total_time_saved_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize stub metadata."""
        return {
            "stub_id": self.stub_id,
            "computation_name": self.computation_name,
            "fidelity_level": self.fidelity_level.value,
            "expected_speedup": self.expected_speedup,
            "accuracy_estimate": self.accuracy_estimate,
            "use_count": self.use_count,
            "total_time_saved_ms": self.total_time_saved_ms,
            "created_at": self.created_at,
        }


@dataclass
class StubResult:
    """Result of a stubbed computation.

    Attributes:
        stub_id: Stub that was used
        result: Computation result
        is_stubbed: Whether result is from a stub
        fidelity_level: Fidelity level used
        execution_time_ms: Actual execution time
        estimated_full_time_ms: Estimated time for full computation
        time_saved_ms: Time saved by using stub
    """

    stub_id: str
    result: Any
    is_stubbed: bool
    fidelity_level: StubFidelityLevel
    execution_time_ms: float
    estimated_full_time_ms: float = 0.0
    time_saved_ms: float = 0.0
    accuracy_confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize stub result."""
        return {
            "stub_id": self.stub_id,
            "is_stubbed": self.is_stubbed,
            "fidelity_level": self.fidelity_level.value,
            "execution_time_ms": self.execution_time_ms,
            "estimated_full_time_ms": self.estimated_full_time_ms,
            "time_saved_ms": self.time_saved_ms,
            "accuracy_confidence": self.accuracy_confidence,
            "timestamp": self.timestamp,
        }


class DeterministicStubRegistry:
    """Registry for deterministic computation stubs.

    Provides:
    - Registration of computation stubs
    - Automatic stub selection based on fidelity requirements
    - Deterministic execution with cached results
    - Performance tracking
    """

    def __init__(
        self,
        registry_id: str = "default",
        default_fidelity: StubFidelityLevel = StubFidelityLevel.MEDIUM,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize stub registry.

        Args:
            registry_id: Unique registry identifier
            default_fidelity: Default fidelity level
            merkle_chain: Merkle chain for audit trail
        """
        self.registry_id = registry_id
        self.default_fidelity = default_fidelity
        self.merkle_chain = merkle_chain or MerkleChain()

        # Stub storage
        self.stubs: dict[str, dict[StubFidelityLevel, ComputationStub]] = {}
        self._stub_counter = 0

        # Result cache for determinism
        self._cache: dict[str, StubResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Statistics
        self._total_invocations = 0
        self._total_time_saved_ms = 0.0

        # Register default stubs
        self._register_default_stubs()

        # Log initialization
        self.merkle_chain.add_event(
            "stub_registry_initialized",
            {
                "registry_id": registry_id,
                "default_fidelity": default_fidelity.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _register_default_stubs(self) -> None:
        """Register default computation stubs."""
        # Matrix multiplication stub
        self.register_stub(
            "matrix_multiply",
            StubFidelityLevel.HIGH,
            lambda ctx: self._stub_matrix_multiply(ctx, StubFidelityLevel.HIGH),
            expected_speedup=5.0,
            accuracy_estimate=0.99,
        )
        self.register_stub(
            "matrix_multiply",
            StubFidelityLevel.MEDIUM,
            lambda ctx: self._stub_matrix_multiply(ctx, StubFidelityLevel.MEDIUM),
            expected_speedup=20.0,
            accuracy_estimate=0.95,
        )
        self.register_stub(
            "matrix_multiply",
            StubFidelityLevel.LOW,
            lambda ctx: self._stub_matrix_multiply(ctx, StubFidelityLevel.LOW),
            expected_speedup=100.0,
            accuracy_estimate=0.90,
        )

        # Neural network inference stub
        self.register_stub(
            "neural_inference",
            StubFidelityLevel.HIGH,
            lambda ctx: self._stub_neural_inference(ctx, StubFidelityLevel.HIGH),
            expected_speedup=3.0,
            accuracy_estimate=0.98,
        )
        self.register_stub(
            "neural_inference",
            StubFidelityLevel.MEDIUM,
            lambda ctx: self._stub_neural_inference(ctx, StubFidelityLevel.MEDIUM),
            expected_speedup=10.0,
            accuracy_estimate=0.93,
        )

        # Simulation step stub
        self.register_stub(
            "simulation_step",
            StubFidelityLevel.HIGH,
            lambda ctx: self._stub_simulation_step(ctx, StubFidelityLevel.HIGH),
            expected_speedup=2.0,
            accuracy_estimate=0.99,
        )
        self.register_stub(
            "simulation_step",
            StubFidelityLevel.MEDIUM,
            lambda ctx: self._stub_simulation_step(ctx, StubFidelityLevel.MEDIUM),
            expected_speedup=5.0,
            accuracy_estimate=0.95,
        )

        # Quantum circuit simulation stub
        self.register_stub(
            "quantum_circuit",
            StubFidelityLevel.HIGH,
            lambda ctx: self._stub_quantum_circuit(ctx, StubFidelityLevel.HIGH),
            expected_speedup=10.0,
            accuracy_estimate=0.97,
        )
        self.register_stub(
            "quantum_circuit",
            StubFidelityLevel.MEDIUM,
            lambda ctx: self._stub_quantum_circuit(ctx, StubFidelityLevel.MEDIUM),
            expected_speedup=50.0,
            accuracy_estimate=0.90,
        )

    def register_stub(
        self,
        computation_name: str,
        fidelity_level: StubFidelityLevel,
        stub_function: Callable[[dict[str, Any]], Any],
        expected_speedup: float = 10.0,
        accuracy_estimate: float = 0.95,
    ) -> ComputationStub:
        """Register a computation stub.

        Args:
            computation_name: Name of the computation
            fidelity_level: Fidelity level of this stub
            stub_function: Function implementing the stub
            expected_speedup: Expected speedup factor
            accuracy_estimate: Estimated accuracy

        Returns:
            Created ComputationStub
        """
        self._stub_counter += 1
        stub_id = f"stub_{computation_name}_{fidelity_level.value}_{self._stub_counter:04d}"

        stub = ComputationStub(
            stub_id=stub_id,
            computation_name=computation_name,
            fidelity_level=fidelity_level,
            stub_function=stub_function,
            expected_speedup=expected_speedup,
            accuracy_estimate=accuracy_estimate,
        )

        if computation_name not in self.stubs:
            self.stubs[computation_name] = {}

        self.stubs[computation_name][fidelity_level] = stub
        return stub

    def execute(
        self,
        computation_name: str,
        context: dict[str, Any],
        fidelity_level: StubFidelityLevel | None = None,
        full_computation: Callable[[dict[str, Any]], Any] | None = None,
        use_cache: bool = True,
    ) -> StubResult:
        """Execute a computation with optional stubbing.

        Args:
            computation_name: Name of the computation
            context: Computation context
            fidelity_level: Desired fidelity level (None for exact)
            full_computation: Full computation function (required for EXACT)
            use_cache: Whether to use result cache

        Returns:
            StubResult with computation result
        """
        self._total_invocations += 1
        effective_fidelity = fidelity_level or self.default_fidelity

        # Check cache
        if use_cache:
            cache_key = self._compute_cache_key(computation_name, context, effective_fidelity)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            self._cache_misses += 1

        start_time = time.perf_counter()

        # Execute based on fidelity level
        if effective_fidelity == StubFidelityLevel.EXACT:
            if full_computation is None:
                raise ValueError("Full computation function required for EXACT fidelity")
            result = full_computation(context)
            is_stubbed = False
            accuracy_confidence = 1.0
            estimated_full_time = (time.perf_counter() - start_time) * 1000
        else:
            # Get appropriate stub
            stub = self._get_stub(computation_name, effective_fidelity)
            if stub is None or stub.stub_function is None:
                raise ValueError(f"No stub available for {computation_name} at {effective_fidelity.value}")

            result = stub.stub_function(context)
            is_stubbed = True
            accuracy_confidence = stub.accuracy_estimate
            estimated_full_time = (
                (time.perf_counter() - start_time) * 1000 * stub.expected_speedup
            )

            # Update stub statistics
            stub.use_count += 1

        execution_time_ms = (time.perf_counter() - start_time) * 1000
        time_saved_ms = estimated_full_time - execution_time_ms if is_stubbed else 0

        if is_stubbed:
            self._total_time_saved_ms += time_saved_ms
            stub = self._get_stub(computation_name, effective_fidelity)
            if stub:
                stub.total_time_saved_ms += time_saved_ms

        stub_id = (
            self._get_stub(computation_name, effective_fidelity).stub_id
            if self._get_stub(computation_name, effective_fidelity)
            else f"direct_{computation_name}"
        )

        stub_result = StubResult(
            stub_id=stub_id,
            result=result,
            is_stubbed=is_stubbed,
            fidelity_level=effective_fidelity,
            execution_time_ms=execution_time_ms,
            estimated_full_time_ms=estimated_full_time,
            time_saved_ms=time_saved_ms,
            accuracy_confidence=accuracy_confidence,
        )

        # Cache result
        if use_cache:
            cache_key = self._compute_cache_key(computation_name, context, effective_fidelity)
            self._cache[cache_key] = stub_result

        # Log execution
        self.merkle_chain.add_event(
            "stub_executed",
            {
                "computation_name": computation_name,
                "fidelity_level": effective_fidelity.value,
                "is_stubbed": is_stubbed,
                "execution_time_ms": execution_time_ms,
            },
        )

        return stub_result

    def _get_stub(
        self,
        computation_name: str,
        fidelity_level: StubFidelityLevel,
    ) -> ComputationStub | None:
        """Get stub for computation at given fidelity level."""
        if computation_name not in self.stubs:
            return None
        return self.stubs[computation_name].get(fidelity_level)

    def _compute_cache_key(
        self,
        computation_name: str,
        context: dict[str, Any],
        fidelity_level: StubFidelityLevel,
    ) -> str:
        """Compute deterministic cache key."""
        key_data = {
            "computation_name": computation_name,
            "context": context,
            "fidelity_level": fidelity_level.value,
        }
        return hashlib.sha3_256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

    # Default stub implementations

    def _stub_matrix_multiply(
        self,
        context: dict[str, Any],
        fidelity: StubFidelityLevel,
    ) -> dict[str, Any]:
        """Stub for matrix multiplication."""
        # Return deterministic approximation based on input shape
        shape = context.get("shape", (10, 10))
        return {
            "result_shape": shape,
            "stub_fidelity": fidelity.value,
            "approximation": True,
        }

    def _stub_neural_inference(
        self,
        context: dict[str, Any],
        fidelity: StubFidelityLevel,
    ) -> dict[str, Any]:
        """Stub for neural network inference."""
        # Return cached/approximated predictions
        input_size = context.get("input_size", 100)
        return {
            "predictions": [0.5] * min(input_size, 10),
            "confidence": 0.95 if fidelity == StubFidelityLevel.HIGH else 0.85,
            "stub_fidelity": fidelity.value,
        }

    def _stub_simulation_step(
        self,
        context: dict[str, Any],
        fidelity: StubFidelityLevel,
    ) -> dict[str, Any]:
        """Stub for simulation step."""
        step = context.get("step", 0)
        return {
            "step": step,
            "state_delta": 0.001 * (1 + step),
            "converged": step > 100,
            "stub_fidelity": fidelity.value,
        }

    def _stub_quantum_circuit(
        self,
        context: dict[str, Any],
        fidelity: StubFidelityLevel,
    ) -> dict[str, Any]:
        """Stub for quantum circuit simulation."""
        qubits = context.get("qubits", 4)
        return {
            "state_vector_size": 2**qubits,
            "measurement_probability": 0.5,
            "fidelity": 0.99 if fidelity == StubFidelityLevel.HIGH else 0.95,
            "stub_fidelity": fidelity.value,
        }

    def clear_cache(self) -> int:
        """Clear the result cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        stub_stats = {}
        for name, fidelity_stubs in self.stubs.items():
            stub_stats[name] = {
                level.value: stub.to_dict()
                for level, stub in fidelity_stubs.items()
            }

        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0
        )

        return {
            "registry_id": self.registry_id,
            "default_fidelity": self.default_fidelity.value,
            "total_computations": len(self.stubs),
            "total_invocations": self._total_invocations,
            "total_time_saved_ms": self._total_time_saved_ms,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "stub_stats": stub_stats,
        }
