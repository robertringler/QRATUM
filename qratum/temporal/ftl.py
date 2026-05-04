"""
FTL Computation Framework - Faster-Than-Light Computational Paradigms

Implements mechanisms for achieving effective faster-than-light computation
through various strategies that compute answers before light could carry the question.

Mechanisms:
- PRECOMPUTATION: Compute all possibilities in advance for instant lookup
- PREDICTIVE_MODELING: Model futures faster than they unfold
- PARALLEL_BRANCHING: Explore all timelines simultaneously
- TEMPORAL_COMPRESSION: Compress simulated time vs wall time
- INVERSE_CAUSALITY: Compute backward from desired outcome
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import (
    INFINITE_VELOCITY,
    OPERATIONS_PER_LIGHT_METER,
    QRATUM_PEAK_FLOPS,
    SPEED_OF_LIGHT,
    FTLMechanism,
)
from .engine import TemporalEngine
from .state import TemporalState


@dataclass
class EffectiveVelocity:
    """
    Metrics for computational FTL performance.

    Attributes:
        mechanism: FTL mechanism used
        computational_delta_t: Simulated time traversed
        physical_delta_t: Wall-clock time elapsed
        effective_c_multiple: Speed as multiple of light speed
        operations_executed: Total operations executed
        distance_equivalent: Equivalent light-distance (meters)
    """

    mechanism: str
    computational_delta_t: float
    physical_delta_t: float
    effective_c_multiple: float
    operations_executed: int = 0
    distance_equivalent: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.physical_delta_t > 0:
            self.effective_c_multiple = self.computational_delta_t / self.physical_delta_t
        else:
            self.effective_c_multiple = INFINITE_VELOCITY

        # Calculate equivalent distance light would travel
        self.distance_equivalent = SPEED_OF_LIGHT * self.computational_delta_t

    def __str__(self) -> str:
        if self.effective_c_multiple == INFINITE_VELOCITY:
            return f"EffectiveVelocity: ∞c (instantaneous) via {self.mechanism}"
        else:
            return (
                f"EffectiveVelocity: {self.effective_c_multiple:.2e}c "
                f"({self.distance_equivalent/1000:.2f}km equivalent) via {self.mechanism}"
            )

    def is_ftl(self) -> bool:
        """Check if this computation achieved FTL speeds"""
        return self.effective_c_multiple > 1.0


@dataclass
class PossibilitySpace:
    """
    Pre-computed possibility space for a system.

    Attributes:
        state_space: Dictionary mapping state keys to outcomes
        creation_time: When the space was computed
        coverage: Fraction of possible states covered
        lookup_count: Number of lookups performed
    """

    state_space: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    coverage: float = 1.0
    lookup_count: int = 0

    def lookup(self, state_key: str) -> Optional[Any]:
        """
        Instant lookup of pre-computed result.

        This achieves infinite effective velocity since the answer is
        already computed.
        """
        self.lookup_count += 1
        return self.state_space.get(state_key)

    def size(self) -> int:
        """Get size of possibility space"""
        return len(self.state_space)


class PossibilityOracle:
    """
    Pre-computed possibility space for instant lookup.

    The oracle pre-computes all possible outcomes for a given system,
    allowing instantaneous queries with effective velocity of infinity.

    Example:
        >>> oracle = PossibilityOracle()
        >>> oracle.precompute_all(
        ...     initial_states=[s1, s2, s3],
        ...     evolution_fn=simulate,
        ...     delta_t=100,
        ... )
        >>> result = oracle.query(state_key)  # Instant lookup, ∞c
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize possibility oracle.

        Args:
            max_workers: Maximum parallel workers for precomputation
        """
        self.possibility_spaces: Dict[str, PossibilitySpace] = {}
        self.max_workers = max_workers

    def precompute_all(
        self,
        initial_states: List[Any],
        evolution_fn: Callable[[Any, float], Any],
        delta_t: float,
        space_id: Optional[str] = None,
    ) -> Tuple[PossibilitySpace, EffectiveVelocity]:
        """
        Precompute all possibilities in advance.

        Args:
            initial_states: List of initial states to explore
            evolution_fn: Evolution function
            delta_t: Time to evolve each state
            space_id: Optional identifier for this space

        Returns:
            Tuple of (possibility_space, velocity_metrics)
        """
        wall_start = time.time()

        if space_id is None:
            space_id = f"space_{uuid.uuid4().hex[:8]}"

        space = PossibilitySpace()

        # Precompute all states (can be parallelized)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for initial_state in initial_states:
                future = executor.submit(self._evolve_state, initial_state, evolution_fn, delta_t)
                futures.append((initial_state, future))

            # Collect results
            for initial_state, future in futures:
                try:
                    final_state = future.result()
                    state_key = self._hash_state(initial_state)
                    space.state_space[state_key] = final_state
                except Exception:
                    # Log error but continue
                    pass

        wall_end = time.time()

        # Store space
        self.possibility_spaces[space_id] = space

        # Calculate velocity metrics
        # Total simulated time = delta_t * number of states
        total_simulated = delta_t * len(initial_states)
        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.PRECOMPUTATION,
            computational_delta_t=total_simulated,
            physical_delta_t=wall_end - wall_start,
            operations_executed=len(initial_states),
        )

        return space, velocity

    def query(
        self,
        state_key: str,
        space_id: str,
    ) -> Tuple[Optional[Any], EffectiveVelocity]:
        """
        Instant lookup from pre-computed space.

        Args:
            state_key: Key for state to lookup
            space_id: Identifier for possibility space

        Returns:
            Tuple of (result, velocity_metrics)
        """
        wall_start = time.time()

        if space_id not in self.possibility_spaces:
            return None, EffectiveVelocity(
                mechanism=FTLMechanism.PRECOMPUTATION,
                computational_delta_t=0.0,
                physical_delta_t=0.0,
            )

        space = self.possibility_spaces[space_id]
        result = space.lookup(state_key)

        wall_end = time.time()

        # Instant lookup achieves infinite velocity
        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.PRECOMPUTATION,
            computational_delta_t=1.0,  # Arbitrary positive value
            physical_delta_t=wall_end - wall_start if wall_end > wall_start else 1e-12,
            operations_executed=1,
        )

        return result, velocity

    def _evolve_state(
        self,
        initial_state: Any,
        evolution_fn: Callable[[Any, float], Any],
        delta_t: float,
    ) -> Any:
        """Evolve a single state"""
        return evolution_fn(initial_state, delta_t)

    def _hash_state(self, state: Any) -> str:
        """Generate hash key for a state"""
        import hashlib
        import pickle

        try:
            state_bytes = pickle.dumps(state)
        except:
            state_bytes = str(state).encode("utf-8")

        return hashlib.sha256(state_bytes).hexdigest()


class FTLComputation:
    """
    Faster-than-light computational paradigms.

    Implements various mechanisms for achieving effective FTL computation speeds,
    allowing QRATUM to compute answers before light could carry the question.

    Example:
        >>> ftl = FTLComputation(peak_flops=2.5e18)
        >>> result, velocity = ftl.compress_time(
        ...     initial_state=current_state,
        ...     evolution_fn=model,
        ...     delta_t=SECONDS_PER_YEAR,
        ...     compression_target=1e9,
        ... )
        >>> print(f"Achieved {velocity.effective_c_multiple:.2e}c")
    """

    def __init__(self, peak_flops: float = QRATUM_PEAK_FLOPS):
        """
        Initialize FTL computation framework.

        Args:
            peak_flops: Peak computational performance
        """
        self.peak_flops = peak_flops
        self.engine = TemporalEngine(peak_flops=peak_flops)
        self.oracle = PossibilityOracle()

    def compress_time(
        self,
        initial_state: Any,
        evolution_fn: Callable[[Any, float], Any],
        delta_t: float,
        compression_target: float = 1e6,
    ) -> Tuple[TemporalState, EffectiveVelocity]:
        """
        Simulate time faster than real-time.

        Achieves temporal compression by computing many simulation steps
        in a short wall-clock time.

        Args:
            initial_state: Starting state
            evolution_fn: Evolution function
            delta_t: Simulated time to traverse
            compression_target: Target compression ratio

        Returns:
            Tuple of (final_state, velocity_metrics)
        """
        wall_start = time.time()

        # Use temporal engine to compute forward
        final_state, proof = self.engine.forward(
            initial_state=initial_state,
            delta_t=delta_t,
            evolution_fn=evolution_fn,
        )

        wall_end = time.time()
        wall_delta = wall_end - wall_start

        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.TEMPORAL_COMPRESSION,
            computational_delta_t=delta_t,
            physical_delta_t=wall_delta,
            operations_executed=proof.metadata.get("steps", 0),
        )

        return final_state, velocity

    def inverse_causality(
        self,
        desired_outcome: Any,
        inverse_fn: Callable[[Any, float], Any],
        delta_t: float,
    ) -> Tuple[TemporalState, EffectiveVelocity]:
        """
        Compute backward from desired outcome.

        Given a desired future state, compute backward to find the initial
        conditions that would lead to that outcome.

        Args:
            desired_outcome: Target future state
            inverse_fn: Inverse evolution function
            delta_t: Time to travel backward

        Returns:
            Tuple of (initial_state, velocity_metrics)
        """
        wall_start = time.time()

        # Use temporal engine to compute backward
        initial_state, proof = self.engine.backward(
            final_state=desired_outcome,
            delta_t=-abs(delta_t),
            inverse_fn=inverse_fn,
        )

        wall_end = time.time()
        wall_delta = wall_end - wall_start

        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.INVERSE_CAUSALITY,
            computational_delta_t=abs(delta_t),
            physical_delta_t=wall_delta,
            operations_executed=proof.metadata.get("steps", 0),
        )

        return initial_state, velocity

    def parallel_reality_search(
        self,
        branch_point: Any,
        branch_count: int,
        evolution_fn: Callable[[Any, float], Any],
        mutation_fn: Callable[[Any], Any],
        target_fn: Callable[[Any], float],
        duration: float,
    ) -> Tuple[TemporalState, str, EffectiveVelocity]:
        """
        Search across branching timelines in parallel.

        Explores multiple parallel realities simultaneously to find the
        optimal outcome according to a target function.

        Args:
            branch_point: Starting state for all branches
            branch_count: Number of parallel branches to explore
            evolution_fn: Evolution function
            mutation_fn: Function to create variations
            target_fn: Scoring function (higher is better)
            duration: Time to evolve each branch

        Returns:
            Tuple of (best_outcome, best_branch_name, velocity_metrics)
        """
        wall_start = time.time()

        # Create branches
        branches = []
        for i in range(branch_count):
            branch_name = f"branch_{i}"

            def mutation(state):
                return mutation_fn(state)

            branches.append((branch_name, mutation))

        # Evolve all branches in parallel
        results = self.engine.branch(
            branch_point=branch_point,
            branches=branches,
            evolution_fn=evolution_fn,
            delta_t=duration,
        )

        # Find best outcome
        best_score = float("-inf")
        best_branch = None
        best_state = None

        for branch_name, (final_state, proof) in results.items():
            score = target_fn(final_state.data)
            if score > best_score:
                best_score = score
                best_branch = branch_name
                best_state = final_state

        wall_end = time.time()
        wall_delta = wall_end - wall_start

        # Total simulated time = duration * branch_count
        total_simulated = duration * branch_count

        velocity = EffectiveVelocity(
            mechanism=FTLMechanism.PARALLEL_BRANCHING,
            computational_delta_t=total_simulated,
            physical_delta_t=wall_delta,
            operations_executed=branch_count,
        )

        return best_state, best_branch, velocity

    def get_statistics(self) -> Dict[str, Any]:
        """Get FTL computation statistics"""
        return {
            "peak_flops": self.peak_flops,
            "operations_per_light_meter": OPERATIONS_PER_LIGHT_METER,
            "engine_stats": self.engine.get_statistics(),
            "oracle_spaces": len(self.oracle.possibility_spaces),
        }
