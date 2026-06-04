"""
Temporal Engine - Core Temporal Computation Engine for QRATUM

Enables computational time travel through QRATUM's speed-of-light-breaking paradigm.
Transforms time from a constraint into a navigable computational dimension.

Key Capabilities:
1. FORWARD PROJECTION: Simulate future states at ExaFLOPS speeds
2. BACKWARD RECONSTRUCTION: Recover past states via inverse computation
3. PARALLEL EXPLORATION: Branch timelines for possibility analysis
4. TEMPORAL VERIFICATION: Cryptographically prove temporal computations
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import (
    QRATUM_PEAK_FLOPS,
    TARGET_COMPRESSION_RATIO,
    ParadoxStrategy,
)
from .state import StateChain, TemporalCoordinate, TemporalState
from .verification import TemporalProof, TemporalVerifier


@dataclass
class TemporalEngineConfig:
    """
    Configuration for the temporal engine.

    Attributes:
        peak_flops: Peak computational performance in FLOPS
        deterministic: Enable deterministic execution mode
        merkle_verified: Enable Merkle verification of state transitions
        paradox_resolution: Strategy for resolving temporal paradoxes
        temporal_resolution: Minimum time step (seconds)
        enable_pqc: Enable post-quantum cryptographic signatures
        max_timeline_depth: Maximum depth of timeline exploration
        compression_target: Target temporal compression ratio
    """

    peak_flops: float = QRATUM_PEAK_FLOPS
    deterministic: bool = True
    merkle_verified: bool = True
    paradox_resolution: str = ParadoxStrategy.NOVIKOV
    temporal_resolution: float = 1e-9  # 1 nanosecond default
    enable_pqc: bool = False
    max_timeline_depth: int = 1_000_000
    compression_target: float = TARGET_COMPRESSION_RATIO


class TemporalEngine:
    """
    Core temporal computation engine for QRATUM.

    The TemporalEngine enables computational time travel by computing temporal
    states faster than light could traverse the physical distance between events.
    At 2.5 ExaFLOPS, QRATUM completes 8.3 billion calculations per meter light travels.

    Example:
        >>> engine = TemporalEngine(
        ...     peak_flops=2.5e18,
        ...     deterministic=True,
        ...     merkle_verified=True,
        ... )
        >>> future = engine.forward(
        ...     initial_state=current_state,
        ...     delta_t=100 * SECONDS_PER_YEAR,
        ...     evolution_fn=climate_model,
        ... )
    """

    def __init__(
        self,
        peak_flops: float = QRATUM_PEAK_FLOPS,
        deterministic: bool = True,
        merkle_verified: bool = True,
        paradox_resolution: str = ParadoxStrategy.NOVIKOV,
        temporal_resolution: float = 1e-9,
        enable_pqc: bool = False,
    ):
        """
        Initialize the temporal engine.

        Args:
            peak_flops: Peak computational performance in FLOPS
            deterministic: Enable deterministic execution mode
            merkle_verified: Enable Merkle verification
            paradox_resolution: Strategy for paradox resolution
            temporal_resolution: Minimum time step (seconds)
            enable_pqc: Enable post-quantum cryptography
        """
        self.config = TemporalEngineConfig(
            peak_flops=peak_flops,
            deterministic=deterministic,
            merkle_verified=merkle_verified,
            paradox_resolution=paradox_resolution,
            temporal_resolution=temporal_resolution,
            enable_pqc=enable_pqc,
        )

        self.verifier = TemporalVerifier(enable_pqc=enable_pqc)
        self.timelines: Dict[str, StateChain] = {}
        self.operation_count = 0

    def forward(
        self,
        initial_state: Any,
        delta_t: float,
        evolution_fn: Callable[[Any, float], Any],
        timeline_id: Optional[str] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[TemporalState, TemporalProof]:
        """
        Travel forward in time via deterministic simulation.

        Computes future states by applying the evolution function repeatedly.
        The computation is faster than real-time, achieving effective FTL speeds.

        Args:
            initial_state: Starting state of the system
            delta_t: Time to travel forward (in seconds)
            evolution_fn: Function that evolves state: (state, dt) -> new_state
            timeline_id: Optional timeline identifier
            num_steps: Number of intermediate steps (auto-calculated if None)

        Returns:
            Tuple of (final_state, proof)

        Example:
            >>> future_climate = engine.forward(
            ...     initial_state=current_climate,
            ...     delta_t=100 * SECONDS_PER_YEAR,
            ...     evolution_fn=climate_model,
            ... )
        """
        wall_start = time.time()

        if timeline_id is None:
            timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"

        # Calculate number of steps based on temporal resolution
        if num_steps is None:
            num_steps = max(1, int(delta_t / self.config.temporal_resolution))
            # Bound auto-calculated steps to avoid unbounded computation and
            # memory growth when delta_t >> temporal_resolution (e.g. a 100s
            # delta at 1ns resolution would otherwise be 1e11 iterations).
            num_steps = min(num_steps, self.config.max_timeline_depth)

        step_size = delta_t / num_steps

        # Create initial temporal state
        initial_coord = TemporalCoordinate(
            state_hash="",  # Will be computed
            timeline_id=timeline_id,
            computational_t=0.0,
            physical_t=wall_start,
            depth=0,
        )

        initial_temporal_state = TemporalState(
            coordinate=initial_coord,
            data=initial_state,
            parent_hash=None,
        )
        initial_temporal_state.coordinate.state_hash = initial_temporal_state.compute_hash()

        # Create state chain
        chain = StateChain(timeline_id=timeline_id)
        chain.add_state(initial_temporal_state)

        # Evolve forward through time
        current_state = initial_state
        current_t = 0.0

        for step in range(num_steps):
            # Apply evolution function
            current_state = evolution_fn(current_state, step_size)
            current_t += step_size

            # Create temporal state for this step
            coord = TemporalCoordinate(
                state_hash="",  # Will be computed
                timeline_id=timeline_id,
                computational_t=current_t,
                physical_t=time.time(),
                depth=step + 1,
            )

            temporal_state = TemporalState(
                coordinate=coord,
                data=current_state,
                parent_hash=chain.states[-1].coordinate.state_hash,
            )
            temporal_state.coordinate.state_hash = temporal_state.compute_hash()

            chain.add_state(temporal_state)

        wall_end = time.time()
        wall_delta = wall_end - wall_start

        # Store timeline
        self.timelines[timeline_id] = chain

        # Generate proof
        final_state = chain.states[-1]
        proof = self.verifier.generate_proof(
            initial_state=initial_temporal_state,
            final_state=final_state,
            chain=chain,
            operation="forward",
        )

        # Add performance metrics
        proof.metadata["wall_time"] = wall_delta
        proof.metadata["steps"] = num_steps
        proof.metadata["effective_velocity_c"] = proof.effective_velocity_c_multiple()

        self.operation_count += 1

        return final_state, proof

    def backward(
        self,
        final_state: Any,
        delta_t: float,
        inverse_fn: Callable[[Any, float], Any],
        timeline_id: Optional[str] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[TemporalState, TemporalProof]:
        """
        Travel backward in time via inverse computation.

        Reconstructs past states by applying the inverse evolution function.
        Requires deterministic, reversible dynamics.

        Args:
            final_state: Ending state of the system
            delta_t: Time to travel backward (negative value)
            inverse_fn: Inverse evolution function: (state, dt) -> prior_state
            timeline_id: Optional timeline identifier
            num_steps: Number of intermediate steps

        Returns:
            Tuple of (initial_state, proof)

        Example:
            >>> past_state = engine.backward(
            ...     final_state=current_evidence,
            ...     delta_t=-1000 * SECONDS_PER_YEAR,
            ...     inverse_fn=geological_reversal,
            ... )
        """
        # Ensure delta_t is negative for backward travel
        if delta_t > 0:
            delta_t = -delta_t

        wall_start = time.time()

        if timeline_id is None:
            timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"

        # Calculate number of steps
        if num_steps is None:
            num_steps = max(1, int(abs(delta_t) / self.config.temporal_resolution))
            # Bound auto-calculated steps to avoid unbounded computation and
            # memory growth when |delta_t| >> temporal_resolution.
            num_steps = min(num_steps, self.config.max_timeline_depth)

        step_size = abs(delta_t) / num_steps

        # Create final temporal state (this is our starting point for backward)
        final_coord = TemporalCoordinate(
            state_hash="",
            timeline_id=timeline_id,
            computational_t=0.0,  # We're going backward from here
            physical_t=wall_start,
            depth=0,
        )

        final_temporal_state = TemporalState(
            coordinate=final_coord,
            data=final_state,
            parent_hash=None,
        )
        final_temporal_state.coordinate.state_hash = final_temporal_state.compute_hash()

        # Create state chain
        chain = StateChain(timeline_id=timeline_id)
        chain.add_state(final_temporal_state)

        # Evolve backward through time
        current_state = final_state
        current_t = 0.0

        for step in range(num_steps):
            # Apply inverse function
            current_state = inverse_fn(current_state, step_size)
            current_t -= step_size  # Moving backward in time

            # Create temporal state
            coord = TemporalCoordinate(
                state_hash="",
                timeline_id=timeline_id,
                computational_t=current_t,
                physical_t=time.time(),
                depth=step + 1,
            )

            temporal_state = TemporalState(
                coordinate=coord,
                data=current_state,
                parent_hash=chain.states[-1].coordinate.state_hash,
            )
            temporal_state.coordinate.state_hash = temporal_state.compute_hash()

            chain.add_state(temporal_state)

        wall_end = time.time()
        wall_delta = wall_end - wall_start

        # Store timeline
        self.timelines[timeline_id] = chain

        # Generate proof
        initial_state = chain.states[-1]
        proof = self.verifier.generate_proof(
            initial_state=final_temporal_state,
            final_state=initial_state,
            chain=chain,
            operation="backward",
        )

        proof.metadata["wall_time"] = wall_delta
        proof.metadata["steps"] = num_steps
        proof.metadata["effective_velocity_c"] = proof.effective_velocity_c_multiple()

        self.operation_count += 1

        return initial_state, proof

    def branch(
        self,
        branch_point: Any,
        branches: List[Tuple[str, Callable[[Any], Any]]],
        evolution_fn: Callable[[Any, float], Any],
        delta_t: float,
        parent_timeline_id: Optional[str] = None,
    ) -> Dict[str, Tuple[TemporalState, TemporalProof]]:
        """
        Branch into parallel timelines from a decision point.

        Creates multiple alternative futures by applying different mutations
        at a branch point and evolving each independently.

        Args:
            branch_point: State to branch from
            branches: List of (name, mutation_fn) tuples
            evolution_fn: Evolution function for each branch
            delta_t: Time to evolve each branch
            parent_timeline_id: Optional parent timeline

        Returns:
            Dictionary mapping branch names to (final_state, proof)

        Example:
            >>> timelines = engine.branch(
            ...     branch_point=decision_moment,
            ...     branches=[
            ...         ("choice_a", apply_choice_a),
            ...         ("choice_b", apply_choice_b),
            ...     ],
            ...     evolution_fn=consequence_model,
            ...     delta_t=50 * SECONDS_PER_YEAR,
            ... )
        """
        results = {}

        for branch_name, mutation_fn in branches:
            # Apply mutation to create branch-specific initial state
            branch_initial = mutation_fn(branch_point)

            # Generate timeline ID
            if parent_timeline_id:
                timeline_id = f"{parent_timeline_id}_branch_{branch_name}"
            else:
                timeline_id = f"branch_{branch_name}_{uuid.uuid4().hex[:8]}"

            # Evolve this branch forward
            final_state, proof = self.forward(
                initial_state=branch_initial,
                delta_t=delta_t,
                evolution_fn=evolution_fn,
                timeline_id=timeline_id,
            )

            # Mark as branch operation
            proof.operation = "branch"
            proof.metadata["branch_name"] = branch_name
            proof.metadata["parent_timeline"] = parent_timeline_id

            results[branch_name] = (final_state, proof)

        self.operation_count += 1

        return results

    def converge(
        self,
        branches: Dict[str, TemporalState],
        convergence_fn: Callable[[List[Any]], Any],
        timeline_id: Optional[str] = None,
    ) -> Tuple[TemporalState, TemporalProof]:
        """
        Merge parallel timelines into a single outcome.

        Takes multiple branch endpoints and combines them using a convergence
        function to create a unified result.

        Args:
            branches: Dictionary mapping branch names to final states
            convergence_fn: Function to merge states: (List[state]) -> merged_state
            timeline_id: Optional timeline ID for converged timeline

        Returns:
            Tuple of (converged_state, proof)
        """
        wall_start = time.time()

        if timeline_id is None:
            timeline_id = f"converged_{uuid.uuid4().hex[:8]}"

        # Extract state data from all branches
        branch_states = [state.data for state in branches.values()]

        # Apply convergence function
        converged_data = convergence_fn(branch_states)

        # Create converged state
        coord = TemporalCoordinate(
            state_hash="",
            timeline_id=timeline_id,
            computational_t=time.time(),
            physical_t=time.time(),
            depth=0,
        )

        converged_state = TemporalState(
            coordinate=coord,
            data=converged_data,
            parent_hash=None,
            metadata={
                "branch_count": len(branches),
                "branch_names": list(branches.keys()),
            },
        )
        converged_state.coordinate.state_hash = converged_state.compute_hash()

        wall_end = time.time()

        # Create minimal proof (no full chain for convergence)
        proof = TemporalProof(
            initial_state_hash=";".join(s.coordinate.state_hash for s in branches.values()),
            final_state_hash=converged_state.coordinate.state_hash,
            merkle_root="",
            operation="converge",
            timeline_id=timeline_id,
            computational_delta_t=0.0,
            physical_delta_t=wall_end - wall_start,
            intermediate_hashes=[],
            metadata={
                "branch_count": len(branches),
                "branch_names": list(branches.keys()),
            },
        )

        self.operation_count += 1

        return converged_state, proof

    def verify_temporal_consistency(self, timeline_id: str) -> bool:
        """
        Verify cryptographic consistency of a timeline.

        Args:
            timeline_id: Timeline to verify

        Returns:
            True if timeline is consistent
        """
        if timeline_id not in self.timelines:
            return False

        chain = self.timelines[timeline_id]
        return self.verifier.verify_state_chain(chain)

    def get_timeline(self, timeline_id: str) -> Optional[StateChain]:
        """Get a timeline by ID"""
        return self.timelines.get(timeline_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "total_operations": self.operation_count,
            "active_timelines": len(self.timelines),
            "total_states": sum(len(chain) for chain in self.timelines.values()),
            "verifications": self.verifier.verification_count,
            "config": {
                "peak_flops": self.config.peak_flops,
                "deterministic": self.config.deterministic,
                "merkle_verified": self.config.merkle_verified,
                "paradox_resolution": self.config.paradox_resolution,
            },
        }
