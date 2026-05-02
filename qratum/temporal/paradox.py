"""
Paradox Resolution for Temporal Computing

Handles temporal paradoxes in computational time travel through various
resolution strategies. Ensures causal consistency and prevents logical contradictions.

Strategies:
- NOVIKOV: Self-consistency principle (paradoxes fail to converge)
- MANY_WORLDS: All outcomes exist in alternate branches
- CHRONOLOGY_PROTECTION: Reject paradoxical computations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import ParadoxStrategy
from .state import StateChain, TemporalCoordinate, TemporalState


class ParadoxType(Enum):
    """Types of temporal paradoxes"""
    GRANDFATHER = "grandfather"  # Effect prevents its own cause
    BOOTSTRAP = "bootstrap"  # Information with no origin
    PREDESTINATION = "predestination"  # Circular causality
    CONSISTENCY = "consistency"  # State contradicts history
    ENTROPY = "entropy"  # Entropy decrease violation


@dataclass
class ParadoxDetection:
    """
    Record of a detected paradox.
    
    Attributes:
        paradox_type: Type of paradox detected
        timeline_id: Timeline where paradox occurred
        state_depth: Depth at which paradox was detected
        description: Human-readable description
        severity: Severity level (0.0 to 1.0)
        resolution: How the paradox was resolved
    """
    paradox_type: ParadoxType
    timeline_id: str
    state_depth: int
    description: str
    severity: float
    resolution: Optional[str] = None

    def __str__(self) -> str:
        return (f"{self.paradox_type.value.upper()} paradox at depth {self.state_depth} "
                f"in {self.timeline_id}: {self.description}")


class CausalityValidator:
    """
    Validates causal consistency of temporal computations.
    
    Ensures that cause always precedes effect and that no causal loops
    or contradictions exist in the timeline.
    """

    def __init__(self):
        """Initialize causality validator"""
        self.detections: List[ParadoxDetection] = []

    def validate_causal_chain(
        self,
        chain: StateChain,
        causality_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> Tuple[bool, List[ParadoxDetection]]:
        """
        Validate causal consistency of a state chain.
        
        Args:
            chain: State chain to validate
            causality_fn: Optional function to check causality between states
            
        Returns:
            Tuple of (is_valid, list_of_paradoxes)
        """
        paradoxes = []

        if len(chain.states) < 2:
            return True, paradoxes

        # Check for temporal ordering violations
        for i in range(1, len(chain.states)):
            prev_state = chain.states[i - 1]
            curr_state = chain.states[i]

            # Check computational time ordering
            if curr_state.coordinate.computational_t < prev_state.coordinate.computational_t:
                paradoxes.append(ParadoxDetection(
                    paradox_type=ParadoxType.CONSISTENCY,
                    timeline_id=chain.timeline_id,
                    state_depth=i,
                    description="Computational time moves backward",
                    severity=0.9,
                ))

            # Check physical time ordering
            if curr_state.coordinate.physical_t < prev_state.coordinate.physical_t:
                paradoxes.append(ParadoxDetection(
                    paradox_type=ParadoxType.CONSISTENCY,
                    timeline_id=chain.timeline_id,
                    state_depth=i,
                    description="Physical time moves backward",
                    severity=0.8,
                ))

            # Check parent hash consistency
            if curr_state.parent_hash != prev_state.coordinate.state_hash:
                paradoxes.append(ParadoxDetection(
                    paradox_type=ParadoxType.CONSISTENCY,
                    timeline_id=chain.timeline_id,
                    state_depth=i,
                    description="Parent hash mismatch",
                    severity=1.0,
                ))

        # Check for causal consistency using custom function
        if causality_fn:
            for i in range(1, len(chain.states)):
                prev_state = chain.states[i - 1]
                curr_state = chain.states[i]

                if not causality_fn(prev_state.data, curr_state.data):
                    paradoxes.append(ParadoxDetection(
                        paradox_type=ParadoxType.CONSISTENCY,
                        timeline_id=chain.timeline_id,
                        state_depth=i,
                        description="Causality violation detected by custom function",
                        severity=0.7,
                    ))

        self.detections.extend(paradoxes)
        return len(paradoxes) == 0, paradoxes

    def detect_grandfather_paradox(
        self,
        initial_state: TemporalState,
        final_state: TemporalState,
        prevents_existence_fn: Callable[[Any, Any], bool],
    ) -> Optional[ParadoxDetection]:
        """
        Detect grandfather paradox: effect prevents its own cause.
        
        Args:
            initial_state: Starting state
            final_state: Ending state
            prevents_existence_fn: Function to check if final prevents initial
            
        Returns:
            ParadoxDetection if paradox found, None otherwise
        """
        if prevents_existence_fn(final_state.data, initial_state.data):
            return ParadoxDetection(
                paradox_type=ParadoxType.GRANDFATHER,
                timeline_id=final_state.coordinate.timeline_id,
                state_depth=final_state.coordinate.depth,
                description="Final state prevents existence of initial state",
                severity=1.0,
            )
        return None

    def detect_bootstrap_paradox(
        self,
        chain: StateChain,
        has_external_origin_fn: Callable[[Any], bool],
    ) -> List[ParadoxDetection]:
        """
        Detect bootstrap paradox: information with no origin.
        
        Args:
            chain: State chain to check
            has_external_origin_fn: Function to check if state has external origin
            
        Returns:
            List of detected paradoxes
        """
        paradoxes = []

        for state in chain.states:
            if not has_external_origin_fn(state.data):
                paradoxes.append(ParadoxDetection(
                    paradox_type=ParadoxType.BOOTSTRAP,
                    timeline_id=chain.timeline_id,
                    state_depth=state.coordinate.depth,
                    description="Information appears without external origin",
                    severity=0.6,
                ))

        return paradoxes

    def detect_causal_loop(
        self,
        chain: StateChain,
        state_equality_fn: Callable[[Any, Any], bool],
    ) -> List[ParadoxDetection]:
        """
        Detect predestination paradox: circular causality.
        
        Args:
            chain: State chain to check
            state_equality_fn: Function to check if two states are equal
            
        Returns:
            List of detected paradoxes
        """
        paradoxes = []

        # Look for repeated states (potential loops)
        seen_states: Dict[int, int] = {}

        for i, state in enumerate(chain.states):
            state_id = hash(str(state.data))  # Simple hash

            if state_id in seen_states:
                prev_idx = seen_states[state_id]
                if state_equality_fn(chain.states[prev_idx].data, state.data):
                    paradoxes.append(ParadoxDetection(
                        paradox_type=ParadoxType.PREDESTINATION,
                        timeline_id=chain.timeline_id,
                        state_depth=i,
                        description=f"Causal loop detected between depth {prev_idx} and {i}",
                        severity=0.5,
                    ))
            else:
                seen_states[state_id] = i

        return paradoxes


class ParadoxResolver:
    """
    Resolves temporal paradoxes using configurable strategies.
    
    The resolver implements different approaches to handling paradoxes:
    - NOVIKOV: Self-consistency principle
    - MANY_WORLDS: Branch on paradox
    - CHRONOLOGY_PROTECTION: Reject paradoxical computations
    """

    def __init__(self, strategy: str = ParadoxStrategy.NOVIKOV):
        """
        Initialize paradox resolver.
        
        Args:
            strategy: Resolution strategy to use
        """
        self.strategy = strategy
        self.validator = CausalityValidator()
        self.resolutions: List[ParadoxDetection] = []

    def resolve(
        self,
        paradox: ParadoxDetection,
        chain: StateChain,
    ) -> Tuple[bool, Optional[StateChain]]:
        """
        Resolve a detected paradox using the configured strategy.
        
        Args:
            paradox: Detected paradox
            chain: State chain containing the paradox
            
        Returns:
            Tuple of (resolution_success, modified_chain)
        """
        if self.strategy == ParadoxStrategy.NOVIKOV:
            return self._resolve_novikov(paradox, chain)
        elif self.strategy == ParadoxStrategy.MANY_WORLDS:
            return self._resolve_many_worlds(paradox, chain)
        elif self.strategy == ParadoxStrategy.CHRONOLOGY_PROTECTION:
            return self._resolve_chronology_protection(paradox, chain)
        else:
            # Unknown strategy, default to chronology protection
            return self._resolve_chronology_protection(paradox, chain)

    def _resolve_novikov(
        self,
        paradox: ParadoxDetection,
        chain: StateChain,
    ) -> Tuple[bool, Optional[StateChain]]:
        """
        Resolve using Novikov self-consistency principle.
        
        Paradoxes are prevented by self-consistency: any action that would
        create a paradox simply fails to happen. The timeline "protects"
        itself by making paradoxical states unreachable.
        """
        # In Novikov self-consistency, paradoxes simply don't converge
        # We reject the paradoxical branch
        paradox.resolution = "novikov_rejected"
        self.resolutions.append(paradox)

        # Truncate chain at paradox point
        if paradox.state_depth > 0:
            truncated_chain = StateChain(timeline_id=chain.timeline_id)
            for i in range(paradox.state_depth):
                truncated_chain.add_state(chain.states[i])
            return True, truncated_chain

        return False, None

    def _resolve_many_worlds(
        self,
        paradox: ParadoxDetection,
        chain: StateChain,
    ) -> Tuple[bool, Optional[StateChain]]:
        """
        Resolve using Many-Worlds interpretation.
        
        Each possible outcome exists in a separate branch. Paradoxes are
        resolved by creating a new branch where the paradox doesn't exist.
        """
        # Create a new branch at the paradox point
        paradox.resolution = "many_worlds_branch"
        self.resolutions.append(paradox)

        # Create new timeline ID for the branch
        new_timeline_id = f"{chain.timeline_id}_mw_{len(self.resolutions)}"

        # Copy chain up to paradox point into new timeline
        new_chain = StateChain(timeline_id=new_timeline_id)
        for i in range(min(paradox.state_depth, len(chain.states))):
            state = chain.states[i]
            # Update timeline ID
            new_coord = TemporalCoordinate(
                state_hash=state.coordinate.state_hash,
                timeline_id=new_timeline_id,
                computational_t=state.coordinate.computational_t,
                physical_t=state.coordinate.physical_t,
                depth=state.coordinate.depth,
            )
            new_state = TemporalState(
                coordinate=new_coord,
                data=state.data,
                parent_hash=state.parent_hash,
                metadata=state.metadata.copy(),
            )
            new_chain.add_state(new_state)

        return True, new_chain

    def _resolve_chronology_protection(
        self,
        paradox: ParadoxDetection,
        chain: StateChain,
    ) -> Tuple[bool, Optional[StateChain]]:
        """
        Resolve using Chronology Protection Conjecture.
        
        Paradoxical computations are simply rejected. This is the most
        conservative approach that maintains strict causality.
        """
        paradox.resolution = "chronology_protection_rejected"
        self.resolutions.append(paradox)

        # Completely reject the paradoxical timeline
        return False, None

    def check_and_resolve(
        self,
        chain: StateChain,
        causality_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> Tuple[bool, StateChain, List[ParadoxDetection]]:
        """
        Check chain for paradoxes and resolve them.
        
        Args:
            chain: State chain to check
            causality_fn: Optional causality checking function
            
        Returns:
            Tuple of (is_valid, resolved_chain, detected_paradoxes)
        """
        # Validate causal chain
        is_valid, paradoxes = self.validator.validate_causal_chain(
            chain=chain,
            causality_fn=causality_fn,
        )

        if is_valid:
            return True, chain, []

        # Resolve each paradox
        current_chain = chain
        resolved_paradoxes = []

        for paradox in paradoxes:
            success, new_chain = self.resolve(paradox, current_chain)
            resolved_paradoxes.append(paradox)

            if not success:
                # Resolution failed
                return False, current_chain, resolved_paradoxes

            if new_chain:
                current_chain = new_chain

        return True, current_chain, resolved_paradoxes

    def get_statistics(self) -> Dict[str, Any]:
        """Get resolver statistics"""
        return {
            'strategy': self.strategy,
            'total_resolutions': len(self.resolutions),
            'total_detections': len(self.validator.detections),
            'resolutions_by_type': self._count_by_type(self.resolutions),
            'detections_by_type': self._count_by_type(self.validator.detections),
        }

    def _count_by_type(self, detections: List[ParadoxDetection]) -> Dict[str, int]:
        """Count paradoxes by type"""
        counts: Dict[str, int] = {}
        for detection in detections:
            ptype = detection.paradox_type.value
            counts[ptype] = counts.get(ptype, 0) + 1
        return counts
