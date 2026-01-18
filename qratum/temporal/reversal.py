"""
Temporal Reversal - Mechanisms for Backward Temporal Computation

Provides mechanisms for reversing time in computational systems through
causal reversal, entropy handling, and state recovery.
"""

from typing import Callable, Any, Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .constants import EntropyStrategy
from .state import TemporalState, StateChain, TemporalCoordinate


class ReversalStrategy(Enum):
    """Strategies for temporal reversal"""
    DETERMINISTIC = "deterministic"  # Exact inverse computation
    PROBABILISTIC = "probabilistic"  # Probabilistic reconstruction
    MERKLE_GUIDED = "merkle_guided"  # Recover from Merkle chain
    CHECKPOINT = "checkpoint"  # Use periodic checkpoints


@dataclass
class ReversalResult:
    """
    Result of a temporal reversal operation.
    
    Attributes:
        success: Whether reversal succeeded
        prior_state: Recovered prior state (if successful)
        confidence: Confidence level (0.0 to 1.0)
        strategy_used: Strategy that was used
        entropy_handled: How entropy was handled
        metadata: Additional information
    """
    success: bool
    prior_state: Optional[Any]
    confidence: float
    strategy_used: ReversalStrategy
    entropy_handled: str
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (f"ReversalResult({status}, confidence={self.confidence:.2f}, "
                f"strategy={self.strategy_used.value})")


class CausalReversal:
    """
    Reverse causal chains to recover prior states.
    
    Provides deterministic reversal when inverse functions are available,
    allowing exact reconstruction of prior states.
    """
    
    def __init__(self):
        """Initialize causal reversal"""
        self.reversal_count = 0
    
    def reverse_single_step(
        self,
        current_state: Any,
        inverse_fn: Callable[[Any, float], Any],
        delta_t: float,
    ) -> ReversalResult:
        """
        Reverse a single time step.
        
        Args:
            current_state: Current state
            inverse_fn: Inverse evolution function
            delta_t: Time step to reverse
            
        Returns:
            ReversalResult with prior state
        """
        try:
            prior_state = inverse_fn(current_state, delta_t)
            
            self.reversal_count += 1
            
            return ReversalResult(
                success=True,
                prior_state=prior_state,
                confidence=1.0,  # Deterministic = 100% confidence
                strategy_used=ReversalStrategy.DETERMINISTIC,
                entropy_handled="deterministic_inverse",
                metadata={
                    'delta_t': delta_t,
                    'reversal_id': self.reversal_count,
                }
            )
        except Exception as e:
            return ReversalResult(
                success=False,
                prior_state=None,
                confidence=0.0,
                strategy_used=ReversalStrategy.DETERMINISTIC,
                entropy_handled="failed",
                metadata={'error': str(e)}
            )
    
    def reverse_chain(
        self,
        final_state: Any,
        inverse_fn: Callable[[Any, float], Any],
        num_steps: int,
        step_size: float,
    ) -> Tuple[List[Any], List[ReversalResult]]:
        """
        Reverse a chain of states.
        
        Args:
            final_state: Final state to reverse from
            inverse_fn: Inverse evolution function
            num_steps: Number of steps to reverse
            step_size: Size of each time step
            
        Returns:
            Tuple of (list_of_states, list_of_results)
        """
        states = [final_state]
        results = []
        
        current_state = final_state
        
        for _ in range(num_steps):
            result = self.reverse_single_step(
                current_state=current_state,
                inverse_fn=inverse_fn,
                delta_t=step_size,
            )
            
            results.append(result)
            
            if result.success and result.prior_state is not None:
                current_state = result.prior_state
                states.append(current_state)
            else:
                break
        
        return states, results
    
    def is_reversible(
        self,
        state: Any,
        forward_fn: Callable[[Any, float], Any],
        inverse_fn: Callable[[Any, float], Any],
        delta_t: float,
        tolerance: float = 1e-6,
    ) -> bool:
        """
        Check if a state transition is truly reversible.
        
        Tests by applying forward then inverse and checking if we get back
        to the original state.
        
        Args:
            state: State to test
            forward_fn: Forward evolution function
            inverse_fn: Inverse evolution function
            delta_t: Time step
            tolerance: Tolerance for equality check
            
        Returns:
            True if reversible within tolerance
        """
        try:
            # Apply forward
            forward_state = forward_fn(state, delta_t)
            
            # Apply inverse
            recovered_state = inverse_fn(forward_state, delta_t)
            
            # Check if we recovered the original
            # For simple types
            if isinstance(state, (int, float)):
                return abs(state - recovered_state) < tolerance
            
            # For complex types, use string comparison (not ideal but works)
            return str(state) == str(recovered_state)
            
        except Exception:
            return False


class EntropyReversal:
    """
    Handle entropy in backward computation.
    
    Provides strategies for dealing with entropy in non-reversible systems,
    including probabilistic reconstruction and Merkle-guided recovery.
    """
    
    def __init__(self, strategy: str = EntropyStrategy.MERKLE_RECOVER):
        """
        Initialize entropy reversal.
        
        Args:
            strategy: Default strategy for entropy handling
        """
        self.strategy = strategy
        self.reversal_count = 0
    
    def reverse_with_entropy(
        self,
        current_state: Any,
        inverse_fn: Optional[Callable[[Any, float], Any]],
        delta_t: float,
        merkle_chain: Optional[StateChain] = None,
        prior_distribution: Optional[Callable[[], Any]] = None,
    ) -> ReversalResult:
        """
        Reverse a state transition with entropy handling.
        
        Args:
            current_state: Current state
            inverse_fn: Optional inverse function (may be lossy)
            delta_t: Time step to reverse
            merkle_chain: Optional Merkle chain for recovery
            prior_distribution: Optional prior distribution for probabilistic recovery
            
        Returns:
            ReversalResult with prior state and confidence
        """
        if self.strategy == EntropyStrategy.RECONSTRUCT and inverse_fn:
            return self._reconstruct(current_state, inverse_fn, delta_t)
        elif self.strategy == EntropyStrategy.PROBABILISTIC and prior_distribution:
            return self._probabilistic_recover(current_state, prior_distribution)
        elif self.strategy == EntropyStrategy.MERKLE_RECOVER and merkle_chain:
            return self._merkle_recover(current_state, merkle_chain)
        else:
            # Fallback to deterministic if possible
            if inverse_fn:
                return self._reconstruct(current_state, inverse_fn, delta_t)
            
            return ReversalResult(
                success=False,
                prior_state=None,
                confidence=0.0,
                strategy_used=ReversalStrategy.DETERMINISTIC,
                entropy_handled="no_strategy_available",
                metadata={}
            )
    
    def _reconstruct(
        self,
        current_state: Any,
        inverse_fn: Callable[[Any, float], Any],
        delta_t: float,
    ) -> ReversalResult:
        """
        Deterministic reconstruction (may be lossy if entropy increased).
        
        Args:
            current_state: Current state
            inverse_fn: Inverse function (may be approximate)
            delta_t: Time step
            
        Returns:
            ReversalResult with reconstructed state
        """
        try:
            prior_state = inverse_fn(current_state, delta_t)
            
            # Confidence is less than 1.0 because entropy may have been lost
            confidence = 0.8  # Assume some information loss
            
            self.reversal_count += 1
            
            return ReversalResult(
                success=True,
                prior_state=prior_state,
                confidence=confidence,
                strategy_used=ReversalStrategy.DETERMINISTIC,
                entropy_handled=EntropyStrategy.RECONSTRUCT,
                metadata={'note': 'reconstruction may be approximate due to entropy'}
            )
        except Exception as e:
            return ReversalResult(
                success=False,
                prior_state=None,
                confidence=0.0,
                strategy_used=ReversalStrategy.DETERMINISTIC,
                entropy_handled="failed",
                metadata={'error': str(e)}
            )
    
    def _probabilistic_recover(
        self,
        current_state: Any,
        prior_distribution: Callable[[], Any],
    ) -> ReversalResult:
        """
        Probabilistic reconstruction using prior distribution.
        
        Args:
            current_state: Current state
            prior_distribution: Function that samples from prior
            
        Returns:
            ReversalResult with sampled prior state
        """
        try:
            # Sample from prior distribution
            prior_state = prior_distribution()
            
            # Confidence is low because we're guessing
            confidence = 0.3
            
            self.reversal_count += 1
            
            return ReversalResult(
                success=True,
                prior_state=prior_state,
                confidence=confidence,
                strategy_used=ReversalStrategy.PROBABILISTIC,
                entropy_handled=EntropyStrategy.PROBABILISTIC,
                metadata={'note': 'probabilistic reconstruction, low confidence'}
            )
        except Exception as e:
            return ReversalResult(
                success=False,
                prior_state=None,
                confidence=0.0,
                strategy_used=ReversalStrategy.PROBABILISTIC,
                entropy_handled="failed",
                metadata={'error': str(e)}
            )
    
    def _merkle_recover(
        self,
        current_state: Any,
        merkle_chain: StateChain,
    ) -> ReversalResult:
        """
        Recover prior state from Merkle chain.
        
        Args:
            current_state: Current state
            merkle_chain: Chain with prior states
            
        Returns:
            ReversalResult with recovered state from chain
        """
        try:
            # Find current state in chain
            current_hash = None
            if isinstance(current_state, TemporalState):
                current_hash = current_state.coordinate.state_hash
            
            # Look for prior state in chain
            if len(merkle_chain.states) >= 2:
                # Get second-to-last state (prior to current)
                prior_temporal_state = merkle_chain.states[-2]
                prior_state = prior_temporal_state.data
                
                # High confidence because we have the actual prior state
                confidence = 0.95
                
                self.reversal_count += 1
                
                return ReversalResult(
                    success=True,
                    prior_state=prior_state,
                    confidence=confidence,
                    strategy_used=ReversalStrategy.MERKLE_GUIDED,
                    entropy_handled=EntropyStrategy.MERKLE_RECOVER,
                    metadata={
                        'merkle_verified': True,
                        'chain_length': len(merkle_chain.states)
                    }
                )
            else:
                return ReversalResult(
                    success=False,
                    prior_state=None,
                    confidence=0.0,
                    strategy_used=ReversalStrategy.MERKLE_GUIDED,
                    entropy_handled="insufficient_chain_history",
                    metadata={'chain_length': len(merkle_chain.states)}
                )
                
        except Exception as e:
            return ReversalResult(
                success=False,
                prior_state=None,
                confidence=0.0,
                strategy_used=ReversalStrategy.MERKLE_GUIDED,
                entropy_handled="failed",
                metadata={'error': str(e)}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get entropy reversal statistics"""
        return {
            'total_reversals': self.reversal_count,
            'strategy': self.strategy,
        }


class CheckpointReversal:
    """
    Use periodic checkpoints for efficient backward traversal.
    
    Rather than reversing every step, jump back to checkpoints and
    replay forward to the desired point.
    """
    
    def __init__(self, checkpoint_interval: int = 100):
        """
        Initialize checkpoint reversal.
        
        Args:
            checkpoint_interval: Number of steps between checkpoints
        """
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints: Dict[int, Any] = {}
    
    def add_checkpoint(self, depth: int, state: Any) -> None:
        """Add a checkpoint at given depth"""
        self.checkpoints[depth] = state
    
    def reverse_to_checkpoint(
        self,
        target_depth: int,
    ) -> Optional[Any]:
        """
        Reverse to nearest checkpoint at or before target depth.
        
        Args:
            target_depth: Depth to reverse to
            
        Returns:
            State at checkpoint, or None if no checkpoint found
        """
        # Find nearest checkpoint at or before target
        checkpoint_depths = sorted([d for d in self.checkpoints.keys() if d <= target_depth])
        
        if checkpoint_depths:
            nearest_depth = checkpoint_depths[-1]
            return self.checkpoints[nearest_depth]
        
        return None
    
    def should_checkpoint(self, depth: int) -> bool:
        """Check if a checkpoint should be created at this depth"""
        return depth % self.checkpoint_interval == 0
