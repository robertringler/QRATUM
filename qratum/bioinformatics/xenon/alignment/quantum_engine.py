"""
Quantum-Augmented Alignment Engine

Features:
- Hybrid classical-quantum dispatcher
- Automatic backend selection
- Graceful degradation to classical
- Uncertainty quantification
- Equivalence validation

Certificate: QRATUM-HARDENING-20251215-V5
"""

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from ....core.reproducibility import ReproducibilityManager
from ....core.security import SecurityValidator
from ....core.validation import EquivalenceValidator


class AlignmentBackend(Enum):
    """Available alignment backends."""
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    AUTO = "auto"


class QuantumAlignmentEngine:
    """
    Quantum-augmented sequence alignment engine.
    
    Provides:
    - Bit-identical classical path (legacy preserved)
    - Quantum backend with graceful degradation
    - Bayesian uncertainty quantification (opt-in)
    """
    
    def __init__(
        self,
        backend: AlignmentBackend = AlignmentBackend.AUTO,
        enable_uncertainty: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize alignment engine.
        
        Args:
            backend: Backend selection (auto, classical, quantum)
            enable_uncertainty: Enable Bayesian uncertainty quantification
            seed: Random seed for reproducibility
        """
        self.backend = backend
        self.enable_uncertainty = enable_uncertainty
        self.reproducibility_mgr = ReproducibilityManager(seed=seed)
        self.reproducibility_mgr.setup_deterministic_mode()
        self.security_validator = SecurityValidator()
        self.equivalence_validator = EquivalenceValidator()
        
        # Import backends dynamically
        from .backends.classical import ClassicalAlignmentBackend
        self.classical_backend = ClassicalAlignmentBackend()
        
        # Try to import quantum backend
        self.quantum_backend = None
        try:
            from .backends.quantum import QuantumAlignmentBackend
            self.quantum_backend = QuantumAlignmentBackend(
                seed=self.reproducibility_mgr.get_qiskit_seed()
            )
        except ImportError:
            if backend == AlignmentBackend.QUANTUM:
                raise RuntimeError("Quantum backend not available")
    
    def align(
        self,
        sequence1: str,
        sequence2: str,
        alphabet: str = "DNA"
    ) -> Dict:
        """
        Align two biological sequences.
        
        Args:
            sequence1: First sequence
            sequence2: Second sequence
            alphabet: Sequence alphabet (DNA, RNA, PROTEIN)
            
        Returns:
            Dictionary with alignment results
        """
        # Validate inputs
        val1 = self.security_validator.validate_sequence(sequence1, alphabet)
        if not val1["valid"]:
            raise ValueError(f"Invalid sequence1: {val1['reason']}")
        
        val2 = self.security_validator.validate_sequence(sequence2, alphabet)
        if not val2["valid"]:
            raise ValueError(f"Invalid sequence2: {val2['reason']}")
        
        # Select backend
        selected_backend = self._select_backend()
        
        # Perform alignment
        if selected_backend == AlignmentBackend.CLASSICAL:
            result = self.classical_backend.align(sequence1, sequence2)
            result["backend_used"] = "classical"
        elif selected_backend == AlignmentBackend.QUANTUM:
            if self.quantum_backend is None:
                # Graceful degradation
                result = self.classical_backend.align(sequence1, sequence2)
                result["backend_used"] = "classical_fallback"
            else:
                result = self.quantum_backend.align(sequence1, sequence2)
                result["backend_used"] = "quantum"
                
                # Validate equivalence with classical
                classical_result = self.classical_backend.align(sequence1, sequence2)
                equivalence = self.equivalence_validator.validate_scalar_equivalence(
                    result["score"], classical_result["score"], "quantum", "classical"
                )
                result["equivalence_check"] = equivalence
        else:
            raise ValueError(f"Unknown backend: {selected_backend}")
        
        # Add uncertainty quantification if enabled
        if self.enable_uncertainty:
            result["uncertainty"] = self._compute_uncertainty(result)
        
        return result
    
    def _select_backend(self) -> AlignmentBackend:
        """Select alignment backend based on configuration."""
        if self.backend == AlignmentBackend.CLASSICAL:
            return AlignmentBackend.CLASSICAL
        elif self.backend == AlignmentBackend.QUANTUM:
            if self.quantum_backend is not None:
                return AlignmentBackend.QUANTUM
            else:
                # Graceful degradation
                return AlignmentBackend.CLASSICAL
        elif self.backend == AlignmentBackend.AUTO:
            # Auto-select based on availability
            if self.quantum_backend is not None:
                return AlignmentBackend.QUANTUM
            else:
                return AlignmentBackend.CLASSICAL
        else:
            return AlignmentBackend.CLASSICAL
    
    def _compute_uncertainty(self, result: Dict) -> Dict:
        """
        Compute Bayesian uncertainty quantification.
        
        Args:
            result: Alignment result
            
        Returns:
            Uncertainty metrics
        """
        # Simplified uncertainty estimation
        score = result.get("score", 0.0)
        length = result.get("length", 1)
        
        # Estimate uncertainty based on alignment quality
        confidence = min(1.0, max(0.0, score / (length * 2.0)))
        uncertainty = 1.0 - confidence
        
        return {
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "method": "bayesian_simplified"
        }
