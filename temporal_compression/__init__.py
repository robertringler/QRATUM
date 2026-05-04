"""Temporal Compression Experimental Framework.

This module implements experimental validation of deterministic temporal
compression in computational systems, designed for DARPA/NSF requirements.

The framework provides:
- Deterministic, rule-based dynamical systems (cellular automata)
- Temporal compression metrics calculation
- Cryptographic verification via Merkle-chain audit trails
- Reproducibility validation across multiple runs
- Branching simulation exploration

Usage:
    from temporal_compression import (
        TemporalCompressionExperiment,
        CellularAutomaton,
        TemporalMetrics,
        VerificationChain,
    )

    # Create deterministic system
    automaton = CellularAutomaton(rule=110, size=100, seed=42)

    # Run experiment
    experiment = TemporalCompressionExperiment(automaton)
    results = experiment.run(simulated_steps=10000)

    # Verify reproducibility
    assert results.is_reproducible
    print(f"Temporal compression ratio: {results.compression_ratio:.2f}×")
"""

from .automata import CellularAutomaton, Rule30, Rule110
from .experiment import ExperimentConfig, TemporalCompressionExperiment
from .metrics import CompressionStatistics, TemporalMetrics
from .verification import AuditEntry, StateVerifier, VerificationChain

__version__ = "1.0.0"

__all__ = [
    # Automata
    "CellularAutomaton",
    "Rule110",
    "Rule30",
    # Experiment
    "TemporalCompressionExperiment",
    "ExperimentConfig",
    # Metrics
    "TemporalMetrics",
    "CompressionStatistics",
    # Verification
    "VerificationChain",
    "StateVerifier",
    "AuditEntry",
]
