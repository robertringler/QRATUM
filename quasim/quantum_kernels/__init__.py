"""QRATUM Quantum-Inspired Kernels Module.

This module provides quantum-inspired algorithms and tensor network methods
for simulation acceleration. These classical algorithms capture some quantum
speedups while remaining verifiable and deterministic.

Key algorithms:
- MPS (Matrix Product States) for 1D quantum systems
- PEPS (Projected Entangled Pair States) for 2D systems (stub)
- VQE classical analog with AHTC acceleration hooks
- QAOA classical analog for combinatorial optimization
- Quantum kernel methods for ML enhancement

All kernels integrate with QRATUM's trust invariants:
- Deterministic execution with seed control
- Full provenance tracking
- Verification hooks for result validation
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "TensorNetworkConfig",
    "MPSSimulator",
    "PEPSSimulator",
    "ClassicalVQE",
    "ClassicalQAOA",
    "QuantumKernel",
    "AHTCAccelerator",
]

from .tensor_networks import MPSSimulator, PEPSSimulator, TensorNetworkConfig
from .classical_analogs import ClassicalVQE, ClassicalQAOA
from .quantum_kernels import QuantumKernel
from .ahtc import AHTCAccelerator
