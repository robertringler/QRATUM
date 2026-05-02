"""Deterministic gate definitions."""

from __future__ import annotations

from ..core.circuits import QuantumGate


def pauli_x() -> QuantumGate:
    return QuantumGate(name="X", operation=lambda amp: -amp)


def identity() -> QuantumGate:
    return QuantumGate(name="I", operation=lambda amp: amp)
