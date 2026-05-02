"""Multi-Qubit Entangled CIIR Controller with Fractal Encoding.

Chapter 21 of the CIIR monograph — executable prototype.

Modules
-------
quantum          : density-matrix simulator (Lindblad open-system dynamics)
                   + non_markovian  (GAP 6 — memory-kernel GKSL)
                   + tensor_network (GAP 5 — MPS for N > 6)
control          : LQR + Pontryagin optimal controller
                   + geometric_control (GAP 7 — Fisher metric, natural gradient)
error_correction : fractal recursive quantum error correction
simulation       : hybrid classical–quantum coupling engine
analysis         : Lyapunov stability & spectral-gap analysis
                   + hardware_params (GAP 2+3 — latency calibration, gap scaling)
visualization    : Bloch sphere, entanglement entropy, phase diagrams
"""

from __future__ import annotations

__all__ = [
    "quantum",
    "control",
    "error_correction",
    "simulation",
    "analysis",
    "visualization",
]
