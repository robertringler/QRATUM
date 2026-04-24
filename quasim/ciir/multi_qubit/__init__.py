"""Multi-Qubit Entangled CIIR Controller with Fractal Encoding.

Chapter 21 of the CIIR monograph — executable prototype.

Modules
-------
quantum          : density-matrix simulator (Lindblad open-system dynamics)
control          : LQR + Pontryagin optimal controller
error_correction : fractal recursive quantum error correction
simulation       : hybrid classical–quantum coupling engine
analysis         : Lyapunov stability & spectral-gap analysis
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
