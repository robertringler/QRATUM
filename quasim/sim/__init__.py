"""QuASIM Simulation Module.

This module provides field simulation capabilities including the
Quantacosmorphysigenetic (QCMG) field evolution system.
"""QCMG - Quantacosmomorphysigenetic Field Simulation Module.

This module provides simulation capabilities for coupled quantum-classical
field dynamics based on the Quantacosmomorphysigenetic (QCMG) model.
"""

from __future__ import annotations

from quasim.sim.qcmg_field import (
    QCMGParameters,
    FieldState,
    QuantacosmomorphysigeneticField,
)

__version__ = "0.1.0"

__all__ = [
    "QCMGParameters",
    "FieldState",
    "QuantacosmomorphysigeneticField",
    "__version__",
from quasim.sim.qcmg import (
    QCMGParameters,
    QCMGState,
    QuantacosmorphysigeneticField,
)

__all__ = [
    "QCMGParameters",
    "QCMGState",
    "QuantacosmorphysigeneticField",
]
