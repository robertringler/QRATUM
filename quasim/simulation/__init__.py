"""
QuASIM Simulation Modules

Symbolic and numerical simulation engines for quantum-classical systems.
"""

from .qcmg_field import (
    FieldState,
    QCMGParameters,
    QuantacosmorphysigeneticField,
)

__all__ = [
    "QuantacosmorphysigeneticField",
    "QCMGParameters",
    "FieldState",
]

__version__ = "0.1.0"
