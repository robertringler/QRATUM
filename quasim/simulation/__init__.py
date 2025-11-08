"""
QuASIM Simulation Modules

Symbolic and numerical simulation engines for quantum-classical systems.
"""

from .qcmg_field import (
    QuantacosmorphysigeneticField,
    QCMGParameters,
    FieldState,
)
from .quantacosmic import (
    ActionFunctional,
    CouplingOperator,
    FieldLattice,
    MetricTensor,
    QuantumManifold,
    coupling_matrix,
    curvature_scalar,
    propagate_field,
    revultra_temporal_curvature,
)
from .qcmg_field import (FieldState, QCMGParameters,
                         QuantacosmorphysigeneticField)

__all__ = [
    "QuantacosmorphysigeneticField",
    "QCMGParameters",
    "FieldState",
    "QuantumManifold",
    "curvature_scalar",
    "FieldLattice",
    "propagate_field",
    "CouplingOperator",
    "coupling_matrix",
    "ActionFunctional",
    "MetricTensor",
    "revultra_temporal_curvature",
]

__version__ = "0.1.0"
