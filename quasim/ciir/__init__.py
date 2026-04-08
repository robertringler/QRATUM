"""CIIR → QuASIM → QRATUM simulation pipeline.

Implements the Cognitive-Information Integration and Representation (CIIR)
framework bridging QuASIM quantum state dynamics to QRATUM adaptive
operator evolution.
"""

from quasim.ciir.core import density_matrix, normalize, von_neumann_entropy
from quasim.ciir.engine import CIIRSimulationEngine, SimulationConfig
from quasim.ciir.operators import OperatorSystem

__all__ = [
    "CIIRSimulationEngine",
    "OperatorSystem",
    "SimulationConfig",
    "density_matrix",
    "normalize",
    "von_neumann_entropy",
]
