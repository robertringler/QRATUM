"""Core mechanism representation for XENON.

Provides the fundamental computational primitive: biological mechanism DAGs.
"""

from .mechanism import BioMechanism, MolecularState, Transition
from .mechanism_graph import MechanismGraph

__all__ = ["BioMechanism", "MolecularState", "Transition", "MechanismGraph"]
"""Core XENON data structures."""

from xenon.core.mechanism import BioMechanism, MolecularState, Transition

__all__ = ["BioMechanism", "MolecularState", "Transition"]
