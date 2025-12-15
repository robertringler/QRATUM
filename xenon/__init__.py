"""XENON: Xenobiotic Execution Network for Organismal Neurosymbolic reasoning.

A post-GPU biological intelligence platform that replaces tensor-based approaches
with mechanism-based continuous learning.

Core Paradigm Shift:
- Computational primitive: Biological mechanism DAGs (not tensors)
- Learning: Sequential Bayesian updating (not gradient descent)
- Value: Mechanistic explanations with provenance (not static weights)
- Moat: Non-exportable experimental history (not replicable datasets)

Architecture:
1. Mechanism representation: Directed acyclic graphs of molecular interactions
2. Bayesian learning: Sequential updates from experimental results
3. Stochastic simulation: Gillespie SSA, Langevin dynamics
4. Symbolic reasoning: GO/ChEBI ontologies, constraint solving
5. Mechanism repository: Versioned storage with experimental provenance
6. Hypothesis generation: Mechanism synthesis and epistemic ranking
7. XENON runtime: Continuous learning loop (no epochs)

Usage:
    from xenon import XENONRuntime
    
    runtime = XENONRuntime()
    runtime.add_target(
        name="KRAS_G12C_inhibition",
        protein="KRAS",
        mutation="G12C",
        objective="find_inhibitor"
    )
    runtime.run(max_iterations=100)
    mechanisms = runtime.get_mechanisms(min_evidence=0.9)
"""

__version__ = "0.1.0"
__author__ = "XENON Project"

from .core.mechanism import BioMechanism, MolecularState, Transition
from .learning.bayesian_updater import BayesianUpdater
from .simulation.gillespie import GillespieSimulator
from .runtime.xenon_kernel import XENONRuntime

__all__ = [
    "BioMechanism",
    "MolecularState",
    "Transition",
    "BayesianUpdater",
    "GillespieSimulator",
    "XENONRuntime",
]
"""XENON: Bio-mechanism simulation and visualization system."""

__version__ = "0.1.0"

from xenon.core.mechanism import BioMechanism, MolecularState, Transition

__all__ = ["BioMechanism", "MolecularState", "Transition"]
