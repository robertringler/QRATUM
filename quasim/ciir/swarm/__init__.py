r"""Programmable Physics Multi-Agent Swarm Framework.

Implements the system P = (S, R, U, Φ, C) where:
- S: State space (discrete, continuous, or hybrid)
- R: Rule set (physics programs)
- U: Update operator (execution engine)
- Φ: Observation/interface map
- C: Consistency constraints

ensuring U(S, R) → S' such that C(S') = 1.

Agents
------
AA  axiom_architect       Minimal, non-redundant axiom systems
FZ  formalizer            Symbolic mathematics conversion
RS  rule_synthesizer      Executable physical rule construction
SIM simulator             Rule execution over state spaces
IM  invariant_miner       Conserved quantities and symmetry detection
CV  consistency_validator  Logical consistency, closure, stability
EM  empirical_mapper      Mapping to known physics
CO  compression_optimizer Rule complexity minimization
HM  hypothesis_mutator    Rule perturbation and axiom mutation
ORC orchestrator          Coordinate agents, evolution loop

References
----------
CIIR Monograph Ch 20 (Programmable Physics).
"""

__version__ = "0.1.0"

from quasim.ciir.swarm.memory import KnowledgeGraph, KNode, KEdge
from quasim.ciir.swarm.physics_lang import (
    PhysicsProgram,
    StateDecl,
    RuleDecl,
    InteractionSpec,
    MetaRule,
)
from quasim.ciir.swarm.orchestrator import Orchestrator, EvolutionResult
from quasim.ciir.swarm.mppk import (
    MPPKEngine,
    MPPKGraph,
    MPPKNode,
    MPPKEdge,
    MPPKRule,
    UpdateOperator,
    Trajectory,
    GenerationResult,
    MetaLearningReport,
)

__all__ = [
    "KnowledgeGraph",
    "KNode",
    "KEdge",
    "PhysicsProgram",
    "StateDecl",
    "RuleDecl",
    "InteractionSpec",
    "MetaRule",
    "Orchestrator",
    "EvolutionResult",
    "MPPKEngine",
    "MPPKGraph",
    "MPPKNode",
    "MPPKEdge",
    "MPPKRule",
    "UpdateOperator",
    "Trajectory",
    "GenerationResult",
    "MetaLearningReport",
]
