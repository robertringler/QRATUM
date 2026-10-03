"""FLUXA -- the QRATUM energy-system vertical, executed on the QRADLE substrate.

FLUXA models an electrical energy system (generation, demand, storage,
transmission) and runs it as a QRADLE domain contract so that every state
transition is deterministic, authorized, event-logged, Merkle-chained and
recoverable.

Layering::

    FLUXA   model / network / profiles / scenarios / dispatch / state / metrics
      |
    QRADLE  DeterministicEngine, FatalInvariants, MerkleNode, RollbackManager
      |
    Q-Substrate   CPU (demonstrated); GPU / HPC / QPU abstractions exist in the
                  repository but are NOT exercised by this vertical.

Note on the name: ``qratum.verticals.fluxa`` and ``verticals.fluxa`` in this
repository implement a *supply-chain and logistics* module also called FLUXA.
This package is the energy-system FLUXA and is independent of those; see
``fluxa/README.md``.

Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"
__vertical__ = "FLUXA"
__domain__ = "electrical-energy-systems"

from fluxa.config import ConfigError, LoadedSystem, load_system
from fluxa.dispatch import DispatchError, DispatchProblem, DispatchSolution
from fluxa.engine import (
    AuthorizationDenied,
    PhysicalInvariantError,
    RunConfig,
    SimulationEngine,
    SimulationResult,
    validate_physical_state,
)
from fluxa.events import FluxaEvent, FluxaEventLedger, FluxaEventType, LedgerIntegrityError
from fluxa.metrics import RunMetrics, compute_all
from fluxa.model import (
    Battery,
    Bus,
    EnergySystem,
    Generator,
    GeneratorKind,
    Interconnection,
    Line,
)
from fluxa.network import DCNetwork
from fluxa.profiles import ExogenousSeries, ProfileParameters, TimeGrid, build_series
from fluxa.provenance import (
    ProvenanceBundle,
    build_inclusion_proof,
    build_merkle_tree,
    verify_inclusion_proof,
)
from fluxa.scenarios import Perturbation, PerturbationKind, Scenario, apply_scenario, load_scenarios
from fluxa.state import FluxaState, OperationalState

__all__ = [
    "__version__",
    "__vertical__",
    "__domain__",
    "AuthorizationDenied",
    "Battery",
    "Bus",
    "ConfigError",
    "DCNetwork",
    "DispatchError",
    "DispatchProblem",
    "DispatchSolution",
    "EnergySystem",
    "ExogenousSeries",
    "FluxaEvent",
    "FluxaEventLedger",
    "FluxaEventType",
    "FluxaState",
    "Generator",
    "GeneratorKind",
    "Interconnection",
    "LedgerIntegrityError",
    "Line",
    "LoadedSystem",
    "OperationalState",
    "Perturbation",
    "PerturbationKind",
    "PhysicalInvariantError",
    "ProfileParameters",
    "ProvenanceBundle",
    "RunConfig",
    "RunMetrics",
    "Scenario",
    "SimulationEngine",
    "SimulationResult",
    "TimeGrid",
    "apply_scenario",
    "build_inclusion_proof",
    "build_merkle_tree",
    "build_series",
    "compute_all",
    "load_scenarios",
    "load_system",
    "validate_physical_state",
    "verify_inclusion_proof",
]
