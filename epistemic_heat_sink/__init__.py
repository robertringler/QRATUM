"""Epistemic Heat Sink - Neurosymbolic Reasoning with zkML Integration.

This module implements the epistemic heat sink for the QRATUM ecosystem,
providing neurosymbolic reasoning with concept bottlenecks and zkML
inference proofs using Plonky3.

Core Principles:
- Error is thermodynamically expensive
- Evolution occurs only in provably safe subspaces
- Power cannot override truth
- Intelligence is optional; verifiability is mandatory

Implemented Features:
- Neurosymbolic reasoning with concept bottlenecks
- zkML inference proofs (Plonky3 integration)
- Folding schemes for incremental proof chains
- Cryptographic audit attestation

Version: 1.0.0
Status: Production
"""

__version__ = "1.0.0"
__author__ = "QRATUM Team"

from .heat_sink import (
    EpistemicHeatSink,
    EpistemicState,
    ErrorCost,
    create_heat_sink,
)
from .neurosymbolic import (
    ConceptBottleneck,
    NeurosymbolicReasoner,
    ReasoningTrace,
    SymbolicConcept,
    create_concept_bottleneck,
)
from .zkml import (
    FoldingScheme,
    IncrementalProofChain,
    Plonky3ProofSystem,
    ZKMLInferenceProof,
    create_zkml_prover,
)

__all__ = [
    # Neurosymbolic
    "ConceptBottleneck",
    "NeurosymbolicReasoner",
    "SymbolicConcept",
    "ReasoningTrace",
    "create_concept_bottleneck",
    # zkML
    "ZKMLInferenceProof",
    "Plonky3ProofSystem",
    "FoldingScheme",
    "IncrementalProofChain",
    "create_zkml_prover",
    # Heat Sink
    "EpistemicHeatSink",
    "EpistemicState",
    "ErrorCost",
    "create_heat_sink",
]
