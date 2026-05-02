"""Search algorithms for QRATUM-Chess.

Implements multiple search strategies:
1. Alpha-Beta with iterative deepening
2. Monte Carlo Tree Search (MCTS)
3. Asymmetric Adaptive Search (AAS) for Stage III
4. AAS Canonical Kernel - Universal decision primitive
5. Self-Modifying Engine for meta-dynamics evolution (imported separately)
"""

from __future__ import annotations

from qratum_chess.search.aas import AsymmetricAdaptiveSearch
from qratum_chess.search.aas_kernel import (
    AASKernel,
    AASMetrics,
    ChessAASKernel,
    DepthBudget,
    EntropyGradient,
    OrthogonalSubspace,
    create_aas_kernel,
)
from qratum_chess.search.alphabeta import AlphaBetaSearch
from qratum_chess.search.mcts import MCTSSearch

__all__ = [
    "AlphaBetaSearch",
    "MCTSSearch",
    "AsymmetricAdaptiveSearch",
    # AAS Canonical Kernel
    "AASKernel",
    "ChessAASKernel",
    "EntropyGradient",
    "DepthBudget",
    "OrthogonalSubspace",
    "AASMetrics",
    "create_aas_kernel",
]


# Lazy import for self-modifying components to avoid circular imports
def get_self_modifying_engine():
    """Get SelfModifyingEngine class (lazy import to avoid circular imports)."""
    from qratum_chess.self_modifying import SelfModifyingEngine

    return SelfModifyingEngine


def get_self_modifying_search():
    """Get SelfModifyingSearch class (lazy import to avoid circular imports)."""
    from qratum_chess.self_modifying import SelfModifyingSearch

    return SelfModifyingSearch
