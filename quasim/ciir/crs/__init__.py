r"""Computational Reality System (CRS) — graph-based computational substrate.

Layers
------
0  graph         Primitive substrate: causal graph with typed nodes and weighted edges.
1  rewrite       Local update operator: rewrite rules as physics-law analogs.
2  evolution     Global evolution engine: causally consistent graph updates.
3  spacetime     Emergent spacetime reconstruction from graph topology.
4  branching     Quantum-like branching with interference and collapse.
5  conservation  Noether-analog conservation law engine.
6  observer      Measurement model with information loss.
7  integration   CIIR / QuASIM / QRATUM bridge.

References
----------
CIIR monograph Ch 3–8; CRS internal spec §0–§7.
"""

__version__ = "0.1.0"

from quasim.ciir.crs.graph import CRSGraph, Node, Edge
from quasim.ciir.crs.evolution import CRSEngine
from quasim.ciir.crs.branching import BranchState
from quasim.ciir.crs.observer import Observer
from quasim.ciir.crs.integration import QRATUM_CRS_Core

__all__ = [
    "CRSGraph",
    "Node",
    "Edge",
    "CRSEngine",
    "BranchState",
    "Observer",
    "QRATUM_CRS_Core",
]
