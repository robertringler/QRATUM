"""DAG (Directed Acyclic Graph) primitives for execution planning.

This module provides structural types for representing computation graphs.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class NodeType(Enum):
    """Type of computation node in DAG.
    
    Attributes:
        CLASSICAL: Classical computation node
        QUANTUM: Quantum computation node
    """
    
    CLASSICAL = "classical"
    QUANTUM = "quantum"


@dataclass
class DAGNode:
    """Node in a directed acyclic graph representing a computation step.
    
    Attributes:
        id: Unique identifier for this node
        type: Node type (classical or quantum)
        dependencies: List of node IDs this node depends on
        payload: Arbitrary payload data for the computation
    """
    
    id: str
    type: NodeType
    dependencies: List[str]
    payload: object
