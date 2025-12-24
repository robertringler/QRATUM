"""AION - Adaptive Integrated Omni-Language Substrate.

A universal polyglot execution language with cross-language/hardware fusion,
zero-copy memory, proof-preserving execution, and adaptive runtime scheduling.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "QRATUM Team"

from .sir import hypergraph, vertices, edges
from .memory import regions
from .concurrency import lattice
from .types import type_system
from .proof import verifier, synthesis
from .lifters import c_lifter, cuda_lifter, sql_lifter
from .optimization import fusion, scheduler
from .backends import llvm_backend, wasm_backend

__all__ = [
    "hypergraph",
    "vertices",
    "edges",
    "regions",
    "lattice",
    "type_system",
    "verifier",
    "synthesis",
    "c_lifter",
    "cuda_lifter",
    "sql_lifter",
    "fusion",
    "scheduler",
    "llvm_backend",
    "wasm_backend",
]
