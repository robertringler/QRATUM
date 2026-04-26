"""AION - Adaptive Integrated Omni-Language Substrate.

A universal polyglot execution language with cross-language/hardware fusion,
zero-copy memory, proof-preserving execution, and adaptive runtime scheduling.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "QRATUM Team"

from . import executor
from .backends import llvm_backend, wasm_backend
from .concurrency import lattice
from .lifters import c_lifter, cuda_lifter, sql_lifter
from .memory import regions
from .optimization import fusion, scheduler
from .proof import synthesis, verifier
from .sir import edges, hypergraph, vertices
from .types import type_system

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
    "executor",
]
