"""
QRATUM Deterministic Compiler Toolchain

Ensures reproducible compilation and binary identity verification across
all compute nodes, with static analysis for determinism violations.

Components:
- pipeline: Deterministic compilation pipeline
- ptx_gen: Reproducible PTX/SASS generation
- verifier: Cross-node binary identity verification
"""

from .pipeline import DeterministicCompiler, CompilationPipeline
from .ptx_gen import PTXGenerator, SASSGenerator
from .verifier import BinaryVerifier, IdentityCheck

__all__ = [
    "DeterministicCompiler",
    "CompilationPipeline",
    "PTXGenerator",
    "SASSGenerator",
    "BinaryVerifier",
    "IdentityCheck",
]
