"""
QRATUM ExaScale Deterministic Compiler Toolchain

A comprehensive compiler infrastructure ensuring bit-identical binaries
across all 3,125 CN-QES nodes with cryptographic verification.

Modules:
- pipeline: Deterministic compilation pipeline orchestration
- ptx_gen: Reproducible PTX/SASS generation (FMA disabled)
- verifier: Cross-node binary identity verification

Key Guarantees:
- Bit-identical binaries across all nodes
- Deterministic PTX/SASS generation
- FMA instructions disabled for reproducibility
- Merkle-verified compilation artifacts
- Byzantine-fault-tolerant verification
- Zero-knowledge proofs for execution equivalence

Usage:
    from qratum.exascale.compiler import get_default_pipeline

    pipeline = get_default_pipeline()
    success, artifacts = pipeline.full_pipeline(sources, output)
"""

from .pipeline import (
    BuildEnvironment,
    CompilationArtifact,
    CompilationStage,
    CompilerFlags,
    DeterminismViolation,
    DeterministicCompilationPipeline,
    OptimizationLevel,
    get_default_pipeline,
)
from .ptx_gen import (
    DeterministicPTXGenerator,
    DeterministicSASSGenerator,
    InstructionScheduling,
    PTXArtifact,
    PTXGenerationConfig,
    PTXVersion,
    RegisterAllocationStrategy,
    SASSArchitecture,
    SASSArtifact,
    SASSGenerationConfig,
    get_default_ptx_generator,
    get_default_sass_generator,
)
from .verifier import (
    BinaryArtifact,
    ConsensusProtocol,
    CrossNodeVerifier,
    HashAlgorithm,
    MerkleNode,
    MerkleTree,
    VerificationLevel,
    VerificationResult,
    VerificationStatus,
    get_default_verifier,
    verify_compilation_reproducibility,
)

__all__ = [
    # Pipeline
    "CompilationStage",
    "OptimizationLevel",
    "DeterminismViolation",
    "CompilerFlags",
    "BuildEnvironment",
    "CompilationArtifact",
    "DeterministicCompilationPipeline",
    "get_default_pipeline",
    # PTX/SASS
    "PTXVersion",
    "SASSArchitecture",
    "InstructionScheduling",
    "RegisterAllocationStrategy",
    "PTXGenerationConfig",
    "SASSGenerationConfig",
    "PTXArtifact",
    "SASSArtifact",
    "DeterministicPTXGenerator",
    "DeterministicSASSGenerator",
    "get_default_ptx_generator",
    "get_default_sass_generator",
    # Verifier
    "VerificationLevel",
    "VerificationStatus",
    "HashAlgorithm",
    "ConsensusProtocol",
    "BinaryArtifact",
    "MerkleNode",
    "MerkleTree",
    "VerificationResult",
    "CrossNodeVerifier",
    "get_default_verifier",
    "verify_compilation_reproducibility",
]
