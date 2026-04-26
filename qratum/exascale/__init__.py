"""
QRATUM ExaScale Initiative

A trust + speed dominant exascale platform that exceeds El Capitan on every
measurable metric while maintaining bit-identical reproducibility, post-quantum
security, and federated sovereign deployment capabilities.

Target Rankings:
- TOP500: #1 (≥ 2.0 ExaFLOPS Rmax)
- HPCG: #1 (≥ 20 PFLOPS)
- Green500: #1 (> 100 GFLOPS/W)
- Graph500: #1

Core Modules:
- hardware: CN-QES node specifications and topology
- network: AetherFabric-X deterministic interconnect
- runtime: QDR (QRATUM Deterministic Runtime)
- compiler: Deterministic compilation toolchain
- security: Post-quantum cryptography integration
- quantum: PTAQ-1000 quantum-classical hybrid
- federation: Byzantine-fault-tolerant control plane
"""

__version__ = "1.0.0"
__all__ = [
    "hardware",
    "network",
    "runtime",
    "compiler",
    "security",
    "quantum",
    "federation",
]

# Performance targets (hard requirements)
PERFORMANCE_TARGETS = {
    "peak_fp64_eflops": 2.5,  # ExaFLOPS
    "linpack_rmax_eflops": 2.0,  # ExaFLOPS
    "peak_fp8_eflops": 40.0,  # ExaFLOPS for AI workloads
    "hpcg_pflops": 20.0,  # PFLOPS
    "hpl_mxp_eflops": 15.0,  # ExaFLOPS mixed precision
    "total_power_mw": 25.0,  # MW
    "fp64_efficiency_gflops_per_watt": 100.0,  # GFLOPS/W
    "pue": 1.05,
}

# Infrastructure scale
INFRASTRUCTURE_SCALE = {
    "gpus": 50000,  # NVIDIA GB200 NVL72
    "cpu_cores": 16000000,
    "system_memory_pb": 11.0,  # Petabytes
    "storage_eb": 2.0,  # Exabytes
    "bisection_bandwidth_pbs": 100.0,  # PB/s
}

# Absolute guarantees
GUARANTEES = [
    "Bit-identical results across all runs (Merkle-verified)",
    "Deterministic collective operations (fixed-order AllReduce)",
    "Post-quantum cryptography everywhere (Kyber, Dilithium, SPHINCS+)",
    "Zero-knowledge result verification (Halo2 proofs)",
    "Byzantine-fault-tolerant control plane",
    "Air-gap/sovereign deployment capability",
]
