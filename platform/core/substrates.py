"""Compute Substrate Definitions.

Mappings for optimal hardware selection based on task characteristics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ComputeSubstrate(Enum):
    """Hardware substrate types for computation."""

    CPU_SERIAL = "cpu_serial"  # Single-threaded CPU
    CPU_PARALLEL = "cpu_parallel"  # Multi-threaded CPU
    GPU_CUDA = "gpu_cuda"  # NVIDIA CUDA GPU
    GPU_METAL = "gpu_metal"  # Apple Metal GPU
    TPU = "tpu"  # Google TPU
    QUANTUM_SIM = "quantum_sim"  # Quantum simulator
    QUANTUM_HW = "quantum_hw"  # Quantum hardware
    DISTRIBUTED = "distributed"  # Distributed cluster
    NEUROMORPHIC = "neuromorphic"  # Neuromorphic hardware


@dataclass(frozen=True)
class SubstrateCapability:
    """Capability profile for a compute substrate.

    Attributes:
        substrate: The substrate type
        min_problem_size: Minimum efficient problem size
        max_problem_size: Maximum supported problem size
        specializations: Task types this substrate excels at
        relative_speed: Speed factor relative to CPU_SERIAL (1.0)
        power_efficiency: Power efficiency factor (higher is better)
        availability: Availability score (0-1)
    """

    substrate: ComputeSubstrate
    min_problem_size: int
    max_problem_size: int
    specializations: List[str]
    relative_speed: float = 1.0
    power_efficiency: float = 1.0
    availability: float = 1.0


# Substrate capability definitions
SUBSTRATE_CAPABILITIES = {
    ComputeSubstrate.CPU_SERIAL: SubstrateCapability(
        substrate=ComputeSubstrate.CPU_SERIAL,
        min_problem_size=1,
        max_problem_size=10000,
        specializations=["small_problems", "control_flow", "general"],
        relative_speed=1.0,
        power_efficiency=1.0,
        availability=1.0,
    ),
    ComputeSubstrate.CPU_PARALLEL: SubstrateCapability(
        substrate=ComputeSubstrate.CPU_PARALLEL,
        min_problem_size=100,
        max_problem_size=1000000,
        specializations=["parallel_algorithms", "data_processing", "optimization"],
        relative_speed=8.0,
        power_efficiency=0.8,
        availability=1.0,
    ),
    ComputeSubstrate.GPU_CUDA: SubstrateCapability(
        substrate=ComputeSubstrate.GPU_CUDA,
        min_problem_size=1000,
        max_problem_size=100000000,
        specializations=["matrix_ops", "neural_networks", "molecular_dynamics"],
        relative_speed=100.0,
        power_efficiency=0.6,
        availability=0.8,
    ),
    ComputeSubstrate.GPU_METAL: SubstrateCapability(
        substrate=ComputeSubstrate.GPU_METAL,
        min_problem_size=1000,
        max_problem_size=50000000,
        specializations=["matrix_ops", "graphics", "signal_processing"],
        relative_speed=80.0,
        power_efficiency=0.9,
        availability=0.7,
    ),
    ComputeSubstrate.TPU: SubstrateCapability(
        substrate=ComputeSubstrate.TPU,
        min_problem_size=10000,
        max_problem_size=1000000000,
        specializations=["neural_networks", "ml_training", "inference"],
        relative_speed=200.0,
        power_efficiency=0.95,
        availability=0.3,
    ),
    ComputeSubstrate.QUANTUM_SIM: SubstrateCapability(
        substrate=ComputeSubstrate.QUANTUM_SIM,
        min_problem_size=2,
        max_problem_size=30,
        specializations=["quantum_algorithms", "chemistry", "optimization"],
        relative_speed=0.1,
        power_efficiency=0.5,
        availability=1.0,
    ),
    ComputeSubstrate.QUANTUM_HW: SubstrateCapability(
        substrate=ComputeSubstrate.QUANTUM_HW,
        min_problem_size=2,
        max_problem_size=100,
        specializations=["quantum_algorithms", "chemistry", "cryptography"],
        relative_speed=0.01,
        power_efficiency=0.3,
        availability=0.1,
    ),
    ComputeSubstrate.DISTRIBUTED: SubstrateCapability(
        substrate=ComputeSubstrate.DISTRIBUTED,
        min_problem_size=100000,
        max_problem_size=10000000000,
        specializations=["big_data", "monte_carlo", "parallel_search"],
        relative_speed=1000.0,
        power_efficiency=0.4,
        availability=0.5,
    ),
    ComputeSubstrate.NEUROMORPHIC: SubstrateCapability(
        substrate=ComputeSubstrate.NEUROMORPHIC,
        min_problem_size=1000,
        max_problem_size=10000000,
        specializations=["spiking_networks", "event_driven", "edge_ai"],
        relative_speed=50.0,
        power_efficiency=0.99,
        availability=0.05,
    ),
}


def select_optimal_substrate(
    problem_size: int,
    task_type: str,
    required_availability: float = 0.8,
) -> ComputeSubstrate:
    """Select optimal compute substrate for a task.

    Args:
        problem_size: Size/scale of the problem
        task_type: Type of task being performed
        required_availability: Minimum availability requirement

    Returns:
        Optimal compute substrate
    """
    candidates = []

    for substrate, capability in SUBSTRATE_CAPABILITIES.items():
        # Check size compatibility
        if problem_size < capability.min_problem_size:
            continue
        if problem_size > capability.max_problem_size:
            continue

        # Check availability
        if capability.availability < required_availability:
            continue

        # Check specialization match
        specialization_score = 0
        if task_type in capability.specializations:
            specialization_score = 2.0
        elif any(spec in task_type for spec in capability.specializations):
            specialization_score = 1.0

        # Compute overall score
        score = (
            capability.relative_speed * 0.4
            + capability.power_efficiency * 0.2
            + capability.availability * 0.2
            + specialization_score * 0.2
        )

        candidates.append((substrate, score))

    if not candidates:
        # Fallback to CPU_SERIAL
        return ComputeSubstrate.CPU_SERIAL

    # Return substrate with highest score
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]
