"""
QRATUM Compute Substrate Selector

Maps tasks to optimal compute substrates based on workload characteristics.
Supports GPU (GB200/MI300X), Cerebras, QPU, IPU, and CPU substrates.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class ComputeSubstrate(Enum):
    """
    Available compute substrates for QRATUM execution.

    Each substrate has specific strengths:
    - GPU: High throughput parallel processing (tensor operations)
    - CEREBRAS: Massive parallelism (wafer-scale integration)
    - QPU: Quantum probabilistic computing (VQE, QAOA)
    - IPU: Streaming/graph-based computation (neural networks)
    - CPU: Deterministic verification and control flow
    """

    GPU_GB200 = "gpu_gb200"  # NVIDIA GB200 Grace-Blackwell
    GPU_MI300X = "gpu_mi300x"  # AMD MI300X
    CEREBRAS = "cerebras"  # Wafer-scale engine
    QPU = "qpu"  # Quantum processing unit
    IPU = "ipu"  # Intelligence processing unit (Graphcore)
    CPU = "cpu"  # Standard CPU (verification/control)


@dataclass
class SubstrateCapability:
    """
    Capability profile for a compute substrate.

    Defines what types of workloads are well-suited for each substrate.
    """

    substrate: ComputeSubstrate
    strengths: List[str]
    typical_tasks: List[str]
    throughput_rating: int  # 1-10 scale
    precision_rating: int  # 1-10 scale
    determinism_rating: int  # 1-10 scale


# Substrate capability profiles
SUBSTRATE_PROFILES = {
    ComputeSubstrate.GPU_GB200: SubstrateCapability(
        substrate=ComputeSubstrate.GPU_GB200,
        strengths=[
            "High throughput tensor operations",
            "Large matrix multiplication",
            "Neural network training/inference",
            "Monte Carlo simulations",
        ],
        typical_tasks=[
            "molecular_dynamics",
            "neural_simulation",
            "climate_modeling",
            "risk_simulation",
        ],
        throughput_rating=10,
        precision_rating=8,
        determinism_rating=7,
    ),
    ComputeSubstrate.GPU_MI300X: SubstrateCapability(
        substrate=ComputeSubstrate.GPU_MI300X,
        strengths=[
            "High memory bandwidth",
            "Large model training",
            "Scientific computing",
        ],
        typical_tasks=[
            "protein_folding",
            "genomic_analysis",
            "fluid_dynamics",
        ],
        throughput_rating=9,
        precision_rating=8,
        determinism_rating=7,
    ),
    ComputeSubstrate.CEREBRAS: SubstrateCapability(
        substrate=ComputeSubstrate.CEREBRAS,
        strengths=[
            "Massive parallelism",
            "Wafer-scale integration",
            "Ultra-low latency",
        ],
        typical_tasks=[
            "swarm_simulation",
            "network_optimization",
            "circuit_simulation",
        ],
        throughput_rating=10,
        precision_rating=7,
        determinism_rating=6,
    ),
    ComputeSubstrate.QPU: SubstrateCapability(
        substrate=ComputeSubstrate.QPU,
        strengths=[
            "Quantum probabilistic computing",
            "Optimization problems",
            "Molecular simulation",
        ],
        typical_tasks=[
            "vqe_calculation",
            "qaoa_optimization",
            "quantum_chemistry",
        ],
        throughput_rating=3,
        precision_rating=6,
        determinism_rating=4,
    ),
    ComputeSubstrate.IPU: SubstrateCapability(
        substrate=ComputeSubstrate.IPU,
        strengths=[
            "Streaming computation",
            "Graph-based processing",
            "Fine-grained parallelism",
        ],
        typical_tasks=[
            "graph_neural_networks",
            "signal_processing",
            "real_time_inference",
        ],
        throughput_rating=8,
        precision_rating=7,
        determinism_rating=7,
    ),
    ComputeSubstrate.CPU: SubstrateCapability(
        substrate=ComputeSubstrate.CPU,
        strengths=[
            "Full determinism",
            "Complex control flow",
            "Verification and validation",
        ],
        typical_tasks=[
            "contract_analysis",
            "policy_simulation",
            "safety_verification",
        ],
        throughput_rating=3,
        precision_rating=10,
        determinism_rating=10,
    ),
}


class SubstrateSelector:
    """
    Intelligent substrate selector for QRATUM tasks.

    Analyzes task characteristics and selects the optimal compute
    substrate(s) for execution. Supports multi-substrate execution
    where tasks can benefit from heterogeneous computing.
    """

    def __init__(self):
        """Initialize substrate selector"""
        self.profiles = SUBSTRATE_PROFILES

    def select_substrate(
        self, task_name: str, task_characteristics: Dict[str, Any]
    ) -> List[ComputeSubstrate]:
        """
        Select optimal substrate(s) for a task.

        Args:
            task_name: Name of the task
            task_characteristics: Dictionary with task properties:
                - requires_determinism: bool
                - requires_high_throughput: bool
                - requires_quantum: bool
                - workload_type: str (tensor, graph, simulation, etc.)

        Returns:
            List of recommended substrates (ordered by preference)
        """
        requires_determinism = task_characteristics.get("requires_determinism", False)
        requires_high_throughput = task_characteristics.get("requires_high_throughput", False)
        requires_quantum = task_characteristics.get("requires_quantum", False)
        workload_type = task_characteristics.get("workload_type", "general")

        candidates = []

        # Quantum tasks must use QPU
        if requires_quantum:
            candidates.append(ComputeSubstrate.QPU)
            return candidates

        # Deterministic tasks prefer CPU
        if requires_determinism:
            candidates.append(ComputeSubstrate.CPU)
            if not requires_high_throughput:
                return candidates

        # High throughput tasks
        if requires_high_throughput:
            if workload_type == "tensor":
                candidates.extend([ComputeSubstrate.GPU_GB200, ComputeSubstrate.GPU_MI300X])
            elif workload_type == "graph":
                candidates.append(ComputeSubstrate.IPU)
            elif workload_type == "massive_parallel":
                candidates.append(ComputeSubstrate.CEREBRAS)
            else:
                candidates.append(ComputeSubstrate.GPU_GB200)

        # Default fallback
        if not candidates:
            candidates.append(ComputeSubstrate.CPU)

        return candidates

    def get_substrate_info(self, substrate: ComputeSubstrate) -> SubstrateCapability:
        """Get capability profile for a substrate"""
        return self.profiles[substrate]

    def recommend_for_vertical(self, vertical_name: str) -> Dict[str, List[ComputeSubstrate]]:
        """
        Get substrate recommendations for common tasks in a vertical.

        Args:
            vertical_name: Name of the vertical module

        Returns:
            Dictionary mapping task types to recommended substrates
        """
        recommendations = {
            "JURIS": {
                "contract_analysis": [ComputeSubstrate.CPU],
                "litigation_prediction": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CPU],
            },
            "VITRA": {
                "genomic_analysis": [ComputeSubstrate.GPU_MI300X],
                "protein_folding": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CEREBRAS],
                "drug_docking": [ComputeSubstrate.GPU_GB200],
            },
            "ECORA": {
                "climate_modeling": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CEREBRAS],
                "energy_optimization": [ComputeSubstrate.QPU, ComputeSubstrate.CPU],
            },
            "CAPRA": {
                "derivatives_pricing": [ComputeSubstrate.GPU_GB200],
                "risk_simulation": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CEREBRAS],
                "portfolio_optimization": [ComputeSubstrate.QPU, ComputeSubstrate.GPU_GB200],
            },
            "SENTRA": {
                "trajectory_simulation": [ComputeSubstrate.GPU_GB200],
                "radar_analysis": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.IPU],
            },
            "NEURA": {
                "neural_simulation": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CEREBRAS],
                "bci_decoding": [ComputeSubstrate.IPU, ComputeSubstrate.GPU_GB200],
            },
            "FLUXA": {
                "route_optimization": [ComputeSubstrate.QPU, ComputeSubstrate.CPU],
                "demand_forecasting": [ComputeSubstrate.GPU_GB200],
            },
            "CHRONA": {
                "circuit_simulation": [ComputeSubstrate.CEREBRAS, ComputeSubstrate.GPU_GB200],
                "timing_analysis": [ComputeSubstrate.CPU],
            },
            "GEONA": {
                "satellite_analysis": [ComputeSubstrate.GPU_GB200],
                "terrain_modeling": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CEREBRAS],
            },
            "FUSIA": {
                "plasma_simulation": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CEREBRAS],
                "neutronics": [ComputeSubstrate.GPU_GB200],
            },
            "STRATA": {
                "economic_modeling": [ComputeSubstrate.CPU, ComputeSubstrate.GPU_GB200],
                "policy_simulation": [ComputeSubstrate.CPU],
            },
            "VEXOR": {
                "threat_detection": [ComputeSubstrate.IPU, ComputeSubstrate.GPU_GB200],
                "malware_analysis": [ComputeSubstrate.CPU],
            },
            "COHORA": {
                "swarm_coordination": [ComputeSubstrate.CEREBRAS, ComputeSubstrate.IPU],
                "path_planning": [ComputeSubstrate.QPU, ComputeSubstrate.GPU_GB200],
            },
            "ORBIA": {
                "orbit_propagation": [ComputeSubstrate.GPU_GB200],
                "collision_avoidance": [ComputeSubstrate.GPU_GB200, ComputeSubstrate.CPU],
            },
        }

        return recommendations.get(vertical_name, {})
