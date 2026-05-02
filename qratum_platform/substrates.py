"""Compute substrate mappings for vertical modules.

Defines optimal hardware for each vertical module's task types.
"""

from qratum_platform.core import ComputeSubstrate, VerticalModule

# Optimal substrate mappings for vertical modules and task types
VERTICAL_SUBSTRATE_MAPPINGS = {
    # JURIS - Legal AI
    VerticalModule.JURIS: {
        "legal_reasoning": ComputeSubstrate.CPU,  # Logic-heavy, not compute-bound
        "contract_analysis": ComputeSubstrate.GB200,  # Large language models
        "litigation_prediction": ComputeSubstrate.GB200,  # ML inference
        "compliance_checking": ComputeSubstrate.CPU,  # Rule-based
        "default": ComputeSubstrate.CPU,
    },
    # VITRA - Bioinformatics
    VerticalModule.VITRA: {
        "sequence_analysis": ComputeSubstrate.MI300X,  # Memory-intensive
        "protein_structure": ComputeSubstrate.CEREBRAS,  # ML-based prediction
        "drug_screening": ComputeSubstrate.GB200,  # Parallel screening
        "molecular_dynamics": ComputeSubstrate.MI300X,  # High-bandwidth memory
        "clinical_trial": ComputeSubstrate.CPU,  # Statistical analysis
        "pharmacokinetics": ComputeSubstrate.GB200,  # Simulation
        "default": ComputeSubstrate.MI300X,
    },
    # ECORA - Climate & Energy
    VerticalModule.ECORA: {
        "climate_projection": ComputeSubstrate.CEREBRAS,  # Large-scale simulation
        "grid_optimization": ComputeSubstrate.GB200,  # Optimization algorithms
        "carbon_analysis": ComputeSubstrate.CPU,  # Accounting
        "weather_prediction": ComputeSubstrate.CEREBRAS,  # Weather models
        "renewable_assessment": ComputeSubstrate.GB200,  # Data analysis
        "default": ComputeSubstrate.CEREBRAS,
    },
    # CAPRA - Financial Risk
    VerticalModule.CAPRA: {
        "option_pricing": ComputeSubstrate.GB200,  # Black-Scholes, parallel
        "monte_carlo": ComputeSubstrate.GB200,  # Massive parallelism
        "var_calculation": ComputeSubstrate.GB200,  # Monte Carlo-based
        "portfolio_optimization": ComputeSubstrate.QPU,  # Quantum annealing
        "credit_risk": ComputeSubstrate.GB200,  # ML models
        "stress_testing": ComputeSubstrate.GB200,  # Scenario simulation
        "default": ComputeSubstrate.GB200,
    },
    # SENTRA - Aerospace & Defense
    VerticalModule.SENTRA: {
        "trajectory_simulation": ComputeSubstrate.GB200,  # Physics simulation
        "radar_analysis": ComputeSubstrate.MI300X,  # Signal processing
        "orbit_propagation": ComputeSubstrate.GB200,  # Orbital mechanics
        "aerodynamic_analysis": ComputeSubstrate.CEREBRAS,  # CFD
        "mission_planning": ComputeSubstrate.QPU,  # Optimization
        "default": ComputeSubstrate.GB200,
    },
    # NEURA - Neuroscience & BCI
    VerticalModule.NEURA: {
        "neural_simulation": ComputeSubstrate.CEREBRAS,  # Spiking neural networks
        "eeg_analysis": ComputeSubstrate.MI300X,  # Signal processing
        "connectivity_mapping": ComputeSubstrate.GB200,  # Graph analysis
        "bci_processing": ComputeSubstrate.IPU,  # Low-latency inference
        "cognitive_modeling": ComputeSubstrate.CEREBRAS,  # Large-scale simulation
        "default": ComputeSubstrate.CEREBRAS,
    },
    # FLUXA - Supply Chain
    VerticalModule.FLUXA: {
        "route_optimization": ComputeSubstrate.QPU,  # TSP/VRP optimization
        "demand_forecasting": ComputeSubstrate.GB200,  # Time series ML
        "inventory_optimization": ComputeSubstrate.GB200,  # Optimization
        "resilience_analysis": ComputeSubstrate.GB200,  # Network simulation
        "logistics_simulation": ComputeSubstrate.GB200,  # Discrete event simulation
        "default": ComputeSubstrate.GB200,
    },
    # SPECTRA - Spectrum Management (not implemented yet)
    VerticalModule.SPECTRA: {
        "default": ComputeSubstrate.MI300X,
    },
    # AEGIS - Cybersecurity (not implemented yet)
    VerticalModule.AEGIS: {
        "default": ComputeSubstrate.IPU,
    },
    # LOGOS - Education (not implemented yet)
    VerticalModule.LOGOS: {
        "default": ComputeSubstrate.GB200,
    },
    # SYNTHOS - Materials Science (not implemented yet)
    VerticalModule.SYNTHOS: {
        "default": ComputeSubstrate.CEREBRAS,
    },
    # TERAGON - Geospatial (not implemented yet)
    VerticalModule.TERAGON: {
        "default": ComputeSubstrate.MI300X,
    },
    # HELIX - Genomic Medicine (not implemented yet)
    VerticalModule.HELIX: {
        "default": ComputeSubstrate.MI300X,
    },
    # NEXUS - Cross-domain Intelligence (not implemented yet)
    VerticalModule.NEXUS: {
        "default": ComputeSubstrate.GB200,
    },
}


def get_optimal_substrate(vertical: VerticalModule, operation: str) -> ComputeSubstrate:
    """Get optimal compute substrate for a vertical module operation.

    Args:
        vertical: Vertical module
        operation: Operation name

    Returns:
        Optimal compute substrate
    """
    if vertical not in VERTICAL_SUBSTRATE_MAPPINGS:
        return ComputeSubstrate.CPU

    module_mappings = VERTICAL_SUBSTRATE_MAPPINGS[vertical]
    return module_mappings.get(operation, module_mappings.get("default", ComputeSubstrate.CPU))
