"""Public Python API for XENON.

Convenience functions for creating mechanisms, running simulations,
and querying results.
"""

from __future__ import annotations

from typing import Any, Optional

from .core.mechanism import BioMechanism, MolecularState, Transition
from .core.mechanism_graph import MechanismGraph
from .learning.bayesian_updater import BayesianUpdater, ExperimentResult
from .learning.mechanism_prior import MechanismPrior
from .runtime.xenon_kernel import XENONRuntime
from .simulation.gillespie import GillespieSimulator
from .simulation.langevin import LangevinSimulator


def create_mechanism(
    name: str,
    states: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> BioMechanism:
    """Create a mechanism from dictionaries.

    Args:
        name: Mechanism name
        states: List of state dictionaries
        transitions: List of transition dictionaries

    Returns:
        BioMechanism object

    Example:
        >>> states = [
        ...     {"name": "S1", "molecule": "Protein", "free_energy": -10.0},
        ...     {"name": "S2", "molecule": "Protein", "free_energy": -12.0},
        ... ]
        >>> transitions = [
        ...     {"source": "S1", "target": "S2", "rate_constant": 1.5e-3},
        ... ]
        >>> mech = create_mechanism("test", states, transitions)
    """
    mechanism = BioMechanism(name=name)

    # Add states
    for state_dict in states:
        state = MolecularState(**state_dict)
        mechanism.add_state(state)

    # Add transitions
    for trans_dict in transitions:
        transition = Transition(**trans_dict)
        mechanism.add_transition(transition)

    return mechanism


def simulate_mechanism(
    mechanism: BioMechanism,
    t_max: float,
    initial_state: dict[str, float],
    method: str = "gillespie",
    **kwargs,
) -> tuple[list[float], dict[str, list[float]]]:
    """Simulate a mechanism.

    Args:
        mechanism: Mechanism to simulate
        t_max: Maximum simulation time
        initial_state: Initial concentrations
        method: Simulation method ("gillespie" or "langevin")
        **kwargs: Additional simulator arguments (volume, seed, dt, etc.)

    Returns:
        Tuple of (times, trajectories)

    Example:
        >>> times, traj = simulate_mechanism(
        ...     mechanism, t_max=1.0, initial_state={"S1": 100.0, "S2": 0.0}
        ... )
    """
    if method == "gillespie":
        # Extract run() parameters
        seed = kwargs.pop("seed", None)
        record_interval = kwargs.pop("record_interval", None)
        # Remaining kwargs go to simulator init
        simulator = GillespieSimulator(mechanism, **kwargs)
        return simulator.run(t_max, initial_state, seed=seed, record_interval=record_interval)
    elif method == "langevin":
        dt = kwargs.pop("dt", 0.01)
        seed = kwargs.pop("seed", None)
        # Remaining kwargs go to simulator init
        simulator = LangevinSimulator(mechanism, **kwargs)
        return simulator.run(t_max, dt, initial_state, seed=seed)
    else:
        raise ValueError(f"Unknown simulation method: {method}")


def run_xenon(
    targets: list[dict[str, str]],
    max_iterations: int = 100,
    **kwargs,
) -> dict[str, Any]:
    """Run XENON learning loop.

    Args:
        targets: List of target dictionaries with keys: name, protein, objective
        max_iterations: Maximum iterations
        **kwargs: Additional runtime arguments

    Returns:
        Summary dictionary

    Example:
        >>> results = run_xenon(
        ...     targets=[{"name": "test", "protein": "EGFR", "objective": "characterize"}],
        ...     max_iterations=10
        ... )
    """
    runtime = XENONRuntime(**kwargs)

    for target_dict in targets:
        runtime.add_target(**target_dict)

    summary = runtime.run(max_iterations)

    return summary


def validate_mechanism(
    mechanism: BioMechanism,
    temperature: float = 310.0,
) -> dict[str, Any]:
    """Validate mechanism constraints.

    Args:
        mechanism: Mechanism to validate
        temperature: Temperature in Kelvin

    Returns:
        Validation results dictionary

    Example:
        >>> validation = validate_mechanism(mechanism)
        >>> if validation["thermodynamically_feasible"]:
        ...     print("Mechanism is valid")
    """
    thermodynamic_feasible = mechanism.is_thermodynamically_feasible(temperature)
    conservation_valid, violations = mechanism.validate_conservation_laws()

    return {
        "thermodynamically_feasible": thermodynamic_feasible,
        "conservation_laws_valid": conservation_valid,
        "violations": violations,
        "n_states": len(mechanism._states),
        "n_transitions": len(mechanism._transitions),
        "mechanism_hash": mechanism.compute_mechanism_hash(),
    }


def compute_mechanism_prior(mechanism: BioMechanism, **kwargs) -> float:
    """Compute prior probability for a mechanism.

    Args:
        mechanism: Mechanism
        **kwargs: MechanismPrior arguments

    Returns:
        Prior probability

    Example:
        >>> prior = compute_mechanism_prior(mechanism)
    """
    prior_calculator = MechanismPrior(**kwargs)
    return prior_calculator.compute_prior(mechanism)


def update_mechanism_posterior(
    mechanisms: list[BioMechanism],
    experiment_result: ExperimentResult,
    **kwargs,
) -> list[BioMechanism]:
    """Update mechanism posteriors from experiment.

    Args:
        mechanisms: List of mechanisms
        experiment_result: Experimental result
        **kwargs: BayesianUpdater arguments

    Returns:
        Updated mechanisms

    Example:
        >>> result = ExperimentResult("concentration", {"S1": 80.0, "S2": 20.0})
        >>> updated = update_mechanism_posterior(mechanisms, result)
    """
    updater = BayesianUpdater(**kwargs)
    return updater.update_mechanisms(mechanisms, experiment_result)


def mutate_mechanism(
    mechanism: BioMechanism,
    mutation_rate: float = 0.1,
    seed: Optional[int] = None,
) -> BioMechanism:
    """Mutate mechanism topology.

    Args:
        mechanism: Parent mechanism
        mutation_rate: Mutation rate
        seed: Random seed

    Returns:
        Mutated mechanism

    Example:
        >>> mutant = mutate_mechanism(mechanism, mutation_rate=0.2)
    """
    return MechanismGraph.mutate_topology(mechanism, mutation_rate, seed)


def recombine_mechanisms(
    mech1: BioMechanism,
    mech2: BioMechanism,
    name: str,
) -> BioMechanism:
    """Recombine two mechanisms.

    Args:
        mech1: First parent
        mech2: Second parent
        name: Name for child mechanism

    Returns:
        Recombined mechanism

    Example:
        >>> child = recombine_mechanisms(parent1, parent2, "child")
    """
    return MechanismGraph.recombine_mechanisms(mech1, mech2, name)


__all__ = [
    "create_mechanism",
    "simulate_mechanism",
    "run_xenon",
    "validate_mechanism",
    "compute_mechanism_prior",
    "update_mechanism_posterior",
    "mutate_mechanism",
    "recombine_mechanisms",
]
