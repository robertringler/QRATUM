"""XENON meta-kernel: Continuous learning runtime.

Main loop:
1. Generate hypothesis mechanisms
2. Simulate mechanisms (Gillespie SSA)
3. Rank by epistemic uncertainty
4. Select next experiment (max info gain)
5. Execute experiment (mock simulator Phase 1)
6. Update mechanism posteriors (Bayesian)
7. Prune low-evidence mechanisms
8. Check convergence (uncertainty < threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.mechanism import BioMechanism, MolecularState, Transition
from ..core.mechanism_graph import MechanismGraph
from ..learning.bayesian_updater import BayesianUpdater, ExperimentResult
from ..learning.mechanism_prior import MechanismPrior
from ..simulation.gillespie import GillespieSimulator


@dataclass
class Target:
    """A learning target for XENON.

    Attributes:
        name: Target identifier
        protein: Protein name
        objective: Learning objective (e.g., 'characterize', 'find_inhibitor')
        constraints: Additional constraints
    """

    name: str
    protein: str
    objective: str
    constraints: Optional[Dict[str, Any]] = None


class XENONRuntime:
    """XENON runtime: Continuous learning loop.

    Orchestrates mechanism generation, simulation, experimentation,
    and Bayesian updating for biological mechanism discovery.

    Attributes:
        max_mechanisms: Maximum number of mechanisms to maintain
        convergence_threshold: Entropy threshold for convergence
        mutation_rate: Rate of topology mutations
    """

    def __init__(
        self,
        max_mechanisms: int = 1000,
        convergence_threshold: float = 0.1,
        mutation_rate: float = 0.1,
    ):
        """Initialize XENON runtime.

        Args:
            max_mechanisms: Maximum mechanisms to maintain
            convergence_threshold: Entropy threshold for convergence
            mutation_rate: Topology mutation rate
        """
        self.max_mechanisms = max_mechanisms
        self.convergence_threshold = convergence_threshold
        self.mutation_rate = mutation_rate

        # Components
        self.bayesian_updater = BayesianUpdater()
        self.mechanism_prior = MechanismPrior()
        self.mechanism_graph = MechanismGraph()

        # State
        self.targets: List[Target] = []
        self.mechanisms: Dict[str, List[BioMechanism]] = {}
        self.iteration_count = 0
        self._rng = np.random.default_rng()

    def add_target(
        self,
        name: str,
        protein: str,
        objective: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a learning target.

        Args:
            name: Target identifier
            protein: Protein name
            objective: Learning objective
            constraints: Optional constraints
        """
        target = Target(name, protein, objective, constraints)
        self.targets.append(target)

        # Initialize mechanism pool for this target
        if name not in self.mechanisms:
            self.mechanisms[name] = []

    def run(
        self,
        max_iterations: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run XENON learning loop.

        Main algorithm:
        while not converged:
            1. Generate hypothesis mechanisms
            2. Simulate mechanisms
            3. Rank by epistemic uncertainty
            4. Select next experiment
            5. Execute experiment (mock)
            6. Update posteriors
            7. Prune low-evidence
            8. Check convergence

        Args:
            max_iterations: Maximum iterations
            seed: Random seed

        Returns:
            Summary of learning process
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        summary = {
            "iterations": 0,
            "converged": False,
            "final_entropy": 0.0,
            "mechanisms_discovered": 0,
        }

        for iteration in range(max_iterations):
            self.iteration_count = iteration

            # Process each target
            for target in self.targets:
                # 1. Generate hypothesis mechanisms
                if iteration == 0 or len(self.mechanisms[target.name]) < 5:
                    self._generate_hypothesis_mechanisms(target)

                # 2. Simulate mechanisms
                self._simulate_mechanisms(target)

                # 3. Rank by epistemic uncertainty
                ranked = self._rank_by_uncertainty(target)

                # 4. Select next experiment
                experiment = self._select_experiment(target, ranked)

                # 5. Execute experiment (mock)
                result = self._execute_experiment(target, experiment)

                # 6. Update posteriors
                self.mechanisms[target.name] = self.bayesian_updater.update_mechanisms(
                    self.mechanisms[target.name], result
                )

                # 7. Prune low-evidence mechanisms
                self.mechanisms[target.name] = self.bayesian_updater.prune_low_evidence(
                    self.mechanisms[target.name], threshold=1e-6
                )

                # Keep only top mechanisms if exceeding max
                if len(self.mechanisms[target.name]) > self.max_mechanisms:
                    self.mechanisms[target.name] = sorted(
                        self.mechanisms[target.name],
                        key=lambda m: m.posterior,
                        reverse=True,
                    )[: self.max_mechanisms]

                # 8. Check convergence
                evidence = self.bayesian_updater.get_evidence_summary(self.mechanisms[target.name])

                summary["final_entropy"] = evidence["entropy"]

                if evidence["entropy"] < self.convergence_threshold:
                    summary["converged"] = True
                    summary["iterations"] = iteration + 1
                    summary["mechanisms_discovered"] = len(self.mechanisms[target.name])
                    return summary

        summary["iterations"] = max_iterations
        summary["mechanisms_discovered"] = sum(len(mechs) for mechs in self.mechanisms.values())

        return summary

    def _generate_hypothesis_mechanisms(self, target: Target) -> None:
        """Generate hypothesis mechanisms for target.

        Phase 1: Generate simple template mechanisms
        Phase 2+: Use mechanism synthesis from literature/databases

        Args:
            target: Learning target
        """
        # Generate initial template mechanisms
        n_generate = min(10, self.max_mechanisms // len(self.targets))

        for i in range(n_generate):
            mech = self._create_template_mechanism(target, i)

            # Add mutations to existing mechanisms
            if self.mechanisms[target.name] and self._rng.random() < 0.5:
                parent = self._rng.choice(self.mechanisms[target.name])
                mech = self.mechanism_graph.mutate_topology(parent, self.mutation_rate, seed=None)
                mech.name = f"{target.name}_mutant_{self.iteration_count}_{i}"

            self.mechanisms[target.name].append(mech)

        # Initialize priors
        self.mechanisms[target.name] = self.mechanism_prior.initialize_mechanism_priors(
            self.mechanisms[target.name]
        )

    def _create_template_mechanism(self, target: Target, index: int) -> BioMechanism:
        """Create a template mechanism for the target.

        Args:
            target: Learning target
            index: Template index

        Returns:
            Template mechanism
        """
        mech = BioMechanism(name=f"{target.name}_template_{index}")

        # Create simple two-state mechanism
        state1 = MolecularState(
            name=f"{target.protein}_inactive",
            molecule=target.protein,
            properties={"active": False},
            concentration=100.0,
            free_energy=-10.0,
        )

        state2 = MolecularState(
            name=f"{target.protein}_active",
            molecule=target.protein,
            properties={"active": True},
            concentration=10.0,
            free_energy=-12.0,
        )

        mech.add_state(state1)
        mech.add_state(state2)

        # Add transition with random rate constant
        rate = 10 ** self._rng.uniform(-3, 3)  # 10^-3 to 10^3 s^-1

        transition = Transition(
            source=state1.name,
            target=state2.name,
            rate_constant=rate,
            activation_energy=15.0,
            reversible=True,
            reverse_rate=rate / 10.0,
        )

        mech.add_transition(transition)

        return mech

    def _simulate_mechanisms(self, target: Target) -> None:
        """Simulate all mechanisms for target.

        Args:
            target: Learning target
        """
        for mechanism in self.mechanisms[target.name]:
            # Run short simulation to check viability
            simulator = GillespieSimulator(mechanism, volume=1e-15)

            # Initial state: equal concentrations
            initial_state = dict.fromkeys(mechanism._states, 50.0)

            try:
                # Run brief simulation (0.1 seconds)
                times, trajectories = simulator.run(
                    t_max=0.1,
                    initial_state=initial_state,
                    seed=None,
                    record_interval=0.01,
                )

                # Update state concentrations from final simulation state
                if times and len(times) > 0:
                    for state_name in mechanism._states:
                        if state_name in trajectories and len(trajectories[state_name]) > 0:
                            final_conc = trajectories[state_name][-1]
                            mechanism._states[state_name].concentration = final_conc

            except Exception:
                # If simulation fails, mark with low posterior
                mechanism.posterior *= 0.1

    def _rank_by_uncertainty(self, target: Target) -> List[BioMechanism]:
        """Rank mechanisms by epistemic uncertainty.

        Higher uncertainty → more informative for next experiment

        Args:
            target: Learning target

        Returns:
            Sorted list of mechanisms (high uncertainty first)
        """
        # Compute uncertainty as entropy contribution
        mechanisms = self.mechanisms[target.name]

        if not mechanisms:
            return []

        # Sort by posterior (highest first for now)
        # In Phase 2+, use more sophisticated uncertainty measures
        ranked = sorted(mechanisms, key=lambda m: m.posterior, reverse=True)

        return ranked

    def _select_experiment(
        self, target: Target, ranked_mechanisms: List[BioMechanism]
    ) -> Dict[str, Any]:
        """Select next experiment to maximize information gain.

        Phase 1: Random experiment selection
        Phase 2+: Use Bayesian experimental design

        Args:
            target: Learning target
            ranked_mechanisms: Ranked mechanisms

        Returns:
            Experiment specification
        """
        # Mock experiment selection
        experiment_types = ["concentration", "kinetics", "perturbation"]
        exp_type = self._rng.choice(experiment_types)

        return {
            "type": exp_type,
            "target": target.name,
            "iteration": self.iteration_count,
        }

    def _execute_experiment(self, target: Target, experiment: Dict[str, Any]) -> ExperimentResult:
        """Execute experiment (mock for Phase 1).

        Phase 1: Generate synthetic data
        Phase 2: Interface with cloud lab automation

        Args:
            target: Learning target
            experiment: Experiment specification

        Returns:
            Experiment result
        """
        exp_type = experiment["type"]

        # Generate mock observations
        if exp_type == "concentration":
            observations = {
                f"{target.protein}_inactive": self._rng.normal(80.0, 10.0),
                f"{target.protein}_active": self._rng.normal(20.0, 5.0),
            }
            uncertainties = {
                f"{target.protein}_inactive": 10.0,
                f"{target.protein}_active": 5.0,
            }

        elif exp_type == "kinetics":
            observations = {
                f"{target.protein}_inactive->{target.protein}_active": 10
                ** self._rng.uniform(-2, 2)
            }
            uncertainties = {}

        else:  # perturbation
            observations = {"response": self._rng.normal(0.5, 0.1)}
            uncertainties = {"response": 0.1}

        return ExperimentResult(
            experiment_type=exp_type,
            observations=observations,
            uncertainties=uncertainties,
            conditions={"temperature": 310.0},
        )

    def get_mechanisms(
        self, min_evidence: float = 0.5, target_name: Optional[str] = None
    ) -> List[BioMechanism]:
        """Get high-confidence mechanisms.

        Args:
            min_evidence: Minimum posterior probability
            target_name: Optional target name filter

        Returns:
            List of high-confidence mechanisms
        """
        mechanisms = []

        if target_name:
            targets = [t for t in self.targets if t.name == target_name]
        else:
            targets = self.targets

        for target in targets:
            mechs = self.mechanisms.get(target.name, [])
            high_conf = [m for m in mechs if m.posterior >= min_evidence]
            mechanisms.extend(high_conf)

        return sorted(mechanisms, key=lambda m: m.posterior, reverse=True)

    def get_summary(self) -> Dict[str, Any]:
        """Get runtime summary statistics.

        Returns:
            Summary dictionary
        """
        total_mechanisms = sum(len(mechs) for mechs in self.mechanisms.values())

        evidence_summaries = {}
        for target in self.targets:
            mechs = self.mechanisms.get(target.name, [])
            evidence_summaries[target.name] = self.bayesian_updater.get_evidence_summary(mechs)

        return {
            "iterations": self.iteration_count,
            "total_mechanisms": total_mechanisms,
            "targets": len(self.targets),
            "evidence": evidence_summaries,
        }
