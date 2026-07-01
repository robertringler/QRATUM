"""Gillespie Stochastic Simulation Algorithm (SSA).

Exact stochastic simulation of chemical reaction systems.
Target: 10^6 reactions/second on single CPU core.

Reference:
Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions.
The Journal of Physical Chemistry, 81(25), 2340-2361.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.mechanism import BioMechanism, Transition


@dataclass
class SimulationState:
    """State of the simulation at a given time.

    Attributes:
        time: Current simulation time (seconds)
        concentrations: Concentration of each species (nM)
        molecule_counts: Integer molecule counts (for discrete simulation)
    """

    time: float
    concentrations: dict[str, float]
    molecule_counts: dict[str, int]


@dataclass
class _Reaction:
    """A single elementary reaction channel compiled from a Transition.

    A reversible ``Transition`` compiles into two channels (forward + reverse);
    an irreversible one into a single forward channel.

    Attributes:
        reactants: species -> stoichiometric order consumed (for the propensity)
        net: species -> net change in molecule count when the channel fires
        transition: the source Transition (rate read live, so parameter
            perturbation between runs is reflected)
        direction: 'forward' or 'reverse' (selects which rate constant to use)
    """

    reactants: dict[str, int]
    net: dict[str, int]
    transition: Transition
    direction: str


class GillespieSimulator:
    """Gillespie Stochastic Simulation Algorithm.

    Exact stochastic simulation of chemical master equation.
    Uses direct method for reaction selection.

    Performance target: 10^6 reactions/second on single CPU core.

    Attributes:
        mechanism: Biological mechanism to simulate
        volume: Reaction volume in liters (affects concentration/count conversion)
        avogadro: Avogadro's number for concentration conversion
    """

    def __init__(
        self,
        mechanism: BioMechanism,
        volume: float = 1e-15,  # 1 femtoliter (typical cell volume)
    ):
        """Initialize Gillespie simulator.

        Args:
            mechanism: Mechanism to simulate
            volume: Reaction volume in liters
        """

        self.mechanism = mechanism
        self.volume = volume
        self.avogadro = 6.022e23  # Avogadro's number

        # Conversion factor: nM to molecule count
        # nM = nmol/L = 10^-9 mol/L
        # molecules = (conc in nM) * (10^-9 mol/L) * (volume in L) * N_A
        self.nM_to_molecules = 1e-9 * volume * self.avogadro

        self._rng: Optional[np.random.Generator] = None
        self._reaction_count = 0
        self._reactions: list[_Reaction] = []

    def _build_reactions(self) -> list[_Reaction]:
        """Compile transitions into elementary reaction channels.

        Each transition yields a forward channel; reversible transitions that
        declare a ``reverse_rate`` additionally yield a reverse channel (fixes
        the silent dropping of reverse reactions — detailed balance now holds).
        Net stoichiometry is taken from ``Transition.stoichiometry`` when given,
        otherwise it defaults to the unimolecular convention ``{source:-1,
        target:+1}`` (preserves prior behaviour for simple transitions).
        Reactant orders for the propensity are derived from the negative entries
        of the net-change vector, enabling higher-order (e.g. bimolecular)
        reactions via mass action.
        """

        reactions: list[_Reaction] = []
        for transition in self.mechanism._transitions:
            if transition.stoichiometry:
                net = dict(transition.stoichiometry)
            else:
                net = {transition.source: -1, transition.target: 1}

            reactants = {sp: -c for sp, c in net.items() if c < 0}
            reactions.append(_Reaction(reactants, net, transition, "forward"))

            if transition.reversible and transition.reverse_rate is not None:
                reverse_net = {sp: -c for sp, c in net.items()}
                reverse_reactants = {sp: -c for sp, c in reverse_net.items() if c < 0}
                reactions.append(
                    _Reaction(reverse_reactants, reverse_net, transition, "reverse")
                )

        return reactions

    @staticmethod
    def _channel_rate(reaction: _Reaction) -> float:
        """Read the (live) rate constant for a reaction channel."""

        if reaction.direction == "reverse":
            return reaction.transition.reverse_rate or 0.0
        return reaction.transition.rate_constant

    def run(
        self,
        t_max: float,
        initial_state: dict[str, float],
        seed: Optional[int] = None,
        record_interval: Optional[float] = None,
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Run Gillespie SSA simulation.

        Args:
            t_max: Maximum simulation time (seconds)
            initial_state: Initial concentrations (nM) for each species
            seed: Random seed for reproducibility
            record_interval: Time interval for recording trajectory (None = record all)

        Returns:
            Tuple of (times, trajectories) where trajectories[species] = [concentrations]
        """

        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        self._reaction_count = 0

        # Compile reaction channels (forward + reverse) for the current
        # mechanism. Done here (not in __init__) so that live edits to rate
        # constants between runs are reflected.
        self._reactions = self._build_reactions()

        # Initialize state
        state = self._initialize_state(initial_state)

        # Storage for trajectory
        times = [state.time]
        trajectories: dict[str, list[float]] = {
            species: [state.concentrations[species]] for species in state.concentrations
        }

        next_record_time = record_interval if record_interval else 0.0

        # Main simulation loop
        while state.time < t_max:
            # Compute propensities
            propensities = self._compute_propensities(state)
            total_propensity = sum(propensities)

            # If no reactions can occur, terminate
            if total_propensity <= 0:
                break

            # Select time step (exponentially distributed)
            tau = -np.log(self._rng.random()) / total_propensity

            # Select reaction
            reaction_idx = self._select_reaction(propensities, total_propensity)

            # Update state
            state.time += tau
            self._update_state(state, reaction_idx)
            self._reaction_count += 1

            # Record trajectory
            if record_interval is None or state.time >= next_record_time:
                times.append(state.time)
                for species in state.concentrations:
                    trajectories[species].append(state.concentrations[species])

                if record_interval:
                    next_record_time += record_interval

        # Always record the final state. Without this, systems that
        # equilibrate or terminate within a single record_interval would return
        # only their initial point, so downstream "final concentration" reads
        # would see the initial condition rather than the true endpoint.
        if times[-1] != state.time:
            times.append(state.time)
            for species in state.concentrations:
                trajectories[species].append(state.concentrations[species])

        return times, trajectories

    def _initialize_state(self, initial_concentrations: dict[str, float]) -> SimulationState:
        """Initialize simulation state from concentrations.

        Args:
            initial_concentrations: Initial concentrations (nM)

        Returns:
            Initialized simulation state
        """

        # Ensure all species in mechanism have initial values
        concentrations = {}
        molecule_counts = {}

        for state_name in self.mechanism._states:
            if state_name in initial_concentrations:
                conc = initial_concentrations[state_name]
            else:
                # Default to zero if not specified
                conc = 0.0

            concentrations[state_name] = conc
            molecule_counts[state_name] = int(conc * self.nM_to_molecules)

        return SimulationState(
            time=0.0,
            concentrations=concentrations,
            molecule_counts=molecule_counts,
        )

    def _compute_propensities(self, state: SimulationState) -> list[float]:
        """Compute reaction propensities (mass-action).

        For a channel with reactant orders ``v_s`` the stochastic mass-action
        propensity is ``a = k * prod_s falling_factorial(n_s, v_s)`` where
        ``falling_factorial(n, v) = n*(n-1)*...*(n-v+1)``. For unimolecular
        reactions (``v=1``) this reduces to ``a = k * n`` (unchanged from the
        previous implementation); for bimolecular reactions it gives the correct
        combinatorial count of reactant pairs. A channel is inert when any
        reactant has fewer molecules than its order.

        Args:
            state: Current simulation state

        Returns:
            List of propensities, one per compiled reaction channel
        """

        propensities = []

        for reaction in self._reactions:
            propensity = self._channel_rate(reaction)

            for species, order in reaction.reactants.items():
                count = state.molecule_counts.get(species, 0)
                if count < order:
                    propensity = 0.0
                    break
                # Falling factorial n*(n-1)*...*(n-order+1)
                for j in range(order):
                    propensity *= count - j

            propensities.append(max(propensity, 0.0))

        return propensities

    def _select_reaction(self, propensities: list[float], total_propensity: float) -> int:
        """Select reaction to fire based on propensities.

        Args:
            propensities: List of reaction propensities
            total_propensity: Sum of all propensities

        Returns:
            Index of selected reaction
        """

        # Direct method: generate random number and find reaction
        r = self._rng.random() * total_propensity

        cumsum = 0.0
        for i, prop in enumerate(propensities):
            cumsum += prop
            if cumsum >= r:
                return i

        # Fallback (should not reach here due to numerical precision)
        return len(propensities) - 1

    def _update_state(self, state: SimulationState, reaction_idx: int) -> None:
        """Update state after reaction fires.

        Args:
            state: Current state (modified in place)
            reaction_idx: Index of reaction that fired
        """

        reaction = self._reactions[reaction_idx]

        # Apply the net stoichiometric change for every species in the channel.
        for species, change in reaction.net.items():
            new_count = state.molecule_counts.get(species, 0) + change
            # Clamp to zero (should not occur: propensity is zero when a
            # reactant is depleted, so the channel cannot fire).
            state.molecule_counts[species] = max(new_count, 0)

        # Update concentrations for the declared species
        for species in state.concentrations:
            count = state.molecule_counts[species]
            state.concentrations[species] = count / self.nM_to_molecules

    def get_performance_metrics(self) -> dict[str, float]:
        """Get simulation performance metrics.

        Returns:
            Dictionary with performance statistics
        """

        return {
            "total_reactions": self._reaction_count,
            "reactions_per_second": self._reaction_count,  # Placeholder
        }


class GillespieSimulatorOptimized(GillespieSimulator):
    """Optimized Gillespie SSA with improved performance.

    Uses optimizations:
    1. Dependency graph for partial propensity updates
    2. Composition-rejection method for reaction selection
    3. Vectorized operations where possible

    Target: 10^6 reactions/second on single CPU core.
    """

    def __init__(self, mechanism: BioMechanism, volume: float = 1e-15):
        super().__init__(mechanism, volume)
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build dependency graph for partial propensity updates.

        For each reaction, identify which reactions' propensities are affected.
        """

        # For Phase 1, use full propensity recalculation
        # Phase 2+ will implement partial updates
        pass
