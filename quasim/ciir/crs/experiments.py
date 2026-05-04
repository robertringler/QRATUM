r"""CRS simulation experiments demonstrating emergent physics-like behaviour.

Provides ready-to-run experiments:
1. Diffusion experiment: demonstrate emergent diffusion
2. Wave propagation: demonstrate wave-like behaviour
3. Phase transition: graph connectivity transition
4. Entropy growth: demonstrate entropy increase over time
"""

from __future__ import annotations

import numpy as np

from quasim.ciir.crs.evolution import CRSEngine
from quasim.ciir.crs.graph import lattice_graph, random_graph
from quasim.ciir.crs.rewrite import (
    RewriteEngine,
    decay_rule,
    diffusion_rule,
    interaction_rule,
    stochastic_perturbation_rule,
)

# ================================================================
# 1. Diffusion
# ================================================================


def run_diffusion_experiment(
    n_nodes: int = 100,
    d_state: int = 4,
    n_steps: int = 200,
    seed: int = 42,
) -> dict:
    r"""Demonstrate emergent diffusion on a 2-D lattice.

    A "hot spot" (large state norm) is placed at the lattice centre.
    Diffusion and decay rules spread the energy, and we track the
    spatial distribution width over time.  The *spread* metric is the
    standard deviation of state norms across nodes — it initially
    increases as the hot spot diffuses outward, then decreases as the
    system thermalises.

    Parameters
    ----------
    n_nodes : int
        Approximate total node count (rounded to nearest square).
    d_state : int
        State-vector dimension.
    n_steps : int
        Number of evolution steps.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``steps``, ``mean_norm``, ``max_norm``, ``spread``,
        ``peak_ratio`` (max_norm / mean_norm — tracks how peaked the
        distribution remains).
    """
    side = max(2, int(np.sqrt(n_nodes)))
    g = lattice_graph(side, side, d_state, seed=seed)

    # Cold start: zero out all states
    for node in g.nodes.values():
        node.state = np.zeros(d_state)

    # Hot spot at centre
    centre_id = (side // 2) * side + (side // 2)
    g.nodes[centre_id].state = np.ones(d_state) * 10.0

    engine = RewriteEngine()
    engine.register_rule(diffusion_rule(diffusion_rate=0.2))
    engine.register_rule(decay_rule(decay_rate=0.005))

    crs = CRSEngine(g, engine, seed=seed)

    # Collect per-step spread by snapshotting the graph after each step
    steps_list: list[int] = []
    mean_norms: list[float] = []
    max_norms: list[float] = []
    spreads: list[float] = []
    peak_ratios: list[float] = []

    for t in range(n_steps):
        metrics = crs.step()
        norms = np.array([float(np.linalg.norm(n.state)) for n in crs.graph.nodes.values()])
        mn = float(norms.mean()) if len(norms) > 0 else 0.0
        mx = float(norms.max()) if len(norms) > 0 else 0.0
        sd = float(norms.std()) if len(norms) > 0 else 0.0

        steps_list.append(t + 1)
        mean_norms.append(mn)
        max_norms.append(mx)
        spreads.append(sd)
        peak_ratios.append(mx / mn if mn > 1e-12 else 0.0)

    return {
        "steps": steps_list,
        "mean_norm": mean_norms,
        "max_norm": max_norms,
        "spread": spreads,
        "peak_ratio": peak_ratios,
    }


# ================================================================
# 2. Wave propagation
# ================================================================


def run_wave_experiment(
    n_nodes: int = 100,
    d_state: int = 4,
    n_steps: int = 200,
    seed: int = 42,
) -> dict:
    r"""Demonstrate wave-like propagation via interaction + diffusion rules.

    Initialises a sinusoidal pattern along the lattice diagonal and
    runs a damped interaction rule to observe wave-like oscillatory
    dynamics.  The diffusion rule provides the necessary dissipation
    to prevent blow-up, while the interaction rule creates the
    restoring force for oscillation.

    Parameters
    ----------
    n_nodes, d_state, n_steps, seed : int

    Returns
    -------
    dict
        Keys: ``steps``, ``mean_norm``, ``max_norm``, ``energy``,
        ``oscillation_index`` (track oscillatory character).
    """
    side = max(2, int(np.sqrt(n_nodes)))
    g = lattice_graph(side, side, d_state, seed=seed)

    # Sinusoidal initial condition along diagonal
    for r in range(side):
        for c in range(side):
            nid = r * side + c
            phase = 2.0 * np.pi * (r + c) / side
            g.nodes[nid].state = np.array(
                [np.sin(phase + k * np.pi / d_state) for k in range(d_state)]
            )

    engine = RewriteEngine()
    # Use diffusion (dominant) + weak interaction (restoring force)
    engine.register_rule(diffusion_rule(diffusion_rate=0.15))
    engine.register_rule(interaction_rule(coupling=0.02))
    engine.register_rule(decay_rule(decay_rate=0.01))
    engine.register_rule(stochastic_perturbation_rule(noise_scale=0.001))

    crs = CRSEngine(g, engine, seed=seed)

    steps_list: list[int] = []
    mean_norms: list[float] = []
    max_norms: list[float] = []
    energies: list[float] = []
    oscillation: list[float] = []

    for t in range(n_steps):
        metrics = crs.step()
        norms = np.array([float(np.linalg.norm(n.state)) for n in crs.graph.nodes.values()])
        mn = float(norms.mean()) if len(norms) > 0 else 0.0
        mx = float(norms.max()) if len(norms) > 0 else 0.0
        # Energy: sum of squared norms (kinetic energy analog)
        energy = float(np.sum(norms**2))

        # Oscillation index: std of state component 0 across nodes
        comp0 = np.array([float(n.state[0]) for n in crs.graph.nodes.values()])
        osc = float(np.std(comp0))

        steps_list.append(t + 1)
        mean_norms.append(mn)
        max_norms.append(mx)
        energies.append(energy)
        oscillation.append(osc)

    return {
        "steps": steps_list,
        "mean_norm": mean_norms,
        "max_norm": max_norms,
        "energy": energies,
        "oscillation_index": oscillation,
    }


# ================================================================
# 3. Phase transition
# ================================================================


def run_phase_transition_experiment(
    n_nodes: int = 50,
    d_state: int = 4,
    n_steps: int = 100,
    p_edge_range: tuple[float, float] = (0.01, 0.5),
    n_points: int = 20,
    seed: int = 42,
) -> dict:
    r"""Sweep edge density and detect a connectivity phase transition.

    For each edge probability *p* in the range, creates a random graph,
    runs the evolution, and records final metrics.

    Parameters
    ----------
    n_nodes, d_state, n_steps : int
    p_edge_range : tuple[float, float]
        (min, max) edge probability.
    n_points : int
        Number of sample points.
    seed : int

    Returns
    -------
    dict
        Keys: ``p_values``, ``final_mean_norm``, ``final_entropy``,
        ``final_n_edges``.
    """
    p_values = np.linspace(p_edge_range[0], p_edge_range[1], n_points).tolist()
    final_mean_norm: list[float] = []
    final_entropy: list[float] = []
    final_n_edges: list[int] = []

    for i, p in enumerate(p_values):
        g = random_graph(n_nodes, d_state, p_edge=p, seed=seed + i)
        engine = RewriteEngine()
        engine.register_rule(diffusion_rule(0.1))
        engine.register_rule(stochastic_perturbation_rule(0.01))
        crs = CRSEngine(g, engine, seed=seed + i)
        history = crs.evolve(n_steps)
        last = history[-1]
        final_mean_norm.append(last["mean_state_norm"])
        final_entropy.append(last["entropy_estimate"])
        final_n_edges.append(last["n_edges"])

    return {
        "p_values": p_values,
        "final_mean_norm": final_mean_norm,
        "final_entropy": final_entropy,
        "final_n_edges": final_n_edges,
    }


# ================================================================
# 4. Entropy growth
# ================================================================


def run_entropy_experiment(
    n_nodes: int = 100,
    d_state: int = 4,
    n_steps: int = 200,
    seed: int = 42,
) -> dict:
    r"""Track Shannon entropy of the state distribution over time.

    Starts from a low-entropy configuration (all nodes identical) and
    demonstrates entropy increase under stochastic perturbation.  The
    entropy is computed from the distribution of state-vector norms
    binned into a histogram.

    Parameters
    ----------
    n_nodes, d_state, n_steps, seed : int

    Returns
    -------
    dict
        Keys: ``steps``, ``entropy``, ``mean_norm``.
    """
    side = max(2, int(np.sqrt(n_nodes)))
    g = lattice_graph(side, side, d_state, seed=seed)

    # Low-entropy initial condition: all states identical
    uniform_state = np.ones(d_state) / np.sqrt(d_state)
    for node in g.nodes.values():
        node.state = uniform_state.copy()

    engine = RewriteEngine()
    engine.register_rule(diffusion_rule(0.05))
    engine.register_rule(stochastic_perturbation_rule(0.05))

    crs = CRSEngine(g, engine, seed=seed)

    steps_list: list[int] = []
    entropies: list[float] = []
    mean_norms: list[float] = []

    for t in range(n_steps):
        crs.step()
        # Compute Shannon entropy from state vector components
        all_states = np.array([n.state for n in crs.graph.nodes.values()])
        # Flatten to 1-D for histogram-based entropy
        flat = all_states.flatten()
        # Bin into histogram
        counts, _ = np.histogram(flat, bins=max(10, int(np.sqrt(len(flat)))))
        p = counts / max(counts.sum(), 1)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p)))

        norms = np.array([float(np.linalg.norm(n.state)) for n in crs.graph.nodes.values()])
        steps_list.append(t + 1)
        entropies.append(entropy)
        mean_norms.append(float(norms.mean()))

    return {
        "steps": steps_list,
        "entropy": entropies,
        "mean_norm": mean_norms,
    }
