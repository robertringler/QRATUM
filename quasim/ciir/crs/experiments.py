r"""CRS simulation experiments demonstrating emergent physics-like behaviour.

Provides ready-to-run experiments:
1. Diffusion experiment: demonstrate emergent diffusion
2. Wave propagation: demonstrate wave-like behaviour
3. Phase transition: graph connectivity transition
4. Entropy growth: demonstrate entropy increase over time
"""

from __future__ import annotations

import numpy as np

from quasim.ciir.crs.graph import CRSGraph, Node, Edge, lattice_graph, random_graph
from quasim.ciir.crs.rewrite import (
    RewriteEngine,
    diffusion_rule,
    decay_rule,
    interaction_rule,
    stochastic_perturbation_rule,
)
from quasim.ciir.crs.evolution import CRSEngine


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
    spatial distribution width over time.

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
        Keys: ``steps``, ``mean_norm``, ``max_norm``, ``spread``
        (standard deviation of state norms across nodes).
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
    history = crs.evolve(n_steps)

    spreads: list[float] = []
    for _ in range(n_steps):
        norms = np.array([np.linalg.norm(n.state) for n in g.nodes.values()])
        spreads.append(float(norms.std()))
    # Use per-step norms from history
    spreads_from_history = []
    for m in history:
        spreads_from_history.append(m["mean_state_norm"])

    return {
        "steps": [m["step"] for m in history],
        "mean_norm": [m["mean_state_norm"] for m in history],
        "max_norm": [m["max_state_norm"] for m in history],
        "spread": spreads_from_history,
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
    r"""Demonstrate wave-like propagation via interaction rule.

    Initialises a sinusoidal pattern along the lattice diagonal and
    runs the interaction rule to observe wave-like dynamics.

    Parameters
    ----------
    n_nodes, d_state, n_steps, seed : int

    Returns
    -------
    dict
        Keys: ``steps``, ``mean_norm``, ``max_norm``, ``energy``.
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
    engine.register_rule(interaction_rule(coupling=0.1))
    engine.register_rule(stochastic_perturbation_rule(noise_scale=0.001))

    crs = CRSEngine(g, engine, seed=seed)
    history = crs.evolve(n_steps)

    energies: list[float] = []
    for m in history:
        energies.append(m["total_causal_weight"])

    return {
        "steps": [m["step"] for m in history],
        "mean_norm": [m["mean_state_norm"] for m in history],
        "max_norm": [m["max_state_norm"] for m in history],
        "energy": energies,
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

    Starts from a low-entropy configuration (all nodes near-identical)
    and demonstrates monotonic entropy increase under stochastic
    perturbation.

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
    engine.register_rule(stochastic_perturbation_rule(0.02))

    crs = CRSEngine(g, engine, seed=seed)
    history = crs.evolve(n_steps)

    return {
        "steps": [m["step"] for m in history],
        "entropy": [m["entropy_estimate"] for m in history],
        "mean_norm": [m["mean_state_norm"] for m in history],
    }
