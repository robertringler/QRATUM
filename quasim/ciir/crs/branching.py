r"""Layer 4 — Quantum-like branching: probabilistic evolution with interference.

Implements:
- BranchState: (graph, amplitude) pairs
- Branch evolution: one state → multiple possible successor states
- Interference: recombination of branches with similar graphs
- Normalisation: sum of |amplitude|^2 = 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasim.ciir.crs.graph import CRSGraph
from quasim.ciir.crs.rewrite import RewriteEngine

# ================================================================
# Branch state
# ================================================================


@dataclass
class BranchState:
    """A single branch in the CRS multiverse.

    Attributes
    ----------
    graph : CRSGraph
        Graph configuration for this branch.
    amplitude : complex
        Quantum-like probability amplitude.
    label : str
        Human-readable branch tag.
    """

    graph: CRSGraph
    amplitude: complex = 1.0 + 0.0j
    label: str = "root"


# ================================================================
# Branching engine
# ================================================================


class BranchingEngine:
    """Quantum-like branching, interference, and collapse engine.

    Parameters
    ----------
    initial_graph : CRSGraph
        Starting graph configuration.
    max_branches : int
        Maximum number of branches kept after pruning.
    interference_threshold : float
        Similarity threshold below which branches interfere.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        initial_graph: CRSGraph,
        max_branches: int = 100,
        interference_threshold: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.states: list[BranchState] = [
            BranchState(graph=initial_graph.copy(), amplitude=1.0 + 0.0j, label="root")
        ]
        self.max_branches = max_branches
        self.interference_threshold = interference_threshold
        self.rng = np.random.default_rng(seed)

    # ---- evolution --------------------------------------------------

    def branch_evolve(self, rewrite_engine: RewriteEngine) -> list[BranchState]:
        """Evolve each branch into one or more successor branches.

        For each existing state, every rule with probability < 1 forks
        the branch into a "fired" and "not-fired" outcome.  Rules with
        probability == 1 are applied deterministically.

        Returns the new list of branch states.
        """
        new_states: list[BranchState] = []

        for bs in self.states:
            branches = [bs]
            for rule in rewrite_engine.rules:
                next_branches: list[BranchState] = []
                for branch in branches:
                    if rule.probability >= 1.0:
                        # Deterministic: apply in-place
                        g = branch.graph.copy()
                        for nid in list(g.nodes):
                            new_state = rule.apply(g, nid, self.rng)
                            g.nodes[nid].state = new_state
                            g.nodes[nid].timestamp += 1
                        next_branches.append(
                            BranchState(
                                graph=g,
                                amplitude=branch.amplitude,
                                label=f"{branch.label}+{rule.name}",
                            )
                        )
                    else:
                        # Probabilistic: fork
                        p = rule.probability
                        # Branch A: rule fires
                        g_fire = branch.graph.copy()
                        for nid in list(g_fire.nodes):
                            new_s = rule.apply(g_fire, nid, self.rng)
                            g_fire.nodes[nid].state = new_s
                            g_fire.nodes[nid].timestamp += 1
                        amp_fire = branch.amplitude * np.sqrt(p)
                        next_branches.append(
                            BranchState(
                                graph=g_fire,
                                amplitude=amp_fire,
                                label=f"{branch.label}+{rule.name}",
                            )
                        )
                        # Branch B: rule does not fire
                        g_no = branch.graph.copy()
                        amp_no = branch.amplitude * np.sqrt(1.0 - p)
                        next_branches.append(
                            BranchState(
                                graph=g_no,
                                amplitude=amp_no,
                                label=f"{branch.label}+~{rule.name}",
                            )
                        )
                branches = next_branches
            new_states.extend(branches)

        self.states = new_states
        return self.states

    # ---- normalisation ----------------------------------------------

    def normalize(self) -> None:
        """Rescale amplitudes so that Σ|a|² = 1."""
        total = sum(abs(s.amplitude) ** 2 for s in self.states)
        if total < 1e-30:
            return
        factor = 1.0 / np.sqrt(total)
        for s in self.states:
            s.amplitude *= factor

    # ---- interference -----------------------------------------------

    def interfere(self) -> None:
        """Merge branches whose graphs are similar (constructive/destructive)."""
        if len(self.states) <= 1:
            return

        merged: list[BranchState] = []
        used: set[int] = set()

        for i in range(len(self.states)):
            if i in used:
                continue
            acc_amp = self.states[i].amplitude
            acc_graph = self.states[i].graph
            acc_label = self.states[i].label
            for j in range(i + 1, len(self.states)):
                if j in used:
                    continue
                sim = self._graph_similarity(acc_graph, self.states[j].graph)
                if sim < self.interference_threshold:
                    acc_amp += self.states[j].amplitude
                    used.add(j)
            used.add(i)
            merged.append(BranchState(graph=acc_graph, amplitude=acc_amp, label=acc_label))

        self.states = merged

    # ---- pruning ----------------------------------------------------

    def prune(self) -> None:
        """Remove low-probability branches and cap at *max_branches*."""
        if not self.states:
            return
        total = sum(abs(s.amplitude) ** 2 for s in self.states)
        if total < 1e-30:
            return
        threshold = 1e-6 * total
        self.states = [s for s in self.states if abs(s.amplitude) ** 2 >= threshold]
        if len(self.states) > self.max_branches:
            self.states.sort(key=lambda s: abs(s.amplitude) ** 2, reverse=True)
            self.states = self.states[: self.max_branches]

    # ---- probabilities & collapse -----------------------------------

    def probabilities(self) -> list[float]:
        """Return |amplitude|² for each branch."""
        return [abs(s.amplitude) ** 2 for s in self.states]

    def collapse(self, rng: np.random.Generator | None = None) -> CRSGraph:
        """Sample one branch according to Born probabilities.

        Returns
        -------
        CRSGraph
            Deep copy of the selected branch's graph.
        """
        if rng is None:
            rng = self.rng
        probs = np.array(self.probabilities())
        total = probs.sum()
        if total < 1e-30:
            return self.states[0].graph.copy()
        probs /= total
        idx = int(rng.choice(len(self.states), p=probs))
        return self.states[idx].graph.copy()

    # ---- similarity -------------------------------------------------

    @staticmethod
    def _graph_similarity(g1: CRSGraph, g2: CRSGraph) -> float:
        """Compare graphs by mean state-vector distance.

        Returns
        -------
        float
            0 means identical state vectors; larger values → more different.
        """
        common = set(g1.nodes) & set(g2.nodes)
        if not common:
            return float("inf")
        diffs: list[float] = []
        for nid in common:
            d = float(np.linalg.norm(g1.nodes[nid].state - g2.nodes[nid].state))
            diffs.append(d)
        return float(np.mean(diffs))
