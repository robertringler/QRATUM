r"""Minimal Programmable Physics Kernel (MPPK) — Law-Discovery Engine.

Implements P = (S, R, U) where:
- S = graph-based state space with vector-valued nodes x_i ∈ R^d
- R = local interaction rules (programs governing state evolution)
- U = update operator executing rule application over S

Initial kernel:
    x_i(t+1) = x_i(t) + Σ_{j ∈ N(i)} w_ij · tanh(x_j(t) − x_i(t))

Energy functional:
    E(t) = Σ_{i,j} w_ij ||x_i(t) − x_j(t)||²

Momentum-like observable:
    P(t) = Σ_i (x_i(t) − x_i(t−1))

Execution loop (per generation):
    1. Simulate: run kernel for T timesteps, store trajectory
    2. Extract structure: fixed points, limit cycles, chaos, clusters
    3. Mine invariants: conserved quantities, symmetries, spectral stability
    4. Evaluate fitness: F = α·stability + β·structure + γ·boundedness + δ·compress
    5. Mutate rules: modify interaction fn, topology, stochasticity, etc.
    6. Select: retain top-k rule systems by fitness

Meta-learning layer:
    After multiple generations, infer emergent force laws, stable invariants,
    low-dimensional embeddings, and candidate "physical laws."

References
----------
CIIR Monograph Ch 20 (Programmable Physics / MPPK).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np
from numpy.typing import NDArray


# ====================================================================
# I. GRAPH STATE SPACE  S = (V, E)
# ====================================================================

@dataclass
class MPPKNode:
    """Graph vertex carrying a vector state x_i ∈ R^d.

    Attributes
    ----------
    id : int
        Unique node identifier.
    state : NDArray
        State vector x_i ∈ R^d.
    """

    id: int
    state: NDArray


@dataclass
class MPPKEdge:
    """Directed edge with weight w_ij ∈ R.

    Attributes
    ----------
    source : int
    target : int
    weight : float
        Interaction strength w_ij.
    """

    source: int
    target: int
    weight: float


class MPPKGraph:
    """Graph-based state space G = (V, E).

    Provides neighbor queries, adjacency, and Laplacian construction.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, MPPKNode] = {}
        self.edges: list[MPPKEdge] = []
        self._adjacency: dict[int, list[tuple[int, float]]] = {}

    def add_node(self, node: MPPKNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []

    def add_edge(self, edge: MPPKEdge) -> None:
        self.edges.append(edge)
        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = []
        self._adjacency[edge.source].append((edge.target, edge.weight))

    def neighbors(self, node_id: int) -> list[tuple[int, float]]:
        """Return [(neighbor_id, weight)] for node_id."""
        return self._adjacency.get(node_id, [])

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def dim(self) -> int:
        """State vector dimension d."""
        if not self.nodes:
            return 0
        return next(iter(self.nodes.values())).state.shape[0]

    def state_matrix(self) -> NDArray:
        """Return (N, d) matrix of all node states, sorted by id."""
        ids = sorted(self.nodes.keys())
        return np.array([self.nodes[i].state for i in ids])

    def set_states(self, X: NDArray) -> None:
        """Set node states from (N, d) matrix, sorted by id."""
        ids = sorted(self.nodes.keys())
        for idx, nid in enumerate(ids):
            self.nodes[nid].state = X[idx].copy()

    def laplacian(self) -> NDArray:
        """Construct graph Laplacian L = D − W (weighted, directed→symmetrized)."""
        n = self.n_nodes
        ids = sorted(self.nodes.keys())
        id_to_idx = {nid: i for i, nid in enumerate(ids)}
        W = np.zeros((n, n))
        for e in self.edges:
            i, j = id_to_idx.get(e.source), id_to_idx.get(e.target)
            if i is not None and j is not None:
                W[i, j] += e.weight
                W[j, i] += e.weight  # symmetrize
        W *= 0.5
        D = np.diag(W.sum(axis=1))
        return D - W

    def copy(self) -> "MPPKGraph":
        """Deep copy the graph."""
        g = MPPKGraph()
        for n in self.nodes.values():
            g.add_node(MPPKNode(n.id, n.state.copy()))
        for e in self.edges:
            g.add_edge(MPPKEdge(e.source, e.target, e.weight))
        return g


# ====================================================================
# II. INTERACTION RULES  R
# ====================================================================

class InteractionType(Enum):
    """Type tag for rule interaction functions."""
    TANH = auto()        # Original kernel
    SIGMOID = auto()     # Smoother saturation
    QUADRATIC = auto()   # Polynomial coupling
    RELU_CLIP = auto()   # Piecewise linear, clipped
    CUBIC = auto()       # Higher-order nonlinearity


@dataclass
class MPPKRule:
    """A local interaction rule.

    Parameters
    ----------
    name : str
    interaction_fn : Callable
        (diff: NDArray) → NDArray — bounded nonlinear activation.
    interaction_type : InteractionType
    noise_sigma : float
        Stochastic perturbation amplitude (0 = deterministic).
    higher_order : int
        Neighbor radius (1 = direct neighbors only).
    coupling_matrix : NDArray or None
        Optional d×d coupling matrix for nonlinear mixing.
    """

    name: str
    interaction_fn: Callable[[NDArray], NDArray]
    interaction_type: InteractionType = InteractionType.TANH
    noise_sigma: float = 0.0
    higher_order: int = 1
    coupling_matrix: NDArray | None = None


def tanh_interaction(diff: NDArray) -> NDArray:
    """Default kernel: tanh(x_j − x_i)."""
    return np.tanh(diff)


def sigmoid_interaction(diff: NDArray) -> NDArray:
    """Sigmoid variant: 2σ(diff) − 1."""
    return 2.0 / (1.0 + np.exp(-diff)) - 1.0


def quadratic_interaction(diff: NDArray) -> NDArray:
    """Quadratic: diff · (1 − ||diff||²/4), bounded."""
    norm_sq = np.sum(diff ** 2)
    return diff * max(0.0, 1.0 - norm_sq / 4.0)


def relu_clip_interaction(diff: NDArray) -> NDArray:
    """ReLU clipped to [−1, 1]."""
    return np.clip(diff, -1.0, 1.0)


def cubic_interaction(diff: NDArray) -> NDArray:
    """Cubic: tanh(diff³ / (1 + ||diff||²))."""
    norm_sq = np.sum(diff ** 2)
    return np.tanh(diff ** 3 / (1.0 + norm_sq))


INTERACTION_REGISTRY: dict[InteractionType, Callable] = {
    InteractionType.TANH: tanh_interaction,
    InteractionType.SIGMOID: sigmoid_interaction,
    InteractionType.QUADRATIC: quadratic_interaction,
    InteractionType.RELU_CLIP: relu_clip_interaction,
    InteractionType.CUBIC: cubic_interaction,
}


def make_default_rule() -> MPPKRule:
    """Construct the initial kernel rule (DO NOT ALTER UNTIL VALIDATED)."""
    return MPPKRule(
        name="TanhKernel",
        interaction_fn=tanh_interaction,
        interaction_type=InteractionType.TANH,
    )


# ====================================================================
# III. UPDATE OPERATOR  U
# ====================================================================

class UpdateOperator:
    r"""Execute rule application: x_i(t+1) = x_i(t) + Σ w_ij f(x_j − x_i).

    Stores previous states for momentum computation.
    """

    def __init__(self, rule: MPPKRule, seed: int = 42) -> None:
        self.rule = rule
        self.rng = np.random.default_rng(seed)
        self._prev_states: NDArray | None = None

    def step(self, graph: MPPKGraph) -> None:
        """Advance all nodes by one timestep in-place."""
        ids = sorted(graph.nodes.keys())
        X_old = graph.state_matrix()
        self._prev_states = X_old.copy()
        X_new = X_old.copy()

        for idx, nid in enumerate(ids):
            delta = np.zeros(graph.dim)
            for neighbor_id, w in graph.neighbors(nid):
                if neighbor_id in graph.nodes:
                    n_idx = ids.index(neighbor_id)
                    diff = X_old[n_idx] - X_old[idx]

                    # Apply coupling matrix if present
                    if self.rule.coupling_matrix is not None:
                        diff = self.rule.coupling_matrix @ diff

                    delta += w * self.rule.interaction_fn(diff)

            # Stochastic perturbation
            if self.rule.noise_sigma > 0:
                delta += self.rule.noise_sigma * self.rng.standard_normal(graph.dim)

            X_new[idx] = X_old[idx] + delta

        graph.set_states(X_new)

    @property
    def prev_states(self) -> NDArray | None:
        return self._prev_states


# ====================================================================
# IV. OBSERVABLES — Energy & Momentum
# ====================================================================

def compute_energy(graph: MPPKGraph) -> float:
    r"""E(t) = Σ_{i,j} w_ij ||x_i(t) − x_j(t)||²."""
    E = 0.0
    for edge in graph.edges:
        if edge.source in graph.nodes and edge.target in graph.nodes:
            xi = graph.nodes[edge.source].state
            xj = graph.nodes[edge.target].state
            E += edge.weight * float(np.sum((xi - xj) ** 2))
    return E


def compute_momentum(graph: MPPKGraph, prev_states: NDArray | None) -> NDArray:
    r"""P(t) = Σ_i (x_i(t) − x_i(t−1))."""
    if prev_states is None:
        return np.zeros(graph.dim)
    X = graph.state_matrix()
    return np.sum(X - prev_states, axis=0)


def compute_total_norm(graph: MPPKGraph) -> float:
    """Σ_i ||x_i||."""
    return float(sum(np.linalg.norm(n.state) for n in graph.nodes.values()))


# ====================================================================
# V. TRAJECTORY & SIMULATION
# ====================================================================

@dataclass
class Trajectory:
    """Full simulation record.

    Attributes
    ----------
    states : list of NDArray
        (N, d) state matrices at each timestep.
    energies : list of float
    momenta : list of NDArray
    norms : list of float
    """

    states: list[NDArray] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    momenta: list[NDArray] = field(default_factory=list)
    norms: list[float] = field(default_factory=list)

    @property
    def T(self) -> int:
        """Number of recorded timesteps."""
        return len(self.states)


def simulate(
    graph: MPPKGraph,
    rule: MPPKRule,
    T: int,
    seed: int = 42,
) -> Trajectory:
    """STEP 1 — Run kernel for T timesteps, store full trajectory."""
    updater = UpdateOperator(rule, seed)
    traj = Trajectory()
    traj.states.append(graph.state_matrix().copy())
    traj.energies.append(compute_energy(graph))
    traj.momenta.append(np.zeros(graph.dim))
    traj.norms.append(compute_total_norm(graph))

    for _ in range(T):
        updater.step(graph)
        traj.states.append(graph.state_matrix().copy())
        traj.energies.append(compute_energy(graph))
        traj.momenta.append(compute_momentum(graph, updater.prev_states))
        traj.norms.append(compute_total_norm(graph))

    return traj


# ====================================================================
# VI. STRUCTURE EXTRACTION (STEP 2)
# ====================================================================

@dataclass
class StructureReport:
    """Detected dynamical structures.

    Attributes
    ----------
    has_fixed_point : bool
    has_limit_cycle : bool
    is_chaotic : bool
    n_clusters : int
    symmetry_breaking : bool
    lyapunov_proxy : float
        Approximate largest Lyapunov exponent.
    """

    has_fixed_point: bool = False
    has_limit_cycle: bool = False
    is_chaotic: bool = False
    n_clusters: int = 1
    symmetry_breaking: bool = False
    lyapunov_proxy: float = 0.0


def extract_structure(
    traj: Trajectory,
    fp_tol: float = 1e-4,
    cycle_window: int = 20,
    cluster_tol: float = 0.5,
) -> StructureReport:
    """STEP 2 — Detect fixed points, limit cycles, chaos, clusters."""
    report = StructureReport()

    if traj.T < 3:
        return report

    # --- Fixed point detection ---
    # Check if last states are nearly identical
    last_diff = np.linalg.norm(traj.states[-1] - traj.states[-2])
    report.has_fixed_point = last_diff < fp_tol

    # --- Limit cycle detection ---
    # Check if trajectory returns close to a previous state
    if traj.T > cycle_window:
        final = traj.states[-1]
        for i in range(max(0, traj.T - cycle_window), traj.T - 2):
            if np.linalg.norm(final - traj.states[i]) < fp_tol * 10:
                report.has_limit_cycle = True
                break

    # --- Chaos proxy (Lyapunov exponent approximation) ---
    diffs = []
    for t in range(1, min(traj.T, 50)):
        d = np.linalg.norm(traj.states[t] - traj.states[t - 1])
        if d > 1e-30:
            diffs.append(np.log(d))
    if len(diffs) >= 2:
        report.lyapunov_proxy = float(np.mean(diffs))
    report.is_chaotic = report.lyapunov_proxy > 0.5

    # --- Cluster formation ---
    final_states = traj.states[-1]  # (N, d)
    N = final_states.shape[0]
    if N > 1:
        # Simple distance-based clustering
        clusters = _simple_cluster(final_states, cluster_tol)
        report.n_clusters = clusters
        report.symmetry_breaking = clusters > 1

    return report


def _simple_cluster(X: NDArray, tol: float) -> int:
    """Count clusters via greedy radius-based grouping."""
    N = X.shape[0]
    assigned = [False] * N
    n_clusters = 0
    for i in range(N):
        if assigned[i]:
            continue
        n_clusters += 1
        assigned[i] = True
        for j in range(i + 1, N):
            if not assigned[j] and np.linalg.norm(X[i] - X[j]) < tol:
                assigned[j] = True
    return n_clusters


# ====================================================================
# VII. INVARIANT MINING (STEP 3)
# ====================================================================

@dataclass
class InvariantReport:
    """Mined invariants and symmetries.

    Attributes
    ----------
    energy_conserved : bool
    energy_rel_var : float
    momentum_conserved : bool
    momentum_rel_var : float
    norm_conserved : bool
    norm_rel_var : float
    laplacian_eigenvalues : NDArray
    spectral_gap : float
    spectral_stable : bool
    """

    energy_conserved: bool = False
    energy_rel_var: float = 1.0
    momentum_conserved: bool = False
    momentum_rel_var: float = 1.0
    norm_conserved: bool = False
    norm_rel_var: float = 1.0
    laplacian_eigenvalues: NDArray = field(
        default_factory=lambda: np.array([])
    )
    spectral_gap: float = 0.0
    spectral_stable: bool = False


def mine_invariants(
    traj: Trajectory,
    graph: MPPKGraph,
    conservation_tol: float = 0.05,
) -> InvariantReport:
    """STEP 3 — Identify conserved quantities, symmetries, spectral stability."""
    report = InvariantReport()

    # Energy conservation
    if traj.energies:
        E = np.array(traj.energies)
        mean_E = np.mean(E)
        report.energy_rel_var = float(np.std(E) / (abs(mean_E) + 1e-30))
        report.energy_conserved = report.energy_rel_var < conservation_tol

    # Momentum conservation
    if traj.momenta:
        P_norms = [float(np.linalg.norm(p)) for p in traj.momenta]
        mean_P = np.mean(P_norms)
        report.momentum_rel_var = float(np.std(P_norms) / (abs(mean_P) + 1e-30))
        report.momentum_conserved = report.momentum_rel_var < conservation_tol

    # Norm conservation
    if traj.norms:
        N_arr = np.array(traj.norms)
        mean_N = np.mean(N_arr)
        report.norm_rel_var = float(np.std(N_arr) / (abs(mean_N) + 1e-30))
        report.norm_conserved = report.norm_rel_var < conservation_tol

    # Spectral stability of graph Laplacian
    L = graph.laplacian()
    eigvals = np.sort(np.linalg.eigvalsh(L))
    report.laplacian_eigenvalues = eigvals
    if len(eigvals) >= 2:
        report.spectral_gap = float(eigvals[1])  # Fiedler value
        report.spectral_stable = report.spectral_gap > 1e-6

    return report


# ====================================================================
# VIII. FITNESS EVALUATION (STEP 4)
# ====================================================================

@dataclass
class FitnessResult:
    """Fitness decomposition.

    F = α·stability + β·structure + γ·boundedness + δ·compressibility
    """

    stability: float = 0.0
    structure_formation: float = 0.0
    boundedness: float = 0.0
    compressibility: float = 0.0
    total: float = 0.0


def evaluate_fitness(
    traj: Trajectory,
    structure: StructureReport,
    invariants: InvariantReport,
    rule: MPPKRule,
    graph: MPPKGraph,
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.2,
) -> FitnessResult:
    """STEP 4 — F = α·stability + β·structure + γ·boundedness + δ·compressibility."""
    result = FitnessResult()

    # --- Stability: low energy variation + fixed point or cycle ---
    stab = 0.0
    if invariants.energy_conserved:
        stab += 0.4
    if structure.has_fixed_point or structure.has_limit_cycle:
        stab += 0.3
    if invariants.spectral_stable:
        stab += 0.3
    result.stability = min(stab, 1.0)

    # --- Structure formation: clusters, symmetry breaking ---
    struct = 0.0
    if structure.n_clusters > 1:
        struct += min(structure.n_clusters / 5.0, 0.5)
    if structure.symmetry_breaking:
        struct += 0.3
    if not structure.is_chaotic:
        struct += 0.2
    result.structure_formation = min(struct, 1.0)

    # --- Boundedness: all states remain finite and bounded ---
    bounded = 1.0
    if traj.norms:
        max_norm = max(traj.norms)
        if max_norm > 1e6 or not np.isfinite(max_norm):
            bounded = 0.0
        elif max_norm > 100:
            bounded = max(0, 1.0 - max_norm / 1000.0)
    result.boundedness = bounded

    # --- Compressibility: lower parameter count → better ---
    n_params = _count_rule_params(rule)
    result.compressibility = 1.0 / (1.0 + n_params / 10.0)

    result.total = (
        alpha * result.stability
        + beta * result.structure_formation
        + gamma * result.boundedness
        + delta * result.compressibility
    )
    return result


def _count_rule_params(rule: MPPKRule) -> int:
    """Count effective parameters in a rule."""
    count = 0
    # noise_sigma is a parameter
    if rule.noise_sigma > 0:
        count += 1
    # coupling matrix parameters
    if rule.coupling_matrix is not None:
        count += int(np.count_nonzero(rule.coupling_matrix))
    # interaction type counts as 1
    count += 1
    return count


# ====================================================================
# IX. RULE MUTATION (STEP 5)
# ====================================================================

class RuleMutator:
    """Generate candidate rule variations.

    Mutation strategies:
    - Modify interaction function f
    - Alter graph topology E
    - Introduce controlled stochasticity
    - Add higher-order neighbor interactions
    - Add nonlinear coupling terms

    Constraint: reject any rule causing immediate divergence.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def mutate_interaction(self, rule: MPPKRule) -> MPPKRule:
        """Switch to a different bounded interaction function."""
        types = list(InteractionType)
        new_type = types[self.rng.integers(len(types))]
        return MPPKRule(
            name=f"{new_type.name}Kernel",
            interaction_fn=INTERACTION_REGISTRY[new_type],
            interaction_type=new_type,
            noise_sigma=rule.noise_sigma,
            higher_order=rule.higher_order,
            coupling_matrix=(
                rule.coupling_matrix.copy() if rule.coupling_matrix is not None else None
            ),
        )

    def mutate_topology(self, graph: MPPKGraph, p_rewire: float = 0.1) -> MPPKGraph:
        """Rewire edges with probability p_rewire."""
        g = graph.copy()
        node_ids = sorted(g.nodes.keys())
        if len(node_ids) < 2:
            return g

        new_edges = []
        for e in g.edges:
            if self.rng.random() < p_rewire:
                new_target = node_ids[self.rng.integers(len(node_ids))]
                while new_target == e.source:
                    new_target = node_ids[self.rng.integers(len(node_ids))]
                new_edges.append(MPPKEdge(e.source, new_target, e.weight))
            else:
                new_edges.append(e)

        g.edges = new_edges
        # Rebuild adjacency
        g._adjacency = {nid: [] for nid in g.nodes}
        for e in g.edges:
            g._adjacency[e.source].append((e.target, e.weight))
        return g

    def mutate_noise(self, rule: MPPKRule, sigma_range: tuple = (0.0, 0.1)) -> MPPKRule:
        """Add or modify stochastic perturbation."""
        new_sigma = float(self.rng.uniform(*sigma_range))
        return MPPKRule(
            name=rule.name + "_noisy",
            interaction_fn=rule.interaction_fn,
            interaction_type=rule.interaction_type,
            noise_sigma=new_sigma,
            higher_order=rule.higher_order,
            coupling_matrix=(
                rule.coupling_matrix.copy() if rule.coupling_matrix is not None else None
            ),
        )

    def mutate_coupling(self, rule: MPPKRule, dim: int, sigma: float = 0.1) -> MPPKRule:
        """Add a coupling matrix perturbation."""
        if rule.coupling_matrix is not None:
            C = rule.coupling_matrix + self.rng.standard_normal((dim, dim)) * sigma
        else:
            C = np.eye(dim) + self.rng.standard_normal((dim, dim)) * sigma
        return MPPKRule(
            name=rule.name + "_coupled",
            interaction_fn=rule.interaction_fn,
            interaction_type=rule.interaction_type,
            noise_sigma=rule.noise_sigma,
            higher_order=rule.higher_order,
            coupling_matrix=C,
        )

    def mutate_weights(self, graph: MPPKGraph, sigma: float = 0.05) -> MPPKGraph:
        """Perturb edge weights."""
        g = graph.copy()
        for e in g.edges:
            e.weight += self.rng.standard_normal() * sigma
        # Rebuild adjacency
        g._adjacency = {nid: [] for nid in g.nodes}
        for e in g.edges:
            g._adjacency[e.source].append((e.target, e.weight))
        return g

    def generate_candidates(
        self,
        rule: MPPKRule,
        graph: MPPKGraph,
        n: int = 5,
    ) -> list[tuple[MPPKRule, MPPKGraph]]:
        """STEP 5 — Generate n candidate (rule, graph) variations."""
        candidates = []
        strategies = ["interaction", "topology", "noise", "coupling", "weights"]
        dim = graph.dim if graph.dim > 0 else 4

        for _ in range(n):
            strat = strategies[self.rng.integers(len(strategies))]
            if strat == "interaction":
                candidates.append((self.mutate_interaction(rule), graph.copy()))
            elif strat == "topology":
                candidates.append((copy.deepcopy(rule), self.mutate_topology(graph)))
            elif strat == "noise":
                candidates.append((self.mutate_noise(rule), graph.copy()))
            elif strat == "coupling":
                candidates.append((self.mutate_coupling(rule, dim), graph.copy()))
            elif strat == "weights":
                candidates.append((copy.deepcopy(rule), self.mutate_weights(graph)))

        return candidates


# ====================================================================
# X. META-LEARNING LAYER
# ====================================================================

@dataclass
class MetaLearningReport:
    """Inferred emergent laws after multiple generations.

    Attributes
    ----------
    effective_force_law : str
        Description of emergent interaction pattern.
    stable_invariants : list of str
        Names of quantities that remained conserved across generations.
    embedding_dim : int
        Effective dimension of dynamics from PCA.
    explained_variance_ratio : float
        Fraction of variance captured by embedding_dim.
    candidate_laws : list of str
        Human-readable candidate "physical law" descriptions.
    """

    effective_force_law: str = ""
    stable_invariants: list[str] = field(default_factory=list)
    embedding_dim: int = 0
    explained_variance_ratio: float = 0.0
    candidate_laws: list[str] = field(default_factory=list)


def meta_learn(
    generation_results: list["GenerationResult"],
    variance_threshold: float = 0.9,
) -> MetaLearningReport:
    """After multiple generations, infer emergent laws and embeddings."""
    report = MetaLearningReport()

    if not generation_results:
        return report

    # --- Stable invariants: quantities conserved in ≥ half of generations ---
    energy_conserved_count = sum(
        1 for g in generation_results if g.invariants.energy_conserved
    )
    momentum_conserved_count = sum(
        1 for g in generation_results if g.invariants.momentum_conserved
    )
    norm_conserved_count = sum(
        1 for g in generation_results if g.invariants.norm_conserved
    )
    half = len(generation_results) / 2
    if energy_conserved_count >= half:
        report.stable_invariants.append("Energy")
    if momentum_conserved_count >= half:
        report.stable_invariants.append("Momentum")
    if norm_conserved_count >= half:
        report.stable_invariants.append("TotalNorm")

    # --- Low-dimensional embedding via PCA ---
    all_states = []
    for g in generation_results:
        if g.trajectory.states:
            for S in g.trajectory.states[-5:]:  # Last 5 timesteps
                all_states.append(S.ravel())
    if len(all_states) >= 3:
        X = np.array(all_states)
        X_centered = X - X.mean(axis=0)
        cov = X_centered.T @ X_centered / len(X)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, 0)
        total_var = eigvals.sum()
        if total_var > 1e-30:
            cum_var = np.cumsum(eigvals) / total_var
            report.embedding_dim = int(np.searchsorted(cum_var, variance_threshold) + 1)
            report.explained_variance_ratio = float(
                cum_var[min(report.embedding_dim - 1, len(cum_var) - 1)]
            )

    # --- Effective force law ---
    best = max(generation_results, key=lambda g: g.fitness.total)
    report.effective_force_law = (
        f"f(Δx) = {best.rule.interaction_type.name.lower()}(Δx)"
    )
    if best.rule.noise_sigma > 0:
        report.effective_force_law += f" + N(0, {best.rule.noise_sigma:.4f})"

    # --- Candidate laws ---
    if report.stable_invariants:
        report.candidate_laws.append(
            f"Conservation: {', '.join(report.stable_invariants)} "
            f"remain approximately constant under evolution."
        )
    if best.structure.has_fixed_point:
        report.candidate_laws.append(
            "Equilibrium law: the system possesses stable fixed points."
        )
    if best.structure.n_clusters > 1:
        report.candidate_laws.append(
            f"Clustering law: dynamics spontaneously break into "
            f"{best.structure.n_clusters} clusters."
        )
    if best.invariants.spectral_stable:
        report.candidate_laws.append(
            f"Spectral law: the graph Laplacian has spectral gap "
            f"Δλ = {best.invariants.spectral_gap:.4f} > 0."
        )

    return report


# ====================================================================
# XI. OPTIONAL EXTENSIONS
# ====================================================================

@dataclass
class PhaseNode(MPPKNode):
    """Extended node with phase variable: x_i → (x_i, θ_i)."""
    phase: float = 0.0


def adaptive_rewire(
    graph: MPPKGraph,
    threshold: float = 0.5,
    rng: np.random.Generator | None = None,
) -> MPPKGraph:
    """Adaptive graph rewiring: E(t+1) = f(E(t), X(t)).

    Remove edges with low-activity endpoints, add edges between
    similar-state nodes.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    g = graph.copy()
    ids = sorted(g.nodes.keys())
    if len(ids) < 2:
        return g

    # Compute pairwise state distances
    X = g.state_matrix()
    new_edges = []
    for e in g.edges:
        i_idx = ids.index(e.source)
        j_idx = ids.index(e.target)
        dist = np.linalg.norm(X[i_idx] - X[j_idx])
        if dist < threshold * 2:  # Keep if states are similar enough
            new_edges.append(e)

    # Add edges between close nodes not already connected
    connected = {(e.source, e.target) for e in new_edges}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if (ids[i], ids[j]) not in connected:
                dist = np.linalg.norm(X[i] - X[j])
                if dist < threshold:
                    w = float(rng.uniform(0.1, 0.5))
                    new_edges.append(MPPKEdge(ids[i], ids[j], w))
                    connected.add((ids[i], ids[j]))

    g.edges = new_edges
    g._adjacency = {nid: [] for nid in g.nodes}
    for e in g.edges:
        g._adjacency[e.source].append((e.target, e.weight))
    return g


# ====================================================================
# XII. GENERATION RESULT & FULL ENGINE
# ====================================================================

@dataclass
class GenerationResult:
    """Result of one generation of the MPPK evolution loop."""

    generation: int
    rule: MPPKRule
    trajectory: Trajectory
    structure: StructureReport
    invariants: InvariantReport
    fitness: FitnessResult


class MPPKEngine:
    """Full Minimal Programmable Physics Kernel engine.

    Orchestrates the 6-step execution loop and meta-learning.
    """

    def __init__(
        self,
        n_nodes: int = 10,
        dim: int = 4,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.n_nodes = n_nodes
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self.mutator = RuleMutator(seed)

    # ---- Initialization (VIII) ----------------------------------

    def initialize_graph(self) -> MPPKGraph:
        """Initialize random graph G with small N nodes."""
        g = MPPKGraph()
        for i in range(self.n_nodes):
            state = self.rng.standard_normal(self.dim) * 0.5
            g.add_node(MPPKNode(id=i, state=state))

        # Random bounded weights
        for i in range(self.n_nodes):
            n_neighbors = self.rng.integers(1, max(2, self.n_nodes // 2))
            targets = self.rng.choice(
                [j for j in range(self.n_nodes) if j != i],
                size=min(n_neighbors, self.n_nodes - 1),
                replace=False,
            )
            for j in targets:
                w = float(self.rng.uniform(0.05, 0.5))
                g.add_edge(MPPKEdge(source=i, target=int(j), weight=w))
        return g

    # ---- Single generation step ---------------------------------

    def run_generation(
        self,
        graph: MPPKGraph,
        rule: MPPKRule,
        T: int = 100,
        generation: int = 0,
    ) -> GenerationResult:
        """Execute Steps 1–4 for a single (rule, graph) pair."""
        # STEP 1 — Simulate
        g = graph.copy()
        traj = simulate(g, rule, T, seed=int(self.rng.integers(2**31)))

        # STEP 2 — Structure extraction
        structure = extract_structure(traj)

        # STEP 3 — Invariant mining
        invariants = mine_invariants(traj, g)

        # STEP 4 — Fitness evaluation
        fitness = evaluate_fitness(
            traj, structure, invariants, rule, g,
            self.alpha, self.beta, self.gamma, self.delta,
        )

        return GenerationResult(
            generation=generation,
            rule=rule,
            trajectory=traj,
            structure=structure,
            invariants=invariants,
            fitness=fitness,
        )

    # ---- Full evolution loop ------------------------------------

    def evolve(
        self,
        generations: int = 10,
        T: int = 100,
        n_candidates: int = 5,
        top_k: int = 1,
    ) -> tuple[list[GenerationResult], MetaLearningReport]:
        """Execute the full mutation-selection loop.

        Returns (generation_results, meta_learning_report).
        """
        # Initialization (VIII)
        graph = self.initialize_graph()
        rule = make_default_rule()

        results: list[GenerationResult] = []

        # Baseline run
        gen_result = self.run_generation(graph, rule, T, generation=0)
        results.append(gen_result)

        best_rule = rule
        best_graph = graph

        for gen in range(1, generations):
            # STEP 5 — Mutation
            candidates = self.mutator.generate_candidates(
                best_rule, best_graph, n=n_candidates,
            )

            # Evaluate each candidate
            scored: list[tuple[float, MPPKRule, MPPKGraph, GenerationResult]] = []
            for c_rule, c_graph in candidates:
                try:
                    c_result = self.run_generation(c_graph, c_rule, T, gen)
                    # Reject immediate divergence
                    if (c_result.trajectory.norms
                            and max(c_result.trajectory.norms) > 1e6):
                        continue
                    scored.append((c_result.fitness.total, c_rule, c_graph, c_result))
                except (ValueError, FloatingPointError):
                    continue  # Skip divergent candidates

            # Also evaluate incumbent
            inc_result = self.run_generation(best_graph, best_rule, T, gen)
            scored.append((inc_result.fitness.total, best_rule, best_graph, inc_result))

            # STEP 6 — Selection: top-k
            scored.sort(key=lambda x: -x[0])
            if scored:
                _, best_rule, best_graph, best_result = scored[0]
                results.append(best_result)
            else:
                results.append(inc_result)

        # Meta-learning
        meta = meta_learn(results)

        return results, meta
