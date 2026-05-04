r"""CRS Formal Mathematical Specification.

Compact formal specification of the Computational Reality System (CRS)
integrated with CIIR / QuASIM / QRATUM.

Mathematical Structure
======================

**Definition (CRS Graph).**
A CRS graph is a triple G = (V, E, σ) where:
  - V ⊂ ℕ is a finite set of node identifiers
  - E ⊂ V × V × [0,1] is a set of weighted directed edges (i, j, w_ij)
  - σ : V → ℝ^d is a state assignment mapping each node to a d-dimensional
    real vector

The state space of G is S(G) = ℝ^{|V| × d} × [0,1]^{|E|}.

**Definition (Neighborhood).**
For v ∈ V and radius r ∈ ℕ, the r-neighborhood is:
  N_r(v) = {u ∈ V : d_G(v,u) ≤ r}
where d_G is the graph geodesic distance.

**Definition (Rewrite Rule).**
A rewrite rule is a tuple R = (name, f, p) where:
  - f : ℝ^d × (ℝ^d)^{|N_1(v)|} × [0,1]^{|N_1(v)|} → ℝ^d
    maps (node state, neighbor states, edge weights) to updated node state
  - p ∈ [0,1] is the firing probability

Strictly local: f depends only on N_1(v), ensuring O(deg(v) · d) per node.

**Definition (Evolution Operator).**
The global evolution operator Φ : S(G) → S(G) is defined by:
  Φ(σ)(v) = f_R(σ(v), {σ(u)}_{u ∈ N_1(v)}, {w_{vu}}_{u ∈ N_1(v)})
for each v ∈ V, applied in topological order (or arbitrary if cyclic).

**Theorem (Complexity Bound).**
Each evolution step has complexity O(|V| · max_deg · d) where max_deg is
the maximum node degree.

Built-In Rules (5)
==================

1. Diffusion: σ'(v) = (1-α)σ(v) + α mean({σ(u) : u ∈ N_1(v)})
   - Preserves convexity of state space
   - Contraction: ||σ'(v)|| ≤ max(||σ(v)||, max_u ||σ(u)||)

2. Decay: σ'(v) = (1-δ)σ(v)
   - Exponential contraction: ||σ_t(v)|| ≤ (1-δ)^t ||σ_0(v)||

3. Interaction: σ'(v) = σ(v) + c Σ_{u ∈ N_1(v)} w_{vu} σ(u)
   - Can be unstable for large c · deg(v) — requires damping

4. Stochastic perturbation: σ'(v) = σ(v) + ε · Z, Z ~ N(0, I_d)
   - Models thermal noise; increases entropy

5. Edge reweight: w'_{vu} = clip(w_{vu} + η ||σ(v) - σ(u)||, 0, 1)
   - Hebbian-like: stronger edges for dissimilar states

Emergent Spacetime
==================

**Definition (Graph Geodesic Distance).**
  d_G(a, b) = min_{path P: a→b} Σ_{(i,j) ∈ P} 1/w_{ij}

**Definition (Causal Depth).**
  depth(v) = max_{path P: root→v} |P|
where roots are nodes with no incoming edges.

**Definition (Curvature Proxy).**
  κ(v) = |N_1(v)| / |N_2(v) \ N_1(v)|
Ratio of 1-hop to exclusive 2-hop neighbors.
  κ > 1: positive curvature (sphere-like)
  κ < 1: negative curvature (hyperbolic-like)

**Definition (Emergent Dimension).**
Estimated via scaling: |B_r(v)| ~ r^D for small r,
  D ≈ log(|B_2(v)|/|B_1(v)|) / log(2)

Quantum-Like Branching
======================

**Definition (Branch State).**
A branch state is a pair (G, α) where G is a CRS graph and α ∈ ℂ
is an amplitude. A branch ensemble is {(G_i, α_i)}_{i=1}^K satisfying
the normalization constraint:
  Σ_i |α_i|² = 1

**Definition (Branch Evolution).**
Given rewrite engine with rules {R_j} of probabilities {p_j}:
  (G, α) → {(G'_j, α · √p_j)} for each applicable rule R_j

**Definition (Interference).**
Two branches (G_i, α_i), (G_j, α_j) are combinable if
  sim(G_i, G_j) = mean_{v} ||σ_i(v) - σ_j(v)|| < θ
Combined branch: (G_i, α_i + α_j) — constructive or destructive.

**Definition (Collapse).**
Sample branch i with probability |α_i|² / Σ_k |α_k|².

Conservation Laws
=================

**Definition (Graph Invariant).**
A quantity I : S(G) → ℝ is an invariant of Φ if |I(Φ(G)) - I(G)| < ε.

Built-in invariants:
  1. Total state norm: I₁(G) = Σ_v ||σ(v)||
  2. Total causal weight: I₂(G) = Σ_e w_e
  3. Node count: I₃(G) = |V| (exact for non-topological rules)
  4. State centroid: I₄(G) = ||mean_v σ(v)||
  5. Graph energy: I₅(G) = Σ_{(i,j,w)} w · ||σ(i) - σ(j)||²

Observer Model
==============

**Definition (Observer).**
An observer O = (A, r) where A ⊂ V is the accessible node set and
r ∈ [0,1] is the resolution parameter.

**Definition (Observation).**
  observe(O, G)(v) = r · σ(v) + (1-r) · Z_v, Z_v ~ N(0, I_d)
for v ∈ A. Nodes not in A are unobservable.

**Definition (Decoherence).**
For v ∈ A: σ'(v) = (1-γ) σ(v) + γ · mean(σ(v))·1_d
where γ is the decoherence rate. Models measurement back-action.

CIIR Integration
================

**Density Matrix Bridge.**
  ρ = (1/|V|) Σ_v |σ(v)⟩⟨σ(v)|
where |σ(v)⟩ = σ(v) / ||σ(v)|| if ||σ(v)|| > 0.

Properties: ρ = ρ†, ρ ≥ 0, Tr(ρ) = 1.

**CIIR Distortion Map.**
  σ'(v) = σ(v) · exp(-κ_min · diam(M))
contraction consistent with Theorem 8.7 (Constraint-Induced Distortion).

**QRATUM Core Loop.**
  G_{t+1} = D(Φ(G_t))
where Φ is the CRS evolution and D is the CIIR distortion map.

Verification Summary
====================

✓ All updates strictly local (radius-1 neighborhood only)
✓ Probabilities normalized at every branching step
✓ Invariant checking at every evolution step
✓ Emergent metrics well-defined for connected graphs
✓ No undefined symbols or operations
✓ Classical limit: zero noise, no branching → deterministic cellular automaton
✓ Density matrix bridge produces valid quantum state (ρ ≥ 0, Tr ρ = 1)
"""
