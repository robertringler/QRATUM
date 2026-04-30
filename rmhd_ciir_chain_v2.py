"""
RMHD/CIIR Whitepaper — Multi-Agent Orchestration Chain v2
==========================================================
Incorporates:
  - Stratified projection resolution (Whitney stratification of C)
  - Revised Theorems 5.2' and 6.1' (reconnection singularity handled)
  - Input-to-state stability (ISS) replacement near C_sing
  - Section 1 LaTeX already resolved from prior agent output
  - Full notation registry from Agent 0 outline JSON

Compatible with: Anthropic API directly, LangGraph, AutoGen
Model: claude-sonnet-4-20250514 throughout
Run order: Agent 0 → Agents 1-5 (parallel) → Agent 6 (red team)
           → Agent 7 (falsifiability) → Agent 8 (integration)

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python rmhd_ciir_chain_v2.py [--model MODEL] [--max-workers N]
                                  [--out-json PATH] [--out-tex PATH]

Options:
    --model       Anthropic model name (default: claude-sonnet-4-20250514)
    --max-workers Number of parallel workers for Phase 2 (default: 5)
    --out-json    Output JSON path (default: whitepaper_v2_output.json)
    --out-tex     Output LaTeX path (default: whitepaper_v2.tex)
"""

import json
import asyncio
import os
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

try:
    import anthropic
except ImportError:
    sys.exit(
        "Error: 'anthropic' package is not installed.\n"
        "Install it with:  pip install anthropic"
    )

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"

# =============================================================================
# SECTION 1 LATEX — already produced by prior agent run, injected directly
# =============================================================================

SECTION_1_LATEX = r"""
\section{Mathematical Foundations: State Spaces and Constraint Manifolds}
\label{sec:foundations}

\subsection{Function Spaces and State Manifold}

\begin{definition}[Periodic Sobolev Spaces]
\label{def:sobolev}
Let $\Omega = [0, L_x) \times [0, L_y) \subset \mathbb{R}^2$ be the periodic domain.
For $s \in \mathbb{R}$, the periodic Sobolev space $H^s_{\mathrm{per}}(\Omega)$ is
\[
H^s_{\mathrm{per}}(\Omega) = \left\{ u \in \mathcal{D}'(\Omega) : u\ \Omega\text{-periodic},\
\|u\|_{H^s} < \infty \right\},
\]
with norm $\|u\|_{H^s}^2 = \sum_{k \in \mathbb{Z}^2} (1 + |k|^2)^s |\hat{u}_k|^2$.
\end{definition}

\begin{definition}[RMHD State Manifold]
\label{def:state-manifold}
$\mathcal{M} = H^2_{\mathrm{per}}(\Omega) \times H^2_{\mathrm{per}}(\Omega)$,
with elements $(\psi, \omega)$: $\psi$ the magnetic flux function
($\mathbf{B}_\perp = \nabla\psi \times \hat{z}$) and $\omega$ the vorticity
($\omega = \nabla^2\varphi$, $\mathbf{v}_\perp = \hat{z} \times \nabla\varphi$).
\end{definition}

\begin{theorem}[Sobolev Embedding]
\label{thm:sobolev-embedding}
For $s > 2$ (with $d=2$): $H^s_{\mathrm{per}}(\Omega) \hookrightarrow C^1(\Omega)$
with $\|u\|_{C^1} \leq C_{\mathrm{Sob}} \|u\|_{H^s}$.
\end{theorem}

\begin{proposition}[Constraint Manifold Regularity]
\label{prop:constraint-regularity}
Under linear independence of $\{\nabla c_i\}$ and smoothness of $g_j$,
$\mathcal{C}_{\mathrm{eq}} = \{c_i = 0\}$ is a $C^\infty$ submanifold of
$\mathcal{M}$ of codimension $n_c$.
\end{proposition}

\begin{theorem}[Projection Existence and Uniqueness]
\label{thm:projection-exists}
For $\mathcal{C} \subset \mathcal{M}$ closed and convex, $\Pi_{\mathcal{C}}$
exists uniquely and satisfies
$\|\Pi_{\mathcal{C}}(x_1) - \Pi_{\mathcal{C}}(x_2)\|_{\mathcal{M}}
\leq \|x_1 - x_2\|_{\mathcal{M}}$ (nonexpansive).
\end{theorem}

\begin{remark}[CIIR Interpretation]
$\mathcal{C}$ encodes physical constraints (C); observables $O_j$ define the
interface manifold (I); feedback between instability and control constitutes
recursion (R). $\Pi_{\mathcal{C}}$ ensures constraint preservation,
underpinning $\|R_{\mathrm{ctrl}}\| > \|R_{\mathrm{instab}}\|$.
\end{remark}
"""

# =============================================================================
# FULL TECHNICAL CONTEXT — injected into every agent
# =============================================================================

RMHD_CIIR_DOCUMENT = r"""
PROJECTED CONSTRAINED RMHD / CIIR FRAMEWORK — FULL TECHNICAL CONTEXT

STATE SPACE:
  M = H^2_per(Omega) x H^2_per(Omega),  Omega = [0,L_x) x [0,L_y)
  State: (psi, omega) where psi = magnetic flux, omega = vorticity

RMHD EQUATIONS (2D incompressible):
  d_t omega + {phi, omega} = {psi, J_z} + nu nabla^2 omega
  d_t psi   + {phi, psi}   = eta nabla^2 psi
  J_z = -nabla^2 psi,   omega = nabla^2 phi
  Poisson bracket: {f,g} = d_x f d_y g - d_y f d_x g

HAMILTONIAN STRUCTURE:
  H = E_M + E_K = int(|nabla psi|^2/2 + |nabla phi|^2/2) dA
  d_t x = J (delta H / delta x),  x = (psi, omega)
  Casimirs: C_1 = int omega dA,  C_2 = int psi dA

CONSTRAINT MANIFOLD:
  C = {x in M : c_i(x)=0, g_j(x)<=0}
  Equality: solenoidality (auto), incompressibility (auto),
            Poisson antisymmetry, energy consistency
  Inequality: enstrophy bound, magnetic energy bound,
              spectral slope (plasmoid suppression),
              reconnection rate limit, current density bound

PROJECTION & CORE SYSTEM:
  Pi_C: M -> C  (metric projection in H^2 x H^2 norm)
  Phi_Dt: Strang-split Arakawa integrator  (O(Dt^2) accurate)
  Core equation: x_{k+1} = Pi_C(Phi_Dt(x_k, u_k))
  Control policy: u_k = pi(h(x_k), C)

CONTROL FORMULATION:
  x_dot = f(x) + g(x)u
  u = (E_drive, eta_eff, F_omega) in U_ad
  J[u] = int_0^T [alpha R + beta I_plasmoid + gamma P] dt
  Lyapunov: V = H + epsilon Z,  V_dot <= -c||u||^2 + control terms

CIIR FRAMEWORK:
  C (Constraints)  <-> constraint manifold C
  I (Interfaces)   <-> observation map h: M -> R^m
  R (Recursion)    <-> iteration x_{k+1} = Pi_C(Phi_Dt(x_k, u_k))
  Stability condition: ||R_ctrl|| > ||R_instab||

STRATIFIED PROJECTION (reconnection singularity resolution):
  C = C_reg union C_sing  (Whitney stratification)
  C_reg: open dense smooth submanifold (away from reconnection events)
  C_sing: closed lower-dimensional stratum where rank(dC_i) drops
          = locus of topology change in psi (Morse index changes)
  At reconnection time tau_rec:
    - X-point annihilation/creation (discrete topology change)
    - h_1 (X/O point count) becomes discontinuous
    - Pi_C loses smoothness precisely at C_sing
  Resolution:
    Pi_C smooth on C_reg (Theorem 5.2' holds)
    Pi_{C_eps} Lipschitz near C_sing with L_eps -> inf as eps -> 0
    Finite resistivity eta > 0 regularizes: topology changes over
    scale eps_rec = (eta tau_rec)^{1/2} — system approaches but
    never reaches C_sing at finite eta
  Revised results:
    Theorem 5.2': Pi_C is C^inf on C_reg; Lipschitz near C_sing
    Theorem 6.1': Exact preservation on C_reg; during reconnection
                  delta_C(t) <= K eps_rec + C Dt^2
    Theorem 7.1 replaced by ISS near C_sing:
      ||s(t) - s_target|| <= beta(||s_0||, t) + gamma(sup delta_C)
  Persistent topology observable (continuous h_1):
    h_1^eps(psi) = sum over persistence pairs (b_i,d_i) with
                   |d_i - b_i| > eps of smooth indicator
    Continuous in psi under W^{1,inf} topology

CONNECTION TO CIIR MONOGRAPH Ch.12:
  Fixed point x* <-> CRS-CIIR-Observer loop fixed point
  Lyapunov decay rate epsilon <-> qubit dephasing rate gamma_phi
    (formal structural analogy, not physical identity)
  Ramsey interferometry prediction: measurable if both rates share
  same scaling with their respective dissipation parameters

FALSIFIABILITY:
  SNR_min ~ 10^3 required for statistically significant detection
  Prediction 1: spectral rolloff at k* = (eta/nu)^{1/4} k_inj
  Prediction 2: reconnection rate v_rec suppressed to R_max / int J
  Prediction 3: fixed-point convergence k* scales as O(N^alpha), alpha<1
  Five-step protocol: baseline -> null -> signal -> artifacts -> verdict
"""

# =============================================================================
# NOTATION REGISTRY — from Agent 0 outline JSON, fully resolved
# =============================================================================

NOTATION_REGISTRY = r"""
CANONICAL NOTATION REGISTRY

DOMAIN AND GRIDS:
  Omega      = [0,L_x) x [0,L_y), periodic boundary conditions
  L_x, L_y   = domain lengths
  G          = discrete grid {(x_i, y_j): i=0..n_x-1, j=0..n_y-1}
  n_x, n_y   = grid point counts
  h          = grid spacing h = Dx = Dy
  N          = total gridpoints N = n_x x n_y

FUNCTION SPACES:
  H^s_per    = periodic Sobolev space of order s on Omega
  M          = state manifold H^2_per x H^2_per
  C          = constraint manifold C subset M
  C_reg      = regular stratum of C (smooth submanifold)
  C_sing     = singular stratum of C (topology-change locus)
  C_eps      = eps-regularization of C near C_sing
  TM         = tangent bundle of M
  T_s C      = tangent space to C at s
  ||.||_s    = Sobolev norm of order s

MHD STATE VARIABLES:
  psi        = magnetic flux function, B_perp = nabla psi x z_hat
  omega      = out-of-plane vorticity, omega = nabla^2 phi
  phi        = stream function, v_perp = z_hat x nabla phi
  J_z        = current density, J_z = -nabla^2 psi
  B_perp     = in-plane magnetic field (B_x, B_y)
  v_perp     = in-plane velocity (v_x, v_y)

PHYSICAL PARAMETERS:
  eta        = resistivity (Ohmic diffusivity)
  nu         = kinematic viscosity
  S          = Lundquist number S = L V_A / eta
  S_c        = critical Lundquist for plasmoid onset, S_c ~ 10^4
  V_A        = Alfven velocity
  eps_rec    = reconnection layer thickness ~ (eta tau_rec)^{1/2}
  tau_rec    = reconnection event time (Morse index change time)

ENERGETICS:
  H          = total energy H = E_M + E_K
  E_M        = magnetic energy int |nabla psi|^2 / 2 dV
  E_K        = kinetic energy int |v_perp|^2 / 2 dV
  P_Omega    = Ohmic dissipation int eta J_z^2 dV
  P_nu       = viscous dissipation int nu omega^2 dV
  Z          = enstrophy int omega^2 dV
  K_M        = magnetic helicity (Casimir)

OPERATORS:
  {f,g}      = Poisson bracket d_x f d_y g - d_y f d_x g
  nabla^2    = Laplacian
  L_h        = discrete Laplacian on grid G
  J_Ar(f,g)  = Arakawa discrete bracket J_1 + J_2 + J_3
  Pi_C       = projection operator onto C
  Pi_{C_eps} = regularized projection near C_sing
  h          = observation map h: M -> R^m
  h_1^eps    = persistent topology observable (continuous X/O count)
  Str(psi)   = Reeb graph (topological skeleton) of psi

CONTROL:
  u          = control vector (E_drive, eta_eff, F_omega)
  E_drive    = reconnection electric field forcing
  eta_eff    = spatially-varying resistivity
  F_omega    = vorticity forcing
  U_ad       = admissible control set
  w_target   = target localization mask in [0,1]
  sigma      = Gaussian width for w_target
  pi         = control policy pi: R^m x C -> U

OBJECTIVE:
  J[u]       = int_0^T [alpha R + beta I_plasmoid + gamma P] dt
  alpha,beta,gamma = cost weights
  R          = reconnection localization metric
  I_plasmoid = plasmoid spectral indicator (k-band energy sum)
  P          = energy release rate P_Omega

CIIR:
  C_i        = constraint functions C_i: M -> R
  O_j        = observable operators O_j: M -> R^m
  R_instab   = instability recursion norm (plasmoid band energy)
  R_ctrl     = control recursion norm
  Sigma(t)   = CIIR snapshot (C, I, R, topology, spectrum)
  F1-F5      = failure modes

TIME INTEGRATION:
  Dt         = timestep
  s_n        = state at timestep n
  k_1, k_2   = Runge-Kutta stages
  F          = unconstrained RMHD RHS operator
  F_proj     = projected RHS: Pi_C circ F
  Phi_Dt     = Strang-split Arakawa integrator
  CFL        = Courant-Friedrichs-Lewy number

STABILITY:
  V          = Lyapunov function V = H + eps Z
  lambda_i   = Lyapunov exponents
  delta_C    = constraint violation max_i |C_i(s)|
  s_target   = target equilibrium
  eps_0      = model robustness threshold
  beta       = class KL function (ISS decay)
  gamma      = class K function (ISS gain)

SPECTRAL:
  k          = wavenumber |k| = sqrt(k_x^2 + k_y^2)
  k_lo, k_hi = plasmoid wavenumber band
  E_mag(k)   = magnetic energy spectrum k^2 |psi_hat(k)|^2
  psi_hat    = Fourier transform of psi

TOPOLOGY:
  X-point    = saddle of psi (nabla psi = 0, det Hess psi < 0)
  O-point    = extremum of psi (nabla psi = 0, det Hess psi > 0)
  R_hat      = local reconnection rate |E_z|/B

FALSIFIABILITY:
  SNR        = |<Dpsi_controlled>| / sigma_noise
  SNR_min    = minimum SNR threshold ~ 10^3
  tau_Alfven = Alfven crossing time L/V_A
  tau_ctrl   = controller latency
  tau_instab = instability growth time
"""

# =============================================================================
# AGENT 0 — ORCHESTRATOR
# =============================================================================

AGENT_0_SYSTEM = f"""You are the lead architect for a rigorous mathematical physics whitepaper.
Role: DECOMPOSITION ONLY. Do not write prose sections.

Output valid JSON only. Schema:
{{
  "title": "string",
  "sections": [
    {{
      "number": int,
      "title": "string",
      "agent": "Agent N",
      "core_objects": ["..."],
      "key_results": ["theorem/proposition names"],
      "depends_on_sections": [int]
    }}
  ],
  "open_problems": ["..."]
}}

The notation registry is injected separately. Do not redefine symbols."""

AGENT_0_USER = f"""Technical document:
{RMHD_CIIR_DOCUMENT}

Note: Section 1 (Mathematical Foundations) has already been produced and
is injected directly. Your outline should reflect 10 sections total
(Sections 1-10), with Section 1 marked as complete.

Produce the section outline JSON."""

# =============================================================================
# AGENT 1 — MATHEMATICAL FOUNDATIONS
# (Section 1 already produced — agent writes extended appendix material)
# =============================================================================

AGENT_1_SYSTEM = f"""You are a functional analyst writing for a rigorous mathematical physics
whitepaper. Output LaTeX section body only (no preamble).
Every claim: definition, lemma, proposition, theorem, proof, or remark.
No informal prose without a formal counterpart.

NOTATION REGISTRY (mandatory — use only these symbols):
{NOTATION_REGISTRY}"""

AGENT_1_USER = f"""Context:
{RMHD_CIIR_DOCUMENT}

Section 1 (core) has already been produced:
{SECTION_1_LATEX}

Write Section 1 Appendix: "Stratification of the Constraint Manifold and
Projection Singularities at Reconnection Events."

This section resolves the projection singularity identified in the framework:
Pi_C loses smoothness exactly at reconnection events, where C_sing is the
locus of topology change. Produce all of the following.

1. WHITNEY STRATIFICATION OF C
Define:
  C_reg = {{x in C : rank(dC_i(x)) = n_c everywhere, Morse index of psi constant}}
  C_sing = C \\ C_reg = {{x in C : rank(dC_i) drops, or Morse index of psi changes}}

\\begin{{definition}}[Whitney Stratification of Constraint Manifold]
State C = C_reg union C_sing as a Whitney stratified space.
Define C_reg as an open dense smooth submanifold of C.
Define C_sing as the closed singular stratum.
\\end{{definition}}

\\begin{{proposition}}[Codimension of C_sing]
State: in 2D RMHD with periodic BC and state space M = H^2_per x H^2_per,
C_sing has codimension 1 in C.
Argument: a generic path in M crosses C_sing transversally (one condition:
det Hess psi = 0 at a critical point of psi) — this is one real equation,
hence codimension 1.
\\end{{proposition}}

2. REGULARIZED PROJECTION NEAR C_sing
\\begin{{definition}}[Eps-Regularized Constraint Set]
Define C_eps as the constraint set with topology-count constraint h_1
replaced by the persistent topology observable:
  h_1^eps(psi) = sum_{{(b_i,d_i) persistence pairs, |d_i - b_i| > eps}}
                 w_eps(d_i - b_i)
where w_eps is a smooth cutoff. This replaces discrete X/O counting
with a C^inf function of psi under the W^{{1,inf}} topology.
\\end{{definition}}

\\begin{{theorem}}[Revised Projection Smoothness — Theorem 5.2']
(a) Pi_C is C^inf on C_reg.
(b) On N_eps(C_sing) (eps-neighborhood of singular stratum),
    Pi_{{C_eps}} is Lipschitz continuous with constant L_eps.
(c) Projection error: ||Pi_{{C_eps}}(s) - Pi_C(s)||_M <= K eps
    for K depending only on the geometry of C near C_sing.
(d) As eps -> 0: L_eps -> inf, recovering the singularity of Pi_C at C_sing.
\\end{{theorem}}

\\begin{{proof}}
(a): on C_reg, rank(dC_i) is constant so the implicit function theorem
applies globally; Pi_C inherits smoothness of C_i (standard argument,
Rockafellar-Wets 1998, Theorem 10.58).
(b): on N_eps(C_sing), C_eps is defined by smooth approximants of C_i;
Lipschitz continuity follows from the fact that smooth constraint
functions with bounded Hessian yield Lipschitz projections (use
Moreau envelope regularization: Pi_{{C_eps}}(s) =
argmin_y [||s-y||^2/2 + delta_{{C_eps}}(y)] where delta_{{C_eps}} is
the Moreau envelope of the indicator of C_eps).
(c): the error bound follows from persistence stability:
||h_1^eps - h_1||_{{W^{{1,inf}}}} = O(eps) by the bottleneck stability
theorem (Edelsbrunner-Harer 2010, Theorem VIII.4.1). QED.
\\end{{proof}}

3. REVISED CONSTRAINT PRESERVATION THEOREM
\\begin{{theorem}}[Approximate Constraint Preservation — Theorem 6.1']
Let [tau_rec - delta, tau_rec + delta] denote a reconnection interval
with delta ~ eps_rec / V_A, eps_rec = (eta tau_rec)^{{1/2}}.
(a) Outside reconnection intervals: s(t) in C_reg exactly.
    (Original Theorem 6.1 holds on C_reg.)
(b) During reconnection interval:
    delta_C(t) <= K eps_rec + C_num Dt^2
    The constraint violation is controlled by the reconnection layer
    thickness, not by pure numerical error.
(c) Post-reconnection recovery:
    delta_C(t) -> 0 within time t_return ~ eps_rec^2 / eta after tau_rec.
\\end{{theorem}}

\\begin{{proof}}
At finite resistivity eta > 0, RMHD reconnection is Sweet-Parker type:
the topology of psi changes continuously through a bifurcation smoothed
over scale eps_rec = (eta tau_rec)^{{1/2}} (Biskamp 2000, Chapter 3).
Therefore C_sing is approached but never reached in finite time at eta > 0.
The system passes through N_{{eps_rec}}(C_sing), where Pi_{{C_{{eps_rec}}}}
is well-defined with error bound K eps_rec (part b).
Post-reconnection: once the current sheet disperses (t >> tau_rec),
eps_rec -> 0 and the state returns to C_reg (part c). QED.
\\end{{proof}}

4. IMPACT ON STABILITY — ISS REPLACEMENT
\\begin{{theorem}}[Input-to-State Stability near C_sing]
On C_reg, the main stability Theorem 7.1 holds:
  ||R_ctrl|| > ||R_instab|| implies exponential stability.
Near C_sing, exponential stability is replaced by input-to-state stability:
  ||s(t) - s_target||_M <= beta(||s_0||_M, t) + gamma(sup_{{tau<=t}} delta_C(tau))
where beta is class KL and gamma is class K (Sontag 1989).
The ISS gain gamma is bounded: gamma(r) <= L_ISS r for r = K eps_rec + C Dt^2,
so stability degrades gracefully through reconnection and recovers afterward.
\\end{{theorem}}

5. PHYSICAL INTERPRETATION
\\begin{{remark}}[Physical Meaning of Stratification]
C_sing is not a failure of the framework. It is the mathematical signature
of a physical phase transition in field-line topology. The projection
operator Pi_C loses smoothness at C_sing for the same reason that phase
transitions are non-analytic in thermodynamics: the system crosses a
structural boundary. Finite resistivity eta > 0 is not an approximation —
it is the physical regularization that makes real reconnection possible
and prevents the system from reaching C_sing exactly. The eps_rec layer
is the physical reconnection current sheet.
\\end{{remark}}

Cite: Edelsbrunner & Harer, Computational Topology (2010) — persistence stability.
Cite: de Silva, Morozov, Vejdemo-Johansson (2011) — Reeb graph stability.
Cite: Rockafellar & Wets, Variational Analysis (1998) — Lipschitz projection.
Cite: Sontag (1989) Syst. Control Lett. — ISS framework.
Cite: Biskamp, Magnetic Reconnection in Plasmas (2000) — Sweet-Parker regularization."""

# =============================================================================
# AGENT 2 — RMHD PHYSICS & HAMILTONIAN STRUCTURE
# =============================================================================

AGENT_2_SYSTEM = f"""You are a mathematical physicist writing a section of a rigorous
whitepaper on 2D incompressible reduced MHD. Output LaTeX section body only.
Every physical claim must have a derivation or explicit citation.

NOTATION REGISTRY (mandatory):
{NOTATION_REGISTRY}"""

AGENT_2_USER = f"""Context:
{RMHD_CIIR_DOCUMENT}

Write Section 2: "Reduced Resistive MHD: Hamiltonian Structure and Conservation Laws."

1. State RMHD equations in strong form:
   d_t omega + {{phi, omega}} = {{psi, J_z}} + nu nabla^2 omega
   d_t psi   + {{phi, psi}}   = eta nabla^2 psi
   J_z = -nabla^2 psi,  omega = nabla^2 phi
   with full definition of {{f,g}} = d_x f d_y g - d_y f d_x g.

2. Derive Hamiltonian structure:
   d_t x = J (delta H / delta x), x = (psi, omega)
   H = int(|nabla psi|^2/2 + |nabla phi|^2/2) dA
   Compute delta H/delta psi = J_z and delta H/delta omega = phi explicitly.
   State the Lie-Poisson structure on the dual of a semidirect product algebra
   (Morrison-Greene 1980).

\\begin{{theorem}}[Hamiltonian Structure of RMHD — Theorem 2.1]
The non-dissipative RMHD equations (nu=eta=0) have Lie-Poisson Hamiltonian
structure d_t x = J (delta H / delta x). The dissipative system is a
perturbation: d_t x = J delta H/delta x + D(x) where D encodes eta, nu terms.
\\end{{theorem}}

3. Prove Casimir invariants:
   C_1 = int omega dA (total vorticity)
   C_2 = int psi dA (total flux)
   via {{C_i, H}} = 0 in the Lie-Poisson bracket.
   Show Casimirs are conserved only at nu=eta=0; state implications.

\\begin{{theorem}}[Energy Evolution — Theorem 2.2]
dH/dt = -P_Omega - P_nu + control power
where P_Omega = int eta J_z^2 dV, P_nu = int nu omega^2 dV.
\\end{{theorem}}

4. Lundquist number and plasmoid onset:
\\begin{{theorem}}[Plasmoid Onset — Theorem 2.4]
The Lundquist number S = L V_A / eta governs tearing instability.
For S > S_c ~ 10^4 (Furth-Killeen-Rosenbluth threshold), the current
sheet undergoes plasmoid cascade with growth rate gamma ~ S^{{1/4}} / tau_A.
State the FKR dispersion relation: gamma tau_A ~ (k delta)^2/3 (eta/eta_crit)^{{1/3}}
where delta is current sheet half-width.
\\end{{theorem}}

5. Remark: dissipation breaks Hamiltonian structure. Lyapunov stability
analysis in Section 7 must use V_dot <= -epsilon||x-x_bar||^2 (not V_dot=0).
The projected controller must not add spurious dissipation beyond eta, nu terms.

Cite: Morrison & Greene (1980) Phys. Rev. Lett. 45:790.
Cite: Holm, Marsden, Ratiu, Weinstein (1985) Phys. Rep. 123:1.
Cite: Biskamp (2000) Magnetic Reconnection in Plasmas, Cambridge.
Cite: Furth, Killeen, Rosenbluth (1963) Phys. Fluids 6:459."""

# =============================================================================
# AGENT 3 — ARAKAWA DISCRETIZATION & ALGORITHMS
# =============================================================================

AGENT_3_SYSTEM = f"""You are a numerical analyst writing sections on structure-preserving
discretization and algorithms. Output LaTeX section body only.
All algorithms in numbered algorithm environments. All claims proved or cited.

NOTATION REGISTRY (mandatory):
{NOTATION_REGISTRY}"""

AGENT_3_USER = f"""Context:
{RMHD_CIIR_DOCUMENT}

Write Sections 3 and 8.

SECTION 3: "Structure-Preserving Discretization: Arakawa Schemes"

1. Define the Arakawa bracket on uniform grid G with spacing h:
   J_1(f,g)_{{ij}} = (1/4h^2)[
     (f_{{i+1,j}}-f_{{i-1,j}})(g_{{i,j+1}}-g_{{i,j-1}})
    -(f_{{i,j+1}}-f_{{i,j-1}})(g_{{i+1,j}}-g_{{i-1,j}})]
   Write J_2 and J_3 stencils in full (standard Arakawa 1966 forms).
   Define J_Ar = (J_1 + J_2 + J_3)/3.

\\begin{{theorem}}[Arakawa Conservation — Theorem 3.1]
J_Ar simultaneously conserves:
  (a) Discrete energy: sum_{{ij}} psi_{{ij}} J_Ar(phi,omega)_{{ij}} = 0
  (b) Discrete enstrophy: sum_{{ij}} omega_{{ij}} J_Ar(phi,omega)_{{ij}} = 0
  (c) Discrete mean square vorticity: sum_{{ij}} omega^2_{{ij}} conserved
Proof: show each J_k conserves one of these; J_Ar as average conserves all.
\\end{{theorem}}

2. FFT Poisson solver:
   omega = nabla^2 phi => phi_hat = -omega_hat / k^2 in Fourier space.
\\begin{{proposition}}[Spectral Accuracy — Proposition 3.3]
FFT inversion achieves O(h^inf) accuracy for smooth periodic data.
State: computational cost O(N log N) per solve.
\\end{{proposition}}

3. CFL stability:
\\begin{{lemma}}[CFL Condition — Lemma 3.4]
Dt <= min(h/|v|_max, h^2/(4(eta+nu)))
First term: advective CFL. Second: diffusive CFL.
\\end{{lemma}}

SECTION 8: "Numerical Algorithms and Implementation"

\\begin{{algorithm}}[ReconnectionController Main Loop]
Input: initial state (psi_0, omega_0), control weights alpha,beta,gamma,
       target mask w_target, final time T, timestep Dt
Output: trajectory {{s_n}}, control history {{u_n}}, diagnostics

1. Initialize: s_0 = (psi_0, omega_0), check s_0 in C
2. For n = 0, 1, ..., T/Dt - 1:
   a. Compute observables: y_n = h(s_n) = (h_1^eps, h_2, h_3)
   b. Compute control: u_n = pi(y_n, C) using gradient law:
        E_drive = A_E w_target sign(J_z)
        eta_eff = eta_boost max(0, I_plasmoid - I_threshold) w_target
        F_omega = A_F w_target nabla^2 phi
   c. Strang split step:
        s_* = S^B_{{Dt/2}}(s_n, u_n)     [half-step dissipation, CN]
        s_** = S^A_Dt(s_*, u_n)          [full Arakawa + FFT step]
        s_tilde = S^B_{{Dt/2}}(s_**, u_n) [half-step dissipation, CN]
   d. Project: s_{{n+1}} = Pi_C(s_tilde)
      [apply Pi_C after full Strang step, not within substeps]
   e. Diagnostics: record H, Z, delta_C, topology_snapshot, spectrum
3. Return {{s_n}}, {{u_n}}, diagnostics
\\end{{algorithm}}

\\begin{{proposition}}[Sign-Following Policy — Proposition 8.2]
The control law E_drive = A_E w_target sign(J_z) avoids flux reversal:
it drives reconnection in the direction of existing current, never opposing it.
\\end{{proposition}}

\\begin{{lemma}}[Projection Ordering Theorem — Lemma 8.3]
Applying Pi_C after the complete Strang step (not within substeps)
minimizes the projection correction ||Pi_C(s_tilde) - s_tilde||_M to O(Dt^2).
Proof sketch: within-step projection introduces O(Dt) perturbations that
accumulate; after the full step the Strang error is already O(Dt^2),
so projection corrects a smaller residual.
\\end{{lemma}}

Cite: Arakawa (1966) J. Comput. Phys. 1:119.
Cite: Blanes & Casas, Geometric Numerical Integration (2016).
Cite: Hesthaven, Gottlieb, Gottlieb, Spectral Methods (2007)."""

# =============================================================================
# AGENT 4 — CONTROL THEORY
# =============================================================================

AGENT_4_SYSTEM = f"""You are a control theorist writing on nonlinear control of
infinite-dimensional PDE systems. Output LaTeX section body only.
State clearly where infinite-dimensional complications arise and how
Galerkin truncation recovers tractability.

NOTATION REGISTRY (mandatory):
{NOTATION_REGISTRY}"""

AGENT_4_USER = f"""Context:
{RMHD_CIIR_DOCUMENT}

Write Section 4: "Control-Theoretic Formulation and Optimal Control."

1. Nonlinear control system:
   x_dot = f(x) + g(x)u,  x in M, u in U_ad
   f(x) = (-{{phi,psi}} + eta nabla^2 psi,  -{{phi,omega}} + {{psi,J_z}} + nu nabla^2 omega)
   g(x)u = linear influence of u = (E_drive, eta_eff, F_omega).
   State: nonlinear control system on infinite-dimensional manifold M
   in the sense of Isidori (1995).

2. Observation map:
   y = h(x) = (h_1^eps(x), h_2(x), h_3(x)) in R^m
   h_1^eps: persistent topology observable (continuous X/O count, see Section 1 Appendix)
   h_2: spectral energy shells E_mag(k) = k^2 |psi_hat(k)|^2
   h_3: reconnection rate proxy R_hat = int |eta J_z| w_target dV

3. Optimal control problem:
\\begin{{theorem}}[Existence of Optimal Control — Theorem 4.2]
Under: (i) U_ad weakly closed, bounded; (ii) f,g continuous; (iii) J[u] coercive;
there exists u* in U_ad minimizing J[u] = int_0^T [alpha R + beta I_plasmoid + gamma P] dt.
Proof: standard direct method in calculus of variations (Lions 1971, Chapter 1).
\\end{{theorem}}

4. First-order optimality conditions:
\\begin{{theorem}}[Adjoint Equations — Theorem 4.3]
At optimum u*, the adjoint variables lambda_psi, lambda_omega satisfy:
  -d_t lambda_psi  = (delta F/delta psi)^* lambda + delta L/delta psi
  -d_t lambda_omega = (delta F/delta omega)^* lambda + delta L/delta omega
  lambda(T) = 0  (terminal condition)
where L is the Lagrangian integrand of J[u].
Optimality condition: delta J/delta u = g(x)^* lambda + delta L/delta u = 0.
\\end{{theorem}}

5. Linearized controllability:
Linearize at Harris sheet equilibrium x_bar:
  delta x_dot = A delta x + B u
Kalman condition (Galerkin truncation to N_g modes):
  rank[B, AB, A^2 B, ..., A^{{N_g-1}} B] = N_g
\\begin{{theorem}}[Local Controllability — Theorem 4.1]
The RMHD system with control u = (E_drive, eta_eff, F_omega) is locally
controllable near the Harris sheet equilibrium psi_bar = ln(cosh(x/lambda_cs))
in the Galerkin truncation to N_g <= 4 * (modal multiplicity) modes.
\\end{{theorem}}

6. Lyapunov stability:
\\begin{{theorem}}[Lyapunov Stability — Theorem 4.4 / Proposition 4.4]
Let V = H + eps Z with eps > 0 small.
Hypotheses: (H1) nu, eta > 0; (H2) ||u||_{{U_ad}} <= M bounded;
(H3) control law satisfies dissipation inequality:
  <g(x) pi(h(x),C), nabla V(x)> <= kappa ||x - x_bar||^2_M, kappa < c_diss
where c_diss arises from the nu, eta dissipation in d_t V.
Conclusion: V_dot <= -(c_diss - kappa) ||x - x_bar||^2_M,
so x_bar is locally exponentially stable.
Warning: global stability cannot be claimed — V may not be radially unbounded.
\\end{{theorem}}

Cite: Isidori, Nonlinear Control Systems (1995).
Cite: Lions, Optimal Control of Systems Governed by PDEs (1971).
Cite: Curtain & Zwart, Infinite-Dimensional Linear Systems Theory (1995)."""

# =============================================================================
# AGENT 5 — CIIR SYNTHESIS
# =============================================================================

AGENT_5_SYSTEM = f"""You are a mathematical systems theorist writing the synthesis section
of a rigorous whitepaper. Unify all prior results into the CIIR framework.
Output LaTeX section body only. Reference prior sections explicitly by number.
Do not re-derive; synthesize and prove new unified results only.

NOTATION REGISTRY (mandatory):
{NOTATION_REGISTRY}"""

AGENT_5_USER = f"""Context:
{RMHD_CIIR_DOCUMENT}

Prior sections summary:
- Section 1 + Appendix: M = H^2_per x H^2_per, C = C_reg union C_sing
  (Whitney stratification), Pi_C smooth on C_reg, Lipschitz near C_sing,
  error bound K eps_rec. ISS stability near C_sing.
- Section 2: Hamiltonian structure, Casimirs, FKR plasmoid onset S_c ~ 10^4
- Section 3: Phi_Dt = Strang-split Arakawa, O(Dt^2), projection ordering theorem
- Section 4: x_dot = f(x)+g(x)u, Lyapunov theorem (local exponential stability),
  adjoint-based optimal control, local controllability near Harris sheet

Write Sections 5 and 6.

SECTION 5: "CIIR Constraint Projection System"

\\begin{{definition}}[CIIR Constraint Set]
C = {{C_1, ..., C_n}} with C_i: M -> R smooth constraint functions.
Stratified as C = C_reg union C_sing (Section 1 Appendix).
\\end{{definition}}

\\begin{{definition}}[Interface Map]
Interface map Pi: M -> I where I is the interface manifold.
Pi(psi, omega) = (X-points, O-points via h_1^eps, J-sheets, E_mag(k))
Note: X/O counting uses persistent topology observable h_1^eps (Section 1
Appendix) to ensure continuity through reconnection events.
\\end{{definition}}

\\begin{{theorem}}[Constraint Preservation — Theorem 5.1]
If s_0 in C_reg and dynamics projected by Pi_C on C_reg, then s(t) in C_reg exactly.
During reconnection intervals [tau_rec - delta, tau_rec + delta]:
delta_C(t) <= K eps_rec + C_num Dt^2 (Theorem 6.1', Section 1 Appendix).
\\end{{theorem}}

\\begin{{theorem}}[Projection Regularity — Theorem 5.2 (revised to 5.2')]
(a) Pi_C is C^inf on C_reg (smooth stratum).
(b) Pi_{{C_eps}} is Lipschitz on N_eps(C_sing) with error O(eps).
(Proof: Section 1 Appendix, Theorem 5.2'.)
\\end{{theorem}}

\\begin{{definition}}[Failure Modes F1-F5]
F1: Unobserved instability — instability in ker(dh), undetectable via y.
F2: Controller lag — tau_ctrl > tau_instab, controller too slow.
F3: Model-experiment mismatch — ||C_model - C_measured|| > eps_0.
F4: Actuator saturation — ||u||_{{U_ad}} at boundary of U_ad.
F5: Plasmoid cascade — S > S_c, instability exceeds control bandwidth.
Each characterized by an inequality on (tau_ctrl, tau_instab, dim ker dh).
\\end{{definition}}

SECTION 6: "Projected Constrained Dynamics and Time Integration"

Central boxed result:

\\begin{{equation}}\\boxed{{
  x_{{k+1}} = \Pi_{{\mathcal{{C}}}}\!\left(\Phi_{{\Delta t}}(x_k,\, u_k)\right),
  \quad u_k = \pi(h(x_k),\, \mathcal{{C}})
}}\\end{{equation}}

Name: the CIIR Projected RMHD System (CPRS).

\\begin{{theorem}}[Well-Posedness of CPRS — Theorem 6.1 / revised 6.1']
Under:
  (A1) C nonempty, closed in M (Section 1, Proposition 1.3)
  (A2) Phi_Dt Lipschitz with constant L_Phi near x_bar
  (A3) pi: R^m x C -> U bounded and locally Lipschitz
(a) Outside reconnection intervals: {{x_k}} well-defined, s(t) in C_reg exactly.
(b) During reconnection: delta_C(t) <= K eps_rec + C_num Dt^2.
(c) If L_Phi * L_pi < 1 (contraction): {{x_k}} converges to fixed point x*
    satisfying x* = Pi_C(Phi_Dt(x*, pi(h(x*), C))).
\\end{{theorem}}

\\begin{{proof}}
Banach fixed-point theorem on the composition Pi_C circ Phi_Dt(., pi(h(.),C)).
On C_reg this is a contraction by (A2)-(A3) and Lipschitz continuity of Pi_C
(Theorem 5.2', part a). Near C_sing use Theorem 6.1' (b). QED.
\\end{{proof}}

\\begin{{definition}}[C-I-R Interpretation Map]
C (Constraints) <-> C subset M, enforced by Pi_C
I (Interfaces)  <-> observation map h: M -> R^m, producing y_k
R (Recursion)   <-> iteration x_{{k+1}} = Pi_C(Phi_Dt(x_k, u_k))
The C-I-R triad is now a mathematically precise object.
\\end{{definition}}

\\begin{{proposition}}[Hard vs. Soft Projection — Proposition 6.3]
Hard projection Pi^hard_C: enforces invariants c_i = 0 only
  (solenoidality, incompressibility, Poisson antisymmetry).
  Preserves Hamiltonian structure at discrete level.
Soft projection Pi^soft_C: enforces stability envelope g_j <= 0.
  Introduces non-Hamiltonian dissipation O(||g_j(x_tilde)||) per step.
\\end{{proposition}}

\\begin{{remark}}[Connection to CIIR Monograph Ch. 12]
Fixed point x* is an instance of the CRS-CIIR-Observer loop fixed point
(Ch. 12, Theorem 12.3: projected dynamics closed under composition).
Formal analogy: Lyapunov decay rate eps <-> qubit dephasing gamma_phi.
Both are fixed-point convergence rates; sharing dissipation-parameter scaling
would strengthen but not establish physical identity.
\\end{{remark}}

Cite: Nagurney & Zhang, Projected Dynamical Systems (1996).
Cite: Henry, Geometric Theory of Semilinear Parabolic Equations (1981)."""

# =============================================================================
# AGENT 6 — RED TEAM PEER REVIEWER
# =============================================================================

AGENT_6_SYSTEM = """You are an anonymous reviewer for Communications in Mathematical Physics.
Rigorous, adversarial, constructive. Deep expertise in: infinite-dimensional
dynamical systems, MHD, nonlinear control, numerical analysis of PDEs,
and stratified geometry.

Per issue: Section / Flaw Type / Severity (Fatal|Major|Minor) / Description / Required Fix.
Do not soften. End with: OVERALL RECOMMENDATION and one-sentence justification."""

def build_agent_6_user(sections: dict) -> str:
    assembled = "\n\n".join(
        f"=== {k} ===\n{v}" for k, v in sections.items()
    )
    return f"""Review the following whitepaper draft:

{assembled}

Check rigorously for:
1. UNDEFINED NOTATION — symbol used before definition
2. MISSING PROOF STEPS — theorems without proof, sketch, or citation
3. OVERCLAIMING — conclusions stronger than proof supports
4. PHYSICAL UNJUSTIFIABILITY — assumptions false for turbulent MHD
   (specifically: convexity of C, Lipschitz on Phi_Dt, bounded pi,
    the ISS gain bound near C_sing, the codimension-1 claim for C_sing)
5. DIMENSIONAL/NORM INCONSISTENCY
6. CIRCULAR REASONING
7. CONTROLLABILITY GAPS — Kalman rank condition in infinite dimensions
8. PROJECTION NONUNIQUENESS — set-valued Pi_C treated as single-valued
9. DISCRETIZATION-CONTINUUM MISMATCH
10. ANALOGY AS IDENTITY — especially qubit/CPRS Lyapunov/dephasing connection
11. STRATIFICATION GAPS — Whitney conditions not verified for RMHD C_sing
12. ISS CLAIM — verify that class KL/K functions are explicitly constructed,
    not merely asserted

Output format per issue:
SECTION: [name]
FLAW TYPE: [from list]
SEVERITY: [Fatal/Major/Minor]
DESCRIPTION: [precise]
REQUIRED FIX: [precise instruction]

OVERALL RECOMMENDATION: [Accept/Major Revision/Reject]
JUSTIFICATION: [one sentence]"""

# =============================================================================
# AGENT 7 — EXPERIMENTAL FALSIFIABILITY
# =============================================================================

AGENT_7_SYSTEM = f"""You are an experimental and computational physicist writing the
falsifiability section of a rigorous whitepaper. Precise and conditional.
Output LaTeX section body only. Use: "the theory predicts...",
"this would be refuted if...", "confirmation requires...". No overclaiming.

NOTATION REGISTRY (mandatory):
{NOTATION_REGISTRY}"""

AGENT_7_USER = f"""Context:
{RMHD_CIIR_DOCUMENT}

Write Sections 9 and 10.

SECTION 9: "Experimental Falsifiability and Connection to CIIR Monograph Ch. 12"

1. Five-step falsification protocol:
\\begin{{enumerate}}
\\item[S1] Baseline: run uncontrolled RMHD at S > S_c. Measure R_hat, E_mag(k),
           X/O topology. Establish noise floor sigma_noise.
\\item[S2] Null hypothesis: run CPRS with u = 0 (no control). Verify system
           behaves identically to baseline within sigma_noise.
\\item[S3] Signal injection: activate control u != 0 with target R_max < R_hat_baseline.
           Measure Dpsi_controlled = psi_controlled - psi_uncontrolled.
\\item[S4] Artifact screening: verify Dpsi_controlled not caused by:
           (a) numerical projection artifact (run at 2x resolution — must persist),
           (b) boundary condition artifact (run with different L_x, L_y),
           (c) initial condition sensitivity (run 10 randomized ICs).
\\item[S5] Verdict: CIIR prediction confirmed iff
           SNR = |<Dpsi_controlled>| / sigma_noise >= SNR_min ~ 10^3.
\\end{{enumerate}}

2. Prediction 1 — spectral signature of Pi_C:
The theory predicts a spectral rolloff steeper than k^{{-5/3}} for k > k*
where k* = (eta/nu)^{{1/4}} k_inj.
This would be refuted if: controlled and uncontrolled runs at identical
eta, nu, N show no slope change at k*.
Confirmation requires: N >= 512, eta/nu in [0.1, 10], measured at
t = 10 tau_Alfven.

3. Prediction 2 — reconnection rate suppression:
Sweet-Parker rate: v_rec = V_A (eta / (V_A L))^{{1/2}}.
Under CIIR control: v_rec_controlled = min(v_rec, R_max / int J dV).
For eta = 10^{{-3}}, V_A = 1, L = 1: v_rec ~ 0.032.
With R_max = 0.02: predict v_rec_controlled <= 0.022 (within 10% of bound).
This would be refuted if: v_rec_controlled > R_max / int J by more than 10%.

4. Prediction 3 — fixed-point convergence scaling:
Theory (Theorem 6.1 contraction): k* <= C log(1/eps) / log(1/r),
r = L_Phi L_pi < 1. Predict k* ~ O(N^alpha), alpha < 1.
Numerical test: measure k* for N in {{128, 256, 512, 1024}}, fit alpha.
This would be refuted if alpha >= 1.

5. Prediction 4 — stratified projection signature:
During reconnection events (identifiable by X-point count change in h_1),
the theory predicts delta_C spike of magnitude K eps_rec + C_num Dt^2,
followed by recovery within t_return ~ eps_rec^2 / eta.
This would be refuted if: delta_C does not recover after reconnection,
or if spike magnitude scales differently from eps_rec = (eta tau_rec)^{{1/2}}.

6. Connection to CIIR monograph Ch. 12:
\\begin{{theorem}}[Formal Closure — Connection 9.4]
Ch. 12 Theorem 12.3 states that projected dynamics are closed under composition.
The CPRS fixed point x* is an instance of this closure.
Formal mapping: Lyapunov decay V(x_k) ~ exp(-eps k Dt) <->
qubit coherence <sigma_x>(t) ~ exp(-gamma_phi t).
State: this is structural analogy. Independent experimental tests required:
both systems must exhibit the same scaling of decay rate with dissipation parameter.
\\end{{theorem}}

Minimal verification specification:
  Grid: N = 256x256, periodic BC
  IC: Harris sheet psi_0 = ln(cosh(x / 0.5))
  Parameters: eta = nu = 10^{{-3}}, V_A = 1
  Dt = 0.1 * CFL, T = 20 tau_Alfven
  Constraints: Z_max = 2||omega_0||^2, R_max = 0.03, I_plasmoid_max = ...
  Confirmation: delta_C < 10^{{-4}} for k > k*, k* < 500 steps

SECTION 10: "Conclusions and Open Questions"

Summarize CPRS framework. State C-I-R interpretation map (formal).

Open problems:
\\begin{{enumerate}}
\\item Whitney stratification: verify Whitney conditions (a) and (b)
  for C_sing in RMHD explicitly. Currently argued by codimension count only.
\\item Non-convexity of C and global projection: extend ISS bound to
  multiple sequential reconnection events.
\\item MPC tractability: adjoint-based optimal control vs. heuristic
  gradient policies at large N.
\\item 3D line-tied geometry (tokamak-relevant): extend to
  3D RMHD with guide field and line-tied boundary conditions.
\\item Hardware implementation: path to ITER-class facility with
  tau_ctrl < tau_instab requirement.
\\end{{enumerate}}

Cite: Sweet (1958), Parker (1957) for reconnection rate.
Cite: Biskamp (2000) for RMHD benchmarks.
Cite: Edelsbrunner & Harer (2010) for persistence stability."""

# =============================================================================
# AGENT 8 — FINAL INTEGRATOR
# =============================================================================

AGENT_8_SYSTEM = """You are the final editor of a mathematical physics whitepaper.
Role: INTEGRATION, CONSISTENCY, RESOLUTION.
Output: complete LaTeX document, compilable with pdflatex.
Use \\documentclass{amsart}.
Do not suppress criticisms — unresolvable fatal flaws become
\\begin{remark}[Open Problem] environments."""

def build_agent_8_user(outline: str, sections: dict, review: str, sec9_10: str) -> str:
    all_secs = "\n\n".join(f"% === {k} ===\n{v}" for k, v in sections.items())
    return f"""OUTLINE: {outline}

NOTATION REGISTRY:
{NOTATION_REGISTRY}

SECTION DRAFTS:
{all_secs}

SECTIONS 9-10:
{sec9_10}

PEER REVIEW:
{review}

TASKS:
1. Resolve all Fatal and Major review issues.
   Resolved: add comment % RESOLVED: [description]
   Unresolvable: add \\begin{{remark}}[Limitation] ... \\end{{remark}}

2. Write Abstract (250 words max):
   - Problem: RMHD control without formal constraint enforcement
   - Contribution: CIIR Projected RMHD System (CPRS)
   - Central results: Theorem 6.1 (well-posedness), Theorem 5.2'
     (stratified projection), ISS stability near C_sing
   - Experimental predictions: spectral rolloff at k*, reconnection
     suppression, delta_C spike-and-recovery at reconnection events
   - Connection to CIIR monograph Ch. 12

3. Write Section 1 Introduction (600 words):
   Motivate, state prior work, list contributions, state organization.

4. Assemble complete LaTeX with:
   \\documentclass{{amsart}}
   \\usepackage{{amsmath,amsthm,amssymb,mathtools,algorithm,algpseudocode}}
   \\newtheorem environments for theorem, lemma, proposition,
     definition, remark (all sharing counter [section])
   Full \\bibliography in \\bibitem format.

Output the complete LaTeX source."""

# =============================================================================
# HELPERS
# =============================================================================

def call_agent(system_prompt: str, user_message: str, max_tokens: int = 4096) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

# =============================================================================
# ORCHESTRATION RUNNER
# =============================================================================

def _strip_code_fence(text: str) -> str:
    """Remove optional ```json or ``` fences from model output."""
    text = text.strip()
    text = re.sub(r'^```[a-z]*\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def run_pipeline(model=None, max_workers=5,
                 out_json="whitepaper_v2_output.json",
                 out_tex="whitepaper_v2.tex"):
    global MODEL

    # Friendly error for missing API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it with:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    if model is not None:
        MODEL = model

    print("=" * 70)
    print("RMHD/CIIR WHITEPAPER v2 — STRATIFIED PROJECTION RESOLUTION")
    print("=" * 70)

    # Phase 1: Agent 0
    print("\n[PHASE 1] Agent 0: Outline...")
    outline_raw = call_agent(AGENT_0_SYSTEM, AGENT_0_USER, max_tokens=2048)
    try:
        clean = _strip_code_fence(outline_raw)
        outline_str = json.dumps(json.loads(clean), indent=2)
    except Exception:
        outline_str = outline_raw
    print("  ✓ Outline complete")

    # Phase 2: Agents 1-5 parallel
    print("\n[PHASE 2] Agents 1-5: Parallel section authoring...")
    agent_calls = [
        (AGENT_1_SYSTEM, AGENT_1_USER, 5000),
        (AGENT_2_SYSTEM, AGENT_2_USER, 4096),
        (AGENT_3_SYSTEM, AGENT_3_USER, 4096),
        (AGENT_4_SYSTEM, AGENT_4_USER, 4096),
        (AGENT_5_SYSTEM, AGENT_5_USER, 4096),
    ]

    async def _async_parallel():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            tasks = [
                loop.run_in_executor(pool, call_agent, sys_p, usr, tok)
                for sys_p, usr, tok in agent_calls
            ]
            return await asyncio.gather(*tasks)

    # Detect whether we are already inside a running event loop (e.g. Jupyter).
    try:
        asyncio.get_running_loop()
        # Running inside an existing event loop — use threads directly.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(call_agent, *args) for args in agent_calls]
            results = [f.result() for f in futures]
    except RuntimeError:
        # No running event loop — safe to use asyncio.run().
        results = asyncio.run(_async_parallel())

    # Inject Section 1 pre-produced content + agent extension
    sections = {
        "Section 1 (Foundations + Stratification Appendix)":
            SECTION_1_LATEX + "\n\n" + results[0],
        "Section 2 (Hamiltonian Structure)": results[1],
        "Sections 3+8 (Discretization + Algorithms)": results[2],
        "Section 4 (Control Theory)": results[3],
        "Sections 5+6 (CIIR Synthesis)": results[4],
    }
    for k in sections:
        print(f"  ✓ {k}")

    # Phase 3: Sequential
    print("\n[PHASE 3] Agent 6: Adversarial review...")
    review = call_agent(AGENT_6_SYSTEM, build_agent_6_user(sections), max_tokens=3000)
    print("  ✓ Review complete")

    print("\n[PHASE 3] Agent 7: Falsifiability + Conclusions...")
    sec9_10 = call_agent(AGENT_7_SYSTEM, AGENT_7_USER, max_tokens=3000)
    print("  ✓ Sections 9-10 complete")

    print("\n[PHASE 3] Agent 8: Final integration...")
    final_latex = call_agent(
        AGENT_8_SYSTEM,
        build_agent_8_user(outline_str, sections, review, sec9_10),
        max_tokens=8192
    )
    print("  ✓ Final document assembled")

    # Save outputs
    output = {
        "outline": outline_str,
        "sections": sections,
        "peer_review": review,
        "sections_9_10": sec9_10,
        "final_latex": final_latex,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    with open(out_tex, "w") as f:
        f.write(final_latex)

    print(f"\n✓ {out_json}")
    print(f"✓ {out_tex}")
    print("\nCompile: pdflatex whitepaper_v2.tex && bibtex whitepaper_v2 && pdflatex whitepaper_v2.tex")
    print("=" * 70)
    return output

# =============================================================================
# LANGGRAPH ADAPTER
# =============================================================================

LANGGRAPH_ADAPTER = '''
# LangGraph adapter — pip install langgraph langchain-anthropic
#
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, Optional
#
# class State(TypedDict):
#     outline: Optional[str]
#     sec1: Optional[str]; sec2: Optional[str]; sec34: Optional[str]
#     sec4c: Optional[str]; sec56: Optional[str]
#     review: Optional[str]; sec910: Optional[str]
#     final_latex: Optional[str]
#
# def n0(s): return {"outline": call_agent(AGENT_0_SYSTEM, AGENT_0_USER)}
# def n1(s): return {"sec1": SECTION_1_LATEX + call_agent(AGENT_1_SYSTEM, AGENT_1_USER)}
# def n2(s): return {"sec2": call_agent(AGENT_2_SYSTEM, AGENT_2_USER)}
# def n3(s): return {"sec34": call_agent(AGENT_3_SYSTEM, AGENT_3_USER)}
# def n4(s): return {"sec4c": call_agent(AGENT_4_SYSTEM, AGENT_4_USER)}
# def n5(s): return {"sec56": call_agent(AGENT_5_SYSTEM, AGENT_5_USER)}
# def n6(s):
#     secs = {"S1":s["sec1"],"S2":s["sec2"],"S34":s["sec34"],
#             "S4":s["sec4c"],"S56":s["sec56"]}
#     return {"review": call_agent(AGENT_6_SYSTEM, build_agent_6_user(secs))}
# def n7(s): return {"sec910": call_agent(AGENT_7_SYSTEM, AGENT_7_USER)}
# def n8(s):
#     secs = {"S1":s["sec1"],"S2":s["sec2"],"S34":s["sec34"],
#             "S4":s["sec4c"],"S56":s["sec56"]}
#     return {"final_latex": call_agent(AGENT_8_SYSTEM,
#         build_agent_8_user(s["outline"], secs, s["review"], s["sec910"]))}
#
# g = StateGraph(State)
# for name, fn in [("a0",n0),("a1",n1),("a2",n2),("a3",n3),
#                  ("a4",n4),("a5",n5),("a6",n6),("a7",n7),("a8",n8)]:
#     g.add_node(name, fn)
# g.set_entry_point("a0")
# for p in ["a1","a2","a3","a4","a5"]: g.add_edge("a0", p)
# for p in ["a1","a2","a3","a4","a5"]: g.add_edge(p, "a6")
# g.add_edge("a6","a7"); g.add_edge("a7","a8"); g.add_edge("a8", END)
# app = g.compile()
# result = app.invoke({})
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RMHD/CIIR Whitepaper Multi-Agent Orchestration Chain v2"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Anthropic model name (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Parallel workers for Phase 2 (default: 5)",
    )
    parser.add_argument(
        "--out-json",
        default="whitepaper_v2_output.json",
        help="Output JSON path (default: whitepaper_v2_output.json)",
    )
    parser.add_argument(
        "--out-tex",
        default="whitepaper_v2.tex",
        help="Output LaTeX path (default: whitepaper_v2.tex)",
    )
    args = parser.parse_args()
    run_pipeline(
        model=args.model,
        max_workers=args.max_workers,
        out_json=args.out_json,
        out_tex=args.out_tex,
    )
