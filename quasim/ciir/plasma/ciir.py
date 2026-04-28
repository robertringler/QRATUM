r"""CIIR (Constraints / Interfaces / Recursion) lens for the plasma module.

**Disclaimer (per problem-statement non-negotiable rules):** CIIR is a
*structuring lens only* — it adds no physical content beyond the equations and
diagnostics already implemented in :mod:`mhd`, :mod:`topology`,
:mod:`spectrum`, and :mod:`control`. The (C, I, R) decomposition here is
interpretive; predictions come from the underlying physics.

The dataclasses below are intentionally lightweight: they document the
mapping and provide programmatic access to it for downstream reporting,
visualization, and stability checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .mhd import MHDState
from .spectrum import SpectrumSnapshot, lundquist_number, spectrum
from .topology import TopologySnapshot, topology_snapshot

# ---------------------------------------------------------------------------
# Constraints (C)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraints:
    """C: physical and engineering constraints on the system [V].

    All entries are *labels* and *numeric thresholds*; the actual constraints
    are enforced by the PDE evolution (Maxwell + RMHD) and by the controller's
    actuator saturation.
    """

    field_equations: tuple[str, ...] = (
        "Maxwell",
        "resistive_MHD_2D",
        "incompressibility (∇·v = 0 by stream-function form)",
        "solenoidality (∇·B = 0 by flux-function form)",
    )
    closures: tuple[str, ...] = ("isothermal_implicit", "scalar_resistivity")
    conservation: tuple[str, ...] = (
        "energy (E_M + E_K, dissipated by η, ν)",
        "magnetic_flux (per island, modulo reconnection)",
    )
    parameters: tuple[str, ...] = ("S = L V_A / η", "Re = L V / ν")
    S_critical: float = 1.0e4
    """Critical Lundquist number for plasmoid onset (Loureiro et al. 2007) [V]."""

    actuator_limits: dict[str, float] = field(
        default_factory=lambda: {
            "E_drive_max": 1.0,
            "eta_boost_max": 100.0,
            "F_omega_max": 1.0,
        }
    )


# ---------------------------------------------------------------------------
# Interfaces (I)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interfaces:
    """I: diagnostic / interface manifold descriptors.

    The runtime interface state is a :class:`topology.TopologySnapshot` plus a
    :class:`spectrum.SpectrumSnapshot`; this dataclass enumerates the *kinds*
    of interfaces tracked.
    """

    topology_features: tuple[str, ...] = (
        "X-points (Δ-test < 0)",
        "O-points (Δ-test > 0)",
        "current_sheets (|Jz| > k·Jrms)",
        "separatrices (level sets through X-points; not explicitly traced)",
    )
    spectral_features: tuple[str, ...] = (
        "magnetic_shell_spectrum E(k)",
        "kinetic_shell_spectrum K(k)",
        "current_spectrum |Ĵz|^2(k)",
    )
    diagnostic_bandwidth_note: str = (
        "All diagnostics are full-state (we have access to ψ, ω). In real "
        "experiments observability is partial — see Failure Mode F1."
    )


# ---------------------------------------------------------------------------
# Recursion (R)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recursion:
    """R: instability cascades and feedback loops [V/D]."""

    cascades: tuple[str, ...] = (
        "tearing → magnetic_island → modified_Rutherford_growth",
        "Sweet–Parker_sheet → plasmoid_chain (S > S_c) → secondary_sheets",
        "turbulence ↔ reconnection (Mallet et al. 2017) [D]",
    )
    feedback_loops: tuple[str, ...] = (
        "current_sheet → reconnection_rate → island_growth → Δ' → faster_growth",
        "controller(E_drive, η_eff) → J_z localization → R̂ on target",
    )


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureMode:
    code: str
    name: str
    inequality: str
    description: str


FAILURE_MODES: tuple[FailureMode, ...] = (
    FailureMode(
        code="F1",
        name="unobserved_instability",
        inequality="dim(ker dh) ≥ dim(V_unstable)",
        description="Sub-resolution mode grows without diagnostic signature [V].",
    ),
    FailureMode(
        code="F2",
        name="control_lag",
        inequality="τ_ctrl > τ_instab",
        description="Controller latency exceeds fastest unstable growth time [V].",
    ),
    FailureMode(
        code="F3",
        name="model_mismatch",
        inequality="C_model ≠ C_real",
        description="E.g. reduced model omits Hall term while δ_SP ~ d_i [V].",
    ),
    FailureMode(
        code="F4",
        name="actuator_saturation",
        inequality="u ∉ U_admissible",
        description="Required forcing exceeds installed actuator capability [V].",
    ),
    FailureMode(
        code="F5",
        name="recursion_outpaces_estimator",
        inequality="γ_plasmoid > 1/τ_estimate",
        description="Plasmoid cascade finishes before state estimate refreshes [D].",
    ),
)


# ---------------------------------------------------------------------------
# CIIR snapshot + stability inequality
# ---------------------------------------------------------------------------


@dataclass
class CIIRSnapshot:
    """Aggregate (C, I, R) snapshot at one simulation time.

    Attributes
    ----------
    t : float
    constraints : Constraints
    interfaces : Interfaces
    recursion : Recursion
    topology : TopologySnapshot
    spectrum : SpectrumSnapshot
    S : float
        Current Lundquist number [V].
    plasmoid_regime : bool
        True iff S > S_c (plasmoid onset, [V]).
    instability_norm : float
        :math:`\\|R_{instab}\\|` proxy: peak magnetic energy at high k.
    control_norm : float
        :math:`\\|R_{ctrl}\\|` proxy: integrated objective gain on target region.
    stable : bool
        True iff control_norm > instability_norm AND no failure-mode predicate
        is currently triggered.
    """

    t: float
    constraints: Constraints
    interfaces: Interfaces
    recursion: Recursion
    topology: TopologySnapshot
    spectrum: SpectrumSnapshot
    S: float
    plasmoid_regime: bool
    instability_norm: float
    control_norm: float
    stable: bool


def _instability_norm(spec: SpectrumSnapshot, k_band: tuple[float, float]) -> float:
    """Sum E_mag over the plasmoid k-band as a recursion-strength proxy [D]."""
    k_lo, k_hi = k_band
    mask = (spec.k_bins >= k_lo) & (spec.k_bins <= k_hi)
    return float(spec.E_mag[mask].sum())


def make_snapshot(
    state: MHDState,
    control_norm: float = 0.0,
    k_band: tuple[float, float] = (4.0, 16.0),
    constraints: Optional[Constraints] = None,
) -> CIIRSnapshot:
    """Compute a complete CIIR snapshot at the current state.

    Parameters
    ----------
    state
        Current MHD state.
    control_norm
        Externally supplied scalar measure of controller strength (e.g. the
        composite objective value, or :math:`||u||_2`). The caller decides the
        norm; here we only compare it against the spectral instability norm.
    k_band
        Plasmoid wavenumber band used for the instability norm.
    constraints
        Optional pre-built Constraints object (so the caller can override
        ``S_critical`` or actuator limits).
    """
    C = constraints or Constraints()
    I = Interfaces()
    R = Recursion()
    topo = topology_snapshot(state)
    spec = spectrum(state)
    S = lundquist_number(state)
    inst_norm = _instability_norm(spec, k_band)
    plasmoid = bool(S > C.S_critical)
    stable = control_norm > inst_norm and not plasmoid
    return CIIRSnapshot(
        t=state.t,
        constraints=C,
        interfaces=I,
        recursion=R,
        topology=topo,
        spectrum=spec,
        S=S,
        plasmoid_regime=plasmoid,
        instability_norm=inst_norm,
        control_norm=control_norm,
        stable=stable,
    )


def stability_inequality(snapshot: CIIRSnapshot) -> bool:
    r"""Return True iff :math:`\|R_{ctrl}\| > \|R_{instab}\|` [D].

    This is the structural stability condition (problem-statement §6); it is
    *interpretive* and does not by itself guarantee disruption avoidance.
    """
    return snapshot.control_norm > snapshot.instability_norm


__all__ = [
    "Constraints",
    "Interfaces",
    "Recursion",
    "FailureMode",
    "FAILURE_MODES",
    "CIIRSnapshot",
    "make_snapshot",
    "stability_inequality",
]
