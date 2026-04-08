"""CIIR → QuASIM → QRATUM simulation engine with full instrumentation.

Each timestep:
    1. Log state
    2. Apply CIIR constraint
    3. Evolve via Hamiltonian (QuASIM layer)
    4. Apply measurement sequence
    5. Update operators (QRATUM)
    6. Log drift + commutators
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quasim.ciir.core import (
    density_matrix,
    measurement_probability,
    normalize,
    von_neumann_entropy,
)
from quasim.ciir.operators import OperatorSystem

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Simulation hyper-parameters.

    Attributes:
        dim: Hilbert-space dimension.
        dt: Time-step size.
        num_steps: Number of evolution steps.
        eta: QRATUM learning rate.
        constraint_strength: CIIR constraint operator strength.
        hamiltonian_scale: Hamiltonian energy scale.
        seed: Random seed for reproducibility.
    """

    dim: int = 4
    dt: float = 0.05
    num_steps: int = 200
    eta: float = 0.05
    constraint_strength: float = 0.3
    hamiltonian_scale: float = 1.0
    seed: int = 42


# ------------------------------------------------------------------
# Timestep record
# ------------------------------------------------------------------


@dataclass
class StepRecord:
    """Single-timestep instrumentation record."""

    t: int
    psi: np.ndarray
    entropy: float
    operator_drifts: dict[str, float]
    commutator_norms: dict[tuple[str, str], float]
    order_effect: float


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


class CIIRSimulationEngine:
    """Full CIIR → QuASIM → QRATUM simulation loop.

    Parameters
    ----------
    config : SimulationConfig
        Simulation hyper-parameters.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or SimulationConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.dim = self.cfg.dim

        # State
        self.psi = self._random_state()

        # Hamiltonian (Hermitian)
        h_raw = self.rng.standard_normal(
            (self.dim, self.dim)
        ) + 1j * self.rng.standard_normal((self.dim, self.dim))
        self.hamiltonian = self.cfg.hamiltonian_scale * (h_raw + h_raw.conj().T) / 2

        # CIIR constraint operator (Hermitian, positive semi-definite)
        c_raw = self.rng.standard_normal(
            (self.dim, self.dim)
        ) + 1j * self.rng.standard_normal((self.dim, self.dim))
        c_herm = (c_raw + c_raw.conj().T) / 2
        self.constraint = (
            np.eye(self.dim) + self.cfg.constraint_strength * c_herm
        )

        # Operators
        self.op_sys = OperatorSystem.random(
            labels=["A", "B", "C"],
            dim=self.dim,
            eta=self.cfg.eta,
            rng=self.rng,
        )

        # Instrumentation
        self.history: list[StepRecord] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_state(self) -> np.ndarray:
        real = self.rng.standard_normal(self.dim)
        imag = self.rng.standard_normal(self.dim)
        return normalize(real + 1j * imag)

    def _apply_constraint(self, psi: np.ndarray) -> np.ndarray:
        """CIIR constraint projection: ψ → C ψ / ‖C ψ‖."""
        return normalize(self.constraint @ psi)

    def _hamiltonian_evolve(self, psi: np.ndarray) -> np.ndarray:
        """First-order unitary: ψ → (I − iHΔt)ψ / ‖…‖."""
        evolved = (np.eye(self.dim) - 1j * self.hamiltonian * self.cfg.dt) @ psi
        return normalize(evolved)

    def _apply_measurement(
        self, psi: np.ndarray, label: str
    ) -> np.ndarray:
        """Projective measurement: ψ → M_i ψ / ‖M_i ψ‖."""
        m = self.op_sys.operators[label]
        result = m @ psi
        if np.linalg.norm(result) < 1e-15:
            return psi  # operator annihilates state — keep unchanged
        return normalize(result)

    # ------------------------------------------------------------------
    # Order-effect metric
    # ------------------------------------------------------------------

    def _order_effect(self, psi: np.ndarray) -> float:
        """Δ = P(A→B) − P(B→A) at state *psi*."""
        m_a = self.op_sys.operators["A"]
        m_b = self.op_sys.operators["B"]

        try:
            # A then B
            psi_a = normalize(m_a @ psi)
            p_ab = measurement_probability(psi_a, m_b)

            # B then A
            psi_b = normalize(m_b @ psi)
            p_ba = measurement_probability(psi_b, m_a)
        except ValueError:
            # Degenerate state — operator maps to zero
            return 0.0

        return float(p_ab - p_ba)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self) -> StepRecord:
        """Execute one simulation timestep and record instrumentation."""
        t = len(self.history)

        # 1. Compute order effect *before* updates
        oe = self._order_effect(self.psi)

        # 2. CIIR constraint
        self.psi = self._apply_constraint(self.psi)

        # 3. Hamiltonian evolution (QuASIM layer)
        self.psi = self._hamiltonian_evolve(self.psi)

        # 4. Sequential measurement (A → B → C)
        for label in ("A", "B", "C"):
            self.psi = self._apply_measurement(self.psi, label)

        # 5. QRATUM operator update
        rho = density_matrix(self.psi)
        drifts = self.op_sys.qratum_update(rho)

        # 6. Entropy & commutators
        entropy = von_neumann_entropy(rho)
        cnorms = self.op_sys.commutator_norms()

        rec = StepRecord(
            t=t,
            psi=self.psi.copy(),
            entropy=entropy,
            operator_drifts=drifts,
            commutator_norms=cnorms,
            order_effect=oe,
        )
        self.history.append(rec)
        return rec

    def run(self) -> list[StepRecord]:
        """Run the full simulation for ``num_steps`` timesteps."""
        for _ in range(self.cfg.num_steps):
            self.step()
        return self.history

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def collect_metrics(self) -> dict[str, Any]:
        """Return aggregate metrics from the recorded history."""
        oes = [r.order_effect for r in self.history]
        entropies = [r.entropy for r in self.history]

        # Drift per operator
        drift_by_label: dict[str, list[float]] = {}
        for r in self.history:
            for lbl, d in r.operator_drifts.items():
                drift_by_label.setdefault(lbl, []).append(d)

        mean_drifts = {
            k: float(np.mean(v)) for k, v in drift_by_label.items()
        }

        # Eigenvalue spectra of current operators
        eigenspectra: dict[str, list[complex]] = {}
        for lbl, m in self.op_sys.operators.items():
            eigenspectra[lbl] = np.linalg.eigvals(m).tolist()

        return {
            "order_effect_mean": float(np.mean(oes)),
            "order_effect_var": float(np.var(oes)),
            "order_effect_first": oes[0] if oes else 0.0,
            "order_effect_last": oes[-1] if oes else 0.0,
            "entropy_trajectory": entropies,
            "mean_operator_drifts": mean_drifts,
            "eigenspectra": eigenspectra,
        }
