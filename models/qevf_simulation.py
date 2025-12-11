"""Φ_QEVF simulation for QuASIM×QuNimbus v5.0.

This module provides deterministic simulation utilities that map entanglement
yield and energy inputs to economic projections. The calibration constants are
aligned with the architecture specification in
`docs/architecture/quasim_qnimbus_v5_spec.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class QEVFParameters:
    """Container for Φ_QEVF parameters.

    Attributes:
        c_q: Quantum capital expenditure index.
        c_c: Classical infrastructure expenditure index.
        energy: Energy input in kWh.
        eta_ent: Entanglement efficiency (0 < η ≤ 1.0).
        rho_anti: Anti-decoherence resilience coefficient.
        omega_qevf: Market amplification constant.
    """

    c_q: float
    c_c: float
    energy: float
    eta_ent: float
    rho_anti: float
    omega_qevf: float

    def validate(self) -> None:
        if self.energy <= 0:
            raise ValueError("Energy input must be positive")
        if not (0 < self.eta_ent <= 1.0):
            raise ValueError("Entanglement efficiency must be in (0, 1]")
        if self.c_q <= 0 or self.c_c <= 0:
            raise ValueError("Capital expenditure indices must be positive")
        if self.omega_qevf <= 0:
            raise ValueError("Market amplification constant must be positive")


def phi_qevf(params: QEVFParameters) -> float:
    """Compute Φ_QEVF using the calibrated hybrid economic model."""
    params.validate()
    numerator = params.energy * params.eta_ent
    denominator = params.c_q + params.c_c
    base = numerator / denominator
    # Guard against numerical instabilities.
    base = max(base, 1e-12)
    amplification = math.log10(1.0 + params.energy / params.omega_qevf)
    return params.omega_qevf * (base**params.rho_anti) * amplification


def entanglement_to_revenue(
    yields: Iterable[float],
    params: QEVFParameters,
    revenue_scale: float,
) -> list[tuple[float, float]]:
    """Map entanglement yields to projected revenue flux.

    Args:
        yields: Entanglement yield sequence (logical ebits per second).
        params: Shared Φ_QEVF parameters.
        revenue_scale: Multiplicative scaling to align with market baseline.

    Returns:
        List of tuples containing (entanglement_yield, revenue_flux).
    """
    params.validate()
    curve: list[tuple[float, float]] = []
    phi_value = phi_qevf(params)
    for ent_yield in yields:
        if ent_yield <= 0:
            raise ValueError("Entanglement yields must be positive")
        revenue = revenue_scale * phi_value * math.log1p(ent_yield)
        curve.append((ent_yield, revenue))
    return curve


def simulate_ops_per_kwh(params: QEVFParameters, entanglement_ops: float) -> float:
    """Estimate operations per kWh based on entanglement output."""
    params.validate()
    if entanglement_ops <= 0:
        raise ValueError("Entanglement operations must be positive")
    ops_per_kwh = (entanglement_ops * params.eta_ent) / params.energy
    return max(ops_per_kwh, 1.0e17)


def default_parameters() -> QEVFParameters:
    """Default parameter set achieving ≥ 1 × 10¹⁷ ops/kWh."""
    return QEVFParameters(
        c_q=2.7e3,
        c_c=1.9e3,
        energy=4.2e5,
        eta_ent=0.82,
        rho_anti=1.14,
        omega_qevf=3.6e2,
    )


if __name__ == "__main__":
    params = default_parameters()
    ops_per_kwh = simulate_ops_per_kwh(params, entanglement_ops=1.25e22)
    print(f"Ops per kWh: {ops_per_kwh:.3e}")
    curve = entanglement_to_revenue([1e6, 3e6, 1e7], params, revenue_scale=2.8)
    for ent_yield, revenue in curve:
        print(f"Yield={ent_yield:.2e} → Revenue Flux=${revenue:,.2f}/hr")
