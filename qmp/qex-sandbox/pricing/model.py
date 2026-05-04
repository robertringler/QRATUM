"""Quantum Market Protocol pricing model for the QEX sandbox.

The model exposes a deterministic valuation function ``R = k · N² · η_ent · P_EPH``
that links telemetry insights to economic projections.  ``N`` denotes the
observed logical qubit population, ``η_ent`` is the entanglement efficiency, and
``P_EPH`` is the entanglement price per hash as supplied by the mock ticker.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_mock_ticker():
    ticker_path = Path(__file__).resolve().parents[1] / "api" / "mock_ticker.py"
    spec = importlib.util.spec_from_file_location("qmp.qex_sandbox.api.mock_ticker", ticker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Unable to resolve QEX mock ticker"
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module.MockQuantumTicker


MockQuantumTicker = _load_mock_ticker()


@dataclass
class PricingSnapshot:
    """Serializable pricing artefact pushed to downstream telemetry."""

    eph_price_usd: float
    logical_qubits: int
    entanglement_efficiency: float
    scaling_constant: float
    projected_revenue_usd: float
    rmse: float
    variance_pct: float
    passed: bool


@dataclass
class QuantumPricingModel:
    """Valuation helper that reacts to telemetry verifier outcomes."""

    scaling_constant: float = 1.0
    ticker: MockQuantumTicker = field(default_factory=MockQuantumTicker)
    latest_snapshot: PricingSnapshot | None = None

    def revenue_projection(
        self,
        logical_qubits: int,
        entanglement_efficiency: float,
        eph_price_usd: float,
    ) -> float:
        """Compute the projected revenue for the supplied parameters."""

        return float(
            self.scaling_constant * (logical_qubits**2) * entanglement_efficiency * eph_price_usd
        )

    def update_from_verification(
        self,
        *,
        rmse: float,
        variance_pct: float,
        passed: bool,
        breach_count: int,
        telemetry: pd.DataFrame,
    ) -> PricingSnapshot:
        """Refresh the valuation state using verifier telemetry."""

        logical_qubits = int(np.round(float(telemetry["live"].max())))
        eph_price = self.ticker.current_price()
        entanglement_eff = float(
            np.clip(telemetry.get("entanglement_efficiency", pd.Series([0.93])).mean(), 0.0, 1.0)
        )
        revenue = self.revenue_projection(logical_qubits, entanglement_eff, eph_price)
        self.latest_snapshot = PricingSnapshot(
            eph_price_usd=eph_price,
            logical_qubits=logical_qubits,
            entanglement_efficiency=entanglement_eff,
            scaling_constant=self.scaling_constant,
            projected_revenue_usd=revenue,
            rmse=rmse,
            variance_pct=variance_pct,
            passed=passed,
        )
        self.ticker.record_snapshot(self.latest_snapshot, breach_count=breach_count)
        return self.latest_snapshot

    def history(self) -> Iterable[PricingSnapshot]:
        return self.ticker.history


__all__ = ["QuantumPricingModel", "PricingSnapshot"]
