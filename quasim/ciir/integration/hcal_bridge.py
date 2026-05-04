"""HCAL bridge: hardware constraint injection Π_C.

Maps CIIR constraint projection to HCAL hardware calibration layer,
enabling constraint enforcement on physical compute backends.

QRATUM subsystem: HCAL (Hardware Configuration and Calibration)
CIIR mapping: Π_C ↔ hardware constraint injection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quasim.ciir.evolution import project_density
from quasim.ciir.theory import CIIRTheory, RealTensor


@dataclass
class HCALConstraintInjector:
    """Bridge CIIR constraint projection to HCAL hardware layer.

    The HCAL system enforces hardware constraints (device precision,
    memory limits, compute topology).  This bridge maps the abstract
    CIIR projection Π_C onto these concrete constraints.

    Parameters
    ----------
    theory : CIIRTheory
        The CIIR theory tuple.
    device_precision : str
        Hardware precision mode ('fp32', 'fp64').
    memory_budget_mb : float
        Available device memory in MB.
    """

    theory: CIIRTheory
    device_precision: str = "fp64"
    memory_budget_mb: float = 1024.0

    @property
    def dtype(self) -> np.dtype:
        """Map precision mode to numpy dtype."""
        return np.float32 if self.device_precision == "fp32" else np.float64

    def state_memory_mb(self, batch: int) -> float:
        """Estimate memory footprint of state tensor in MB."""
        D = self.theory.manifold.rep_dim
        bytes_per_elem = 4 if self.device_precision == "fp32" else 8
        return batch * D * D * bytes_per_elem / (1024 * 1024)

    def max_batch_size(self) -> int:
        """Maximum batch size within memory budget."""
        D = self.theory.manifold.rep_dim
        bytes_per_elem = 4 if self.device_precision == "fp32" else 8
        per_state = D * D * bytes_per_elem / (1024 * 1024)
        return max(1, int(self.memory_budget_mb / per_state))

    def inject_constraints(self, rho: RealTensor) -> RealTensor:
        """Apply hardware-aware constraint projection.

        1. Cast to device precision
        2. Project onto density operator cone
        3. Clip to hardware-representable range
        """
        rho = rho.astype(self.dtype)
        rho = project_density(rho)
        # Clip to representable range
        finfo = np.finfo(self.dtype)
        return np.clip(rho, finfo.tiny, finfo.max)

    def to_hcal_config(self) -> dict[str, Any]:
        """Export as HCAL-compatible configuration dict."""
        return {
            "ciir_rank": self.theory.manifold.rank,
            "ciir_rep_dim": self.theory.manifold.rep_dim,
            "n_constraints": len(self.theory.constraints),
            "n_observers": len(self.theory.observers),
            "precision": self.device_precision,
            "memory_budget_mb": self.memory_budget_mb,
            "max_batch": self.max_batch_size(),
        }
