"""Linearised (DC) power-flow network model for FLUXA.

Assumptions of the DC approximation [ASSUMED]
---------------------------------------------
1. Bus voltage magnitudes are fixed at 1.0 p.u.
2. Branch resistance is negligible relative to reactance (r << x), so
   branches are lossless and only the reactance enters the flow equations.
3. Voltage-angle differences across branches are small, so
   sin(theta_i - theta_j) ~= theta_i - theta_j.
4. Reactive power is not modelled.

Under these assumptions real-power flows are an exact linear function of bus
injections, which is what lets the dispatch engine embed network limits
directly in a linear program. The consequence is that FLUXA cannot speak to
voltage collapse, reactive-reserve adequacy or transmission losses; see the
report's Limitations section.

Formulation
-----------
With susceptance ``b_l = 1 / x_l`` for each branch ``l = (i, j)``:

    B_branch[l, i] = +b_l,  B_branch[l, j] = -b_l
    B_bus         = A^T diag(b) A          (A = branch-bus incidence)

Deleting the slack row and column gives the invertible reduced matrix
``B_red``. Then, for injections ``P`` (MW, summing to zero):

    theta_nonslack = B_red^-1 @ P_nonslack / S_BASE_MVA   [rad]
    flow           = PTDF @ P                             [MW]
    PTDF[:, nonslack] = B_branch[:, nonslack] @ B_red^-1
    PTDF[:, slack]    = 0

Because ``B_branch`` and ``B_bus`` carry the same susceptance units, the PTDF
is dimensionless and maps MW injections directly to MW flows.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fluxa.model import EnergySystem
from fluxa.units import S_BASE_MVA


@dataclass(frozen=True)
class DCNetwork:
    """Precomputed linear network operators for a fixed topology.

    Attributes:
        bus_ids: Bus identifiers in matrix column order.
        line_ids: Line identifiers in matrix row order.
        slack_index: Column index of the angle-reference bus.
        ptdf: (n_lines, n_buses) power-transfer distribution factors.
        b_reduced_inv: (n_buses-1, n_buses-1) inverse reduced susceptance
            matrix, used to recover bus angles.
        capacity_mw: (n_lines,) flow limits.
    """

    bus_ids: tuple[str, ...]
    line_ids: tuple[str, ...]
    slack_index: int
    ptdf: np.ndarray
    b_reduced_inv: np.ndarray
    capacity_mw: np.ndarray

    @classmethod
    def from_system(cls, system: EnergySystem) -> DCNetwork:
        """Build the DC operators for ``system``.

        Raises:
            ValueError: if the network is electrically disconnected, which
                makes the reduced susceptance matrix singular.
        """
        bus_index = system.bus_index
        n_bus = len(system.buses)
        n_line = len(system.lines)

        b_branch = np.zeros((n_line, n_bus), dtype=np.float64)
        b_bus = np.zeros((n_bus, n_bus), dtype=np.float64)

        for row, line in enumerate(system.lines):
            i = bus_index[line.from_bus]
            j = bus_index[line.to_bus]
            b = 1.0 / line.reactance_pu
            b_branch[row, i] = b
            b_branch[row, j] = -b
            b_bus[i, i] += b
            b_bus[j, j] += b
            b_bus[i, j] -= b
            b_bus[j, i] -= b

        slack = bus_index[system.slack_bus_id]
        keep = [k for k in range(n_bus) if k != slack]
        b_red = b_bus[np.ix_(keep, keep)]

        cond = np.linalg.cond(b_red)
        if not np.isfinite(cond) or cond > 1e12:
            raise ValueError(
                "reduced susceptance matrix is singular or near-singular "
                f"(condition number {cond:.3e}); the network is likely disconnected"
            )
        b_red_inv = np.linalg.inv(b_red)

        ptdf = np.zeros((n_line, n_bus), dtype=np.float64)
        ptdf[:, keep] = b_branch[:, keep] @ b_red_inv

        return cls(
            bus_ids=tuple(b.bus_id for b in system.buses),
            line_ids=tuple(line.line_id for line in system.lines),
            slack_index=slack,
            ptdf=ptdf,
            b_reduced_inv=b_red_inv,
            capacity_mw=np.array([line.capacity_mw for line in system.lines], dtype=np.float64),
        )

    # ------------------------------------------------------------------ use
    def line_flows_mw(self, injections_mw: np.ndarray) -> np.ndarray:
        """Signed branch flows (MW) for bus injections (MW).

        Positive flow is from ``from_bus`` to ``to_bus``.
        """
        self._check_injections(injections_mw)
        return self.ptdf @ injections_mw

    def bus_angles_rad(self, injections_mw: np.ndarray) -> np.ndarray:
        """Bus voltage angles (rad) with the slack bus at zero."""
        self._check_injections(injections_mw)
        keep = [k for k in range(len(self.bus_ids)) if k != self.slack_index]
        theta = np.zeros(len(self.bus_ids), dtype=np.float64)
        theta[keep] = self.b_reduced_inv @ (injections_mw[keep] / S_BASE_MVA)
        return theta

    def line_utilisation(self, flows_mw: np.ndarray) -> np.ndarray:
        """|flow| / capacity per branch (dimensionless, 1.0 == at limit)."""
        return np.abs(flows_mw) / self.capacity_mw

    def _check_injections(self, injections_mw: np.ndarray) -> None:
        if injections_mw.shape != (len(self.bus_ids),):
            raise ValueError(
                f"injections must have shape ({len(self.bus_ids)},), got {injections_mw.shape}"
            )
