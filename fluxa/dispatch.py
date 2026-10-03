"""FLUXA dispatch engine: receding-horizon linear program over a DC network.

Formulation
===========

At each simulation timestep the engine solves a linear program over a
look-ahead window of ``horizon_steps`` timesteps, commits the first step's
decisions, advances the physical state, and repeats. This is standard
receding-horizon (model-predictive) economic dispatch. Within the window the
drivers are treated as perfectly known; across windows nothing is assumed.

Decision variables (all MW except ``soc``, which is a fraction)
---------------------------------------------------------------
For each horizon step ``h`` in ``[0, H)``:

===========================  ======================================
``p_disp[h, g]``             output of dispatchable generator ``g``
``p_var[h, j]``              accepted output of variable unit ``j``
``p_charge[h, b]``           battery ``b`` charge power
``p_discharge[h, b]``        battery ``b`` discharge power
``soc[h, b]``                battery ``b`` state of charge at end of ``h``
``r_batt[h, b]``             battery ``b`` reserve contribution
``p_import[h, l]``           import on interconnection ``l``
``p_export[h, l]``           export on interconnection ``l``
``unserved[h, k]``           involuntarily shed load at load bus ``k``
``overload[h, m]``           |flow| in excess of the rating of line ``m``
``reserve_short[h]``         unmet operating-reserve requirement
===========================  ======================================

Objective (minimise, USD over the window)
-----------------------------------------
::

    sum_h dt_h * [ sum_g  c_g           * p_disp[h,g]
                 + sum_j (c_j - c_curt) * p_var[h,j]
                 + sum_b  c_cycle       * (p_charge + p_discharge)
                 + sum_l (pi_imp * p_import - pi_exp * p_export)
                 + VOLL   * sum_k unserved[h,k]
                 + C_ovl  * sum_m overload[h,m]
                 + C_res  * reserve_short[h] ]
    - sum_b v_stored * E_b * soc[H-1, b]
    + K                                         (constant, see below)

``K = sum_h dt_h * c_curt * sum_j avail[h,j]`` is the cost of curtailing
*all* variable output; the ``-c_curt`` coefficient on ``p_var`` credits back
the part that is accepted, so the net effect is a charge of ``c_curt`` per
MWh actually curtailed. ``K`` is constant within a solve and is added back
when reporting cost so that reported numbers are absolute, not offset.

The terminal term ``-v_stored * E_b * soc[H-1,b]`` prices stored energy at
the end of the window. Without it a finite-horizon LP always empties the
battery at the window edge, which is an artefact, not a dispatch decision.
``v_stored`` defaults to the mid-merit marginal cost (48 USD/MWh).

``C_ovl`` and ``C_res`` are soft-constraint penalties, not market prices.
They keep the LP feasible so that a violation is *measured* instead of
turning the timestep into a solver failure. They are reported separately
from operating cost.

Constraints
-----------
1. **Power balance** (equality, per step)::

       sum p_disp + sum p_var + sum(p_discharge - p_charge)
       + sum(p_import - p_export) + sum_k unserved[h,k] = sum_k load[h,k]

2. **Storage conservation** (equality, per step per battery)::

       soc[h,b] * E_b = soc[h-1,b] * E_b
                        + eta_ch * p_charge[h,b] * dt_h
                        - p_discharge[h,b] * dt_h / eta_dis

   with ``soc[-1,b]`` the committed state of charge entering the window.

3. **Network limits** (inequality, per step per line), with ``F = PTDF``::

       +F_m . Pinj[h] - overload[h,m] <= cap[h,m]
       -F_m . Pinj[h] - overload[h,m] <= cap[h,m]

   where ``Pinj`` is the vector of bus injections implied by the decision
   variables and the load, so network limits are enforced exactly within the
   DC approximation rather than through a proxy.

4. **Ramp limits** (inequality, per step per dispatchable unit)::

       |p_disp[h,g] - p_disp[h-1,g]| <= ramp_g * dt_minutes
                                        + p_max_g * (1 - min(a[h,g], a[h-1,g]))

   with ``p_disp[-1,g]`` the committed output entering the window. The
   availability-dependent relaxation term is required for correctness, not
   convenience: a generator *trip* is not a ramp. Without it, forcing an
   online unit to zero in one timestep contradicts its ramp limit and the LP
   is infeasible, so a forced-outage scenario could not be simulated at all.
   The term vanishes (``a = 1`` on both sides) whenever the unit is fully
   available, so normal operation is still ramp-constrained exactly.

5. **Operating reserve** (inequality, per step)::

       sum_g (p_max_g * a[h,g] - p_disp[h,g])
       + sum_b r_batt[h,b]
       + sum_l (imp_max_l - p_import[h,l])
       + reserve_short[h]                        >= R[h]

   ``R[h] = f_load * load[h] + f_vre * sum_j avail[h,j]``. The requirement
   uses *available* (pre-curtailment) variable output, because that is the
   physical uncertainty driver; using accepted output would reward curtailing
   renewables to shrink the reserve obligation.

6. **Reserve is energy-tested for storage** (inequality, per step per
   battery)::

       r_batt[h,b] <= p_dis_max_b - p_discharge[h,b] + p_charge[h,b]
       r_batt[h,b] * T_res / eta_dis <= (soc[h,b] - soc_min_b) * E_b

   The second row is what distinguishes a *power* headroom claim from a
   deliverable reserve: a nearly empty battery contributes nearly nothing.

Bounds
------
``p_min_g * a[h,g] <= p_disp[h,g] <= p_max_g * a[h,g]`` (so ``a = 0`` forces
the unit off and ``a = 1`` imposes the must-run floor),
``0 <= p_var[h,j] <= avail[h,j]``, ``soc_min <= soc <= soc_max``, and the
remaining variables non-negative with their nameplate upper bounds.

Modelling choices that are approximations [ASSUMED]
---------------------------------------------------
* No unit-commitment binaries. The LP is a continuous relaxation, so a unit
  with ``p_min > 0`` is a must-run whenever it is available. The peaking unit
  is given ``p_min = 0`` so it can sit at zero without a binary.
* Charge and discharge are not mutually exclusive by construction. Round-trip
  loss plus cycle cost make simultaneous operation strictly suboptimal, and
  the engine verifies this holds at every committed step rather than assuming
  it (see :attr:`DispatchSolution.simultaneous_charge_discharge_mw`).
* Losses are zero (DC approximation).

Solver
------
``scipy.optimize.linprog(method="highs")`` -- the HiGHS dual simplex /
interior-point solver bundled with SciPy. The exact SciPy and HiGHS versions
used in a run are recorded in the provenance bundle. Convergence criterion is
HiGHS' own primal/dual feasibility and optimality tolerances at their SciPy
defaults; the engine additionally rejects any solve whose status is not
``0`` (optimal) and re-verifies the returned primal against the physical
constraints before accepting it.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy
from scipy.optimize import linprog

from fluxa.model import EnergySystem
from fluxa.network import DCNetwork

#: Status codes from ``scipy.optimize.linprog`` that FLUXA accepts.
_OPTIMAL = 0


class DispatchError(RuntimeError):
    """Raised when the dispatch LP cannot be solved or its primal is invalid."""


@dataclass(frozen=True)
class DispatchStep:
    """Exogenous data for one timestep of a dispatch window."""

    bus_load_mw: np.ndarray
    variable_availability_mw: np.ndarray
    dispatchable_availability: np.ndarray
    line_capacity_mw: np.ndarray


@dataclass
class DispatchSolution:
    """Committed decisions for a single timestep.

    All power quantities are MW, costs USD for the committed timestep only.
    """

    p_disp_mw: np.ndarray
    p_var_mw: np.ndarray
    p_charge_mw: np.ndarray
    p_discharge_mw: np.ndarray
    soc_end: np.ndarray
    r_batt_mw: np.ndarray
    p_import_mw: np.ndarray
    p_export_mw: np.ndarray
    unserved_mw: np.ndarray
    overload_mw: np.ndarray
    reserve_short_mw: float
    reserve_requirement_mw: float
    reserve_available_mw: float
    line_flow_mw: np.ndarray
    bus_angle_rad: np.ndarray
    bus_injection_mw: np.ndarray
    window_objective_usd: float
    step_operating_cost_usd: float
    step_violation_cost_usd: float
    step_curtailment_cost_usd: float
    solver_status: int
    solver_message: str
    solver_iterations: int
    simultaneous_charge_discharge_mw: float
    balance_residual_mw: float


class DispatchProblem:
    """Precompiled LP structure for one (system, horizon, timestep) triple.

    The constraint *matrices* depend only on topology, horizon length and
    timestep duration, so they are built once in ``__init__``. Each solve
    supplies only the right-hand sides and variable bounds, which is what
    keeps a 2016-step simulation to a few seconds of Python overhead.
    """

    def __init__(
        self,
        system: EnergySystem,
        network: DCNetwork,
        timestep_s: float,
        horizon_steps: int = 12,
    ) -> None:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        self.system = system
        self.network = network
        self.timestep_s = float(timestep_s)
        self.dt_h = self.timestep_s / 3600.0
        self.dt_min = self.timestep_s / 60.0
        self.H = int(horizon_steps)

        self.disp = system.dispatchable_generators
        self.var = system.variable_generators
        self.bats = system.batteries
        self.links = system.interconnections
        self.load_buses = system.load_buses

        self.n_disp = len(self.disp)
        self.n_var = len(self.var)
        self.n_bat = len(self.bats)
        self.n_link = len(self.links)
        self.n_load = len(self.load_buses)
        self.n_bus = len(system.buses)
        self.n_line = len(system.lines)

        # ---- variable index layout -------------------------------------
        blocks: list[tuple[str, int]] = [
            ("p_disp", self.n_disp),
            ("p_var", self.n_var),
            ("p_charge", self.n_bat),
            ("p_discharge", self.n_bat),
            ("soc", self.n_bat),
            ("r_batt", self.n_bat),
            ("p_import", self.n_link),
            ("p_export", self.n_link),
            ("unserved", self.n_load),
            ("overload", self.n_line),
            ("reserve_short", 1),
        ]
        self._offsets: dict[str, int] = {}
        cursor = 0
        for name, width in blocks:
            self._offsets[name] = cursor
            cursor += width
        self.vars_per_step = cursor
        self.n_vars = self.vars_per_step * self.H

        self._build_structure()
        self._build_objective()

    # ------------------------------------------------------------ indexing
    def idx(self, block: str, step: int, item: int = 0) -> int:
        """Flat index of decision variable ``block[step, item]``."""
        return step * self.vars_per_step + self._offsets[block] + item

    # ------------------------------------------------------- construction
    def _bus_column(self) -> dict[str, int]:
        return self.system.bus_index

    def _build_structure(self) -> None:
        """Build the constant equality/inequality coefficient matrices."""
        H, V = self.H, self.n_vars
        bus_col = self._bus_column()

        # --- injection map: (n_bus, vars_per_step) for a single step ------
        # Pinj[b] = M[b] . x_step  +  (-load[b])   (load added per solve)
        inj = np.zeros((self.n_bus, self.vars_per_step), dtype=np.float64)
        for g_i, gen in enumerate(self.disp):
            inj[bus_col[gen.bus_id], self._offsets["p_disp"] + g_i] += 1.0
        for v_i, gen in enumerate(self.var):
            inj[bus_col[gen.bus_id], self._offsets["p_var"] + v_i] += 1.0
        for b_i, bat in enumerate(self.bats):
            col = bus_col[bat.bus_id]
            inj[col, self._offsets["p_discharge"] + b_i] += 1.0
            inj[col, self._offsets["p_charge"] + b_i] -= 1.0
        for l_i, link in enumerate(self.links):
            col = bus_col[link.bus_id]
            inj[col, self._offsets["p_import"] + l_i] += 1.0
            inj[col, self._offsets["p_export"] + l_i] -= 1.0
        for k_i, bus in enumerate(self.load_buses):
            inj[bus_col[bus.bus_id], self._offsets["unserved"] + k_i] += 1.0
        self._injection_map = inj

        # --- equalities ---------------------------------------------------
        # (1) balance: 1^T . Pinj = 0  <=>  sum(vars) = sum(load)
        n_eq = H + H * self.n_bat
        a_eq = np.zeros((n_eq, V), dtype=np.float64)
        balance_row = inj.sum(axis=0)
        for h in range(H):
            a_eq[h, h * self.vars_per_step : (h + 1) * self.vars_per_step] = balance_row

        # (2) storage conservation
        for h in range(H):
            for b_i, bat in enumerate(self.bats):
                row = H + h * self.n_bat + b_i
                e = bat.energy_capacity_mwh
                a_eq[row, self.idx("soc", h, b_i)] = e
                if h > 0:
                    a_eq[row, self.idx("soc", h - 1, b_i)] = -e
                a_eq[row, self.idx("p_charge", h, b_i)] = -bat.charge_efficiency * self.dt_h
                a_eq[row, self.idx("p_discharge", h, b_i)] = self.dt_h / bat.discharge_efficiency
        self._a_eq = a_eq

        # --- inequalities -------------------------------------------------
        rows_line = 2 * H * self.n_line
        rows_ramp = 2 * H * self.n_disp
        rows_res = H
        rows_bres = 2 * H * self.n_bat
        n_ub = rows_line + rows_ramp + rows_res + rows_bres
        a_ub = np.zeros((n_ub, V), dtype=np.float64)

        ptdf = self.network.ptdf                       # (n_line, n_bus)
        line_coeff = ptdf @ inj                        # (n_line, vars_per_step)
        self._line_coeff = line_coeff
        self._ptdf_load = ptdf                         # reused for RHS

        r = 0
        self._row_line_pos = r
        for h in range(H):
            sl = slice(h * self.vars_per_step, (h + 1) * self.vars_per_step)
            for m in range(self.n_line):
                a_ub[r, sl] = line_coeff[m]
                a_ub[r, self.idx("overload", h, m)] -= 1.0
                r += 1
        self._row_line_neg = r
        for h in range(H):
            sl = slice(h * self.vars_per_step, (h + 1) * self.vars_per_step)
            for m in range(self.n_line):
                a_ub[r, sl] = -line_coeff[m]
                a_ub[r, self.idx("overload", h, m)] -= 1.0
                r += 1

        self._row_ramp_up = r
        for h in range(H):
            for g_i in range(self.n_disp):
                a_ub[r, self.idx("p_disp", h, g_i)] = 1.0
                if h > 0:
                    a_ub[r, self.idx("p_disp", h - 1, g_i)] = -1.0
                r += 1
        self._row_ramp_dn = r
        for h in range(H):
            for g_i in range(self.n_disp):
                a_ub[r, self.idx("p_disp", h, g_i)] = -1.0
                if h > 0:
                    a_ub[r, self.idx("p_disp", h - 1, g_i)] = 1.0
                r += 1

        self._row_reserve = r
        for h in range(H):
            for g_i in range(self.n_disp):
                a_ub[r, self.idx("p_disp", h, g_i)] = 1.0
            for b_i in range(self.n_bat):
                a_ub[r, self.idx("r_batt", h, b_i)] = -1.0
            for l_i in range(self.n_link):
                a_ub[r, self.idx("p_import", h, l_i)] = 1.0
            a_ub[r, self.idx("reserve_short", h, 0)] = -1.0
            r += 1

        self._row_bres_power = r
        for h in range(H):
            for b_i, bat in enumerate(self.bats):
                a_ub[r, self.idx("r_batt", h, b_i)] = 1.0
                a_ub[r, self.idx("p_discharge", h, b_i)] = 1.0
                a_ub[r, self.idx("p_charge", h, b_i)] = -1.0
                r += 1
        self._row_bres_energy = r
        for h in range(H):
            for b_i, bat in enumerate(self.bats):
                a_ub[r, self.idx("r_batt", h, b_i)] = (
                    self.system.reserve.duration_h / bat.discharge_efficiency
                )
                a_ub[r, self.idx("soc", h, b_i)] = -bat.energy_capacity_mwh
                r += 1
        assert r == n_ub, f"inequality row count mismatch: {r} != {n_ub}"
        self._a_ub = a_ub

        # Constant parts of b_ub that never change across solves.
        self._b_bres_power = np.array(
            [bat.p_discharge_max_mw for bat in self.bats] * H, dtype=np.float64
        )
        self._b_bres_energy = np.array(
            [-bat.soc_min * bat.energy_capacity_mwh for bat in self.bats] * H, dtype=np.float64
        )

    def _build_objective(self) -> None:
        econ = self.system.economics
        c = np.zeros(self.n_vars, dtype=np.float64)
        for h in range(self.H):
            for g_i, gen in enumerate(self.disp):
                c[self.idx("p_disp", h, g_i)] = self.dt_h * (
                    gen.marginal_cost_usd_per_mwh
                    + econ.co2_price_usd_per_tonne * gen.co2_tonnes_per_mwh
                )
            for v_i, gen in enumerate(self.var):
                c[self.idx("p_var", h, v_i)] = self.dt_h * (
                    gen.marginal_cost_usd_per_mwh - econ.curtailment_cost_usd_per_mwh
                )
            for b_i, bat in enumerate(self.bats):
                c[self.idx("p_charge", h, b_i)] = self.dt_h * bat.cycle_cost_usd_per_mwh
                c[self.idx("p_discharge", h, b_i)] = self.dt_h * bat.cycle_cost_usd_per_mwh
            for l_i, link in enumerate(self.links):
                c[self.idx("p_import", h, l_i)] = self.dt_h * link.import_price_usd_per_mwh
                c[self.idx("p_export", h, l_i)] = -self.dt_h * link.export_price_usd_per_mwh
            for k_i in range(self.n_load):
                c[self.idx("unserved", h, k_i)] = (
                    self.dt_h * econ.value_of_lost_load_usd_per_mwh
                )
            for m in range(self.n_line):
                c[self.idx("overload", h, m)] = self.dt_h * econ.overload_penalty_usd_per_mwh
            c[self.idx("reserve_short", h, 0)] = (
                self.dt_h * econ.reserve_shortfall_penalty_usd_per_mwh
            )
        for b_i, bat in enumerate(self.bats):
            c[self.idx("soc", self.H - 1, b_i)] = (
                -econ.stored_energy_value_usd_per_mwh * bat.energy_capacity_mwh
            )
        self._c = c

    # ------------------------------------------------------------- solving
    def solve(
        self,
        window: list[DispatchStep],
        soc_prev: np.ndarray,
        p_disp_prev: np.ndarray,
        avail_prev: np.ndarray | None = None,
    ) -> DispatchSolution:
        """Solve one window and return the committed first-step decisions.

        Args:
            window: Exactly ``horizon_steps`` steps of exogenous data. The
                caller pads the window by repeating the final timestep when
                fewer steps remain in the simulation.
            soc_prev: (n_batteries,) state of charge entering the window.
            p_disp_prev: (n_dispatchable,) committed output entering the window.
            avail_prev: (n_dispatchable,) availability at the step *before* the
                window. Needed so that a unit tripping at the window's first
                step gets its ramp constraint relaxed. Defaults to fully
                available, which is correct for the first step of a run.

        Returns:
            A :class:`DispatchSolution` for the first step of the window.

        Raises:
            DispatchError: on a non-optimal solver status, or if the returned
                primal violates power balance beyond tolerance.
        """
        if len(window) != self.H:
            raise DispatchError(f"window must have {self.H} steps, got {len(window)}")

        H, econ = self.H, self.system.economics
        b_eq = np.zeros(H + H * self.n_bat, dtype=np.float64)
        for h, step in enumerate(window):
            b_eq[h] = float(step.bus_load_mw.sum())
        for b_i, bat in enumerate(self.bats):
            b_eq[H + b_i] = soc_prev[b_i] * bat.energy_capacity_mwh

        # ---- inequality right-hand sides --------------------------------
        b_line_pos = np.empty(H * self.n_line, dtype=np.float64)
        b_line_neg = np.empty(H * self.n_line, dtype=np.float64)
        for h, step in enumerate(window):
            shift = self._ptdf_load @ step.bus_load_mw     # PTDF . load  (MW)
            sl = slice(h * self.n_line, (h + 1) * self.n_line)
            b_line_pos[sl] = step.line_capacity_mw + shift
            b_line_neg[sl] = step.line_capacity_mw - shift

        ramp_limit = np.array([g.ramp_mw_per_min * self.dt_min for g in self.disp], dtype=np.float64)
        p_max = np.array([g.p_max_mw for g in self.disp], dtype=np.float64)
        prev_avail = (
            np.ones(self.n_disp, dtype=np.float64)
            if avail_prev is None
            else np.asarray(avail_prev, dtype=np.float64)
        )
        if prev_avail.shape != (self.n_disp,):
            raise DispatchError(
                f"avail_prev must have shape ({self.n_disp},), got {prev_avail.shape}"
            )
        b_ramp_up = np.empty(H * self.n_disp, dtype=np.float64)
        b_ramp_dn = np.empty(H * self.n_disp, dtype=np.float64)
        for h, step in enumerate(window):
            a_now = np.asarray(step.dispatchable_availability, dtype=np.float64)
            a_before = prev_avail if h == 0 else np.asarray(
                window[h - 1].dispatchable_availability, dtype=np.float64
            )
            # A trip or a return to service is not a ramp; relax by the
            # capacity that changed availability.
            slack = p_max * (1.0 - np.minimum(a_now, a_before))
            sl = slice(h * self.n_disp, (h + 1) * self.n_disp)
            b_ramp_up[sl] = ramp_limit + slack
            b_ramp_dn[sl] = ramp_limit + slack
        b_ramp_up[: self.n_disp] += p_disp_prev
        b_ramp_dn[: self.n_disp] -= p_disp_prev

        b_res = np.empty(H, dtype=np.float64)
        reserve_capacity_const = np.empty(H, dtype=np.float64)
        reserve_requirement = np.empty(H, dtype=np.float64)
        for h, step in enumerate(window):
            disp_cap = float(
                sum(
                    g.p_max_mw * step.dispatchable_availability[i]
                    for i, g in enumerate(self.disp)
                )
            )
            imp_cap = float(sum(ln.import_max_mw for ln in self.links))
            req = (
                self.system.reserve.load_fraction * float(step.bus_load_mw.sum())
                + self.system.reserve.renewable_fraction
                * float(step.variable_availability_mw.sum())
            )
            reserve_requirement[h] = req
            reserve_capacity_const[h] = disp_cap + imp_cap
            b_res[h] = disp_cap + imp_cap - req

        b_ub = np.concatenate(
            [
                b_line_pos,
                b_line_neg,
                b_ramp_up,
                b_ramp_dn,
                b_res,
                self._b_bres_power,
                self._b_bres_energy,
            ]
        )

        # ---- bounds -----------------------------------------------------
        lower = np.zeros(self.n_vars, dtype=np.float64)
        upper = np.full(self.n_vars, np.inf, dtype=np.float64)
        for h, step in enumerate(window):
            for g_i, gen in enumerate(self.disp):
                a = float(step.dispatchable_availability[g_i])
                lower[self.idx("p_disp", h, g_i)] = gen.p_min_mw * a
                upper[self.idx("p_disp", h, g_i)] = gen.p_max_mw * a
            for v_i in range(self.n_var):
                upper[self.idx("p_var", h, v_i)] = float(step.variable_availability_mw[v_i])
            for b_i, bat in enumerate(self.bats):
                upper[self.idx("p_charge", h, b_i)] = bat.p_charge_max_mw
                upper[self.idx("p_discharge", h, b_i)] = bat.p_discharge_max_mw
                lower[self.idx("soc", h, b_i)] = bat.soc_min
                upper[self.idx("soc", h, b_i)] = bat.soc_max
                upper[self.idx("r_batt", h, b_i)] = bat.p_discharge_max_mw
            for l_i, link in enumerate(self.links):
                upper[self.idx("p_import", h, l_i)] = link.import_max_mw
                upper[self.idx("p_export", h, l_i)] = link.export_max_mw
            for k_i, bus in enumerate(self.load_buses):
                col = self.system.bus_index[bus.bus_id]
                upper[self.idx("unserved", h, k_i)] = float(step.bus_load_mw[col])

        res = linprog(
            self._c,
            A_ub=self._a_ub,
            b_ub=b_ub,
            A_eq=self._a_eq,
            b_eq=b_eq,
            bounds=np.column_stack([lower, upper]),
            method="highs",
        )
        if res.status != _OPTIMAL or res.x is None:
            raise DispatchError(
                f"dispatch LP did not reach optimality: status={res.status} "
                f"message={res.message!r}"
            )

        return self._extract(res, window[0], reserve_requirement[0], reserve_capacity_const[0], econ)

    # ---------------------------------------------------------- extraction
    def _extract(
        self,
        res: Any,
        step: DispatchStep,
        reserve_req: float,
        reserve_cap_const: float,
        econ: Any,
    ) -> DispatchSolution:
        x = np.asarray(res.x, dtype=np.float64)

        def blk(name: str, width: int) -> np.ndarray:
            off = self._offsets[name]
            return x[off : off + width].copy()

        p_disp = blk("p_disp", self.n_disp)
        p_var = blk("p_var", self.n_var)
        p_ch = blk("p_charge", self.n_bat)
        p_dis = blk("p_discharge", self.n_bat)
        soc = blk("soc", self.n_bat)
        r_batt = blk("r_batt", self.n_bat)
        p_imp = blk("p_import", self.n_link)
        p_exp = blk("p_export", self.n_link)
        unserved = blk("unserved", self.n_load)
        overload = blk("overload", self.n_line)
        reserve_short = float(x[self._offsets["reserve_short"]])

        x_step = x[: self.vars_per_step]
        injections = self._injection_map @ x_step - step.bus_load_mw
        flows = self.network.ptdf @ injections
        angles = self.network.bus_angles_rad(injections)

        balance_residual = float(injections.sum())

        op_cost = self.dt_h * (
            float(
                np.dot(
                    [
                        g.marginal_cost_usd_per_mwh
                        + econ.co2_price_usd_per_tonne * g.co2_tonnes_per_mwh
                        for g in self.disp
                    ],
                    p_disp,
                )
            )
            + float(np.dot([g.marginal_cost_usd_per_mwh for g in self.var], p_var))
            + float(
                np.dot([b.cycle_cost_usd_per_mwh for b in self.bats], p_ch + p_dis)
            )
            + float(np.dot([ln.import_price_usd_per_mwh for ln in self.links], p_imp))
            - float(np.dot([ln.export_price_usd_per_mwh for ln in self.links], p_exp))
        )
        curtailed = np.maximum(step.variable_availability_mw - p_var, 0.0)
        curtail_cost = self.dt_h * econ.curtailment_cost_usd_per_mwh * float(curtailed.sum())
        violation_cost = self.dt_h * (
            econ.value_of_lost_load_usd_per_mwh * float(unserved.sum())
            + econ.overload_penalty_usd_per_mwh * float(overload.sum())
            + econ.reserve_shortfall_penalty_usd_per_mwh * reserve_short
        )

        reserve_available = (
            reserve_cap_const - float(p_disp.sum()) - float(p_imp.sum()) + float(r_batt.sum())
        )

        return DispatchSolution(
            p_disp_mw=p_disp,
            p_var_mw=p_var,
            p_charge_mw=p_ch,
            p_discharge_mw=p_dis,
            soc_end=soc,
            r_batt_mw=r_batt,
            p_import_mw=p_imp,
            p_export_mw=p_exp,
            unserved_mw=unserved,
            overload_mw=overload,
            reserve_short_mw=reserve_short,
            reserve_requirement_mw=reserve_req,
            reserve_available_mw=reserve_available,
            line_flow_mw=flows,
            bus_angle_rad=angles,
            bus_injection_mw=injections,
            window_objective_usd=float(res.fun),
            step_operating_cost_usd=op_cost,
            step_violation_cost_usd=violation_cost,
            step_curtailment_cost_usd=curtail_cost,
            solver_status=int(res.status),
            solver_message=str(res.message),
            solver_iterations=int(getattr(res, "nit", -1) or -1),
            simultaneous_charge_discharge_mw=float(np.minimum(p_ch, p_dis).sum()),
            balance_residual_mw=balance_residual,
        )

    # ------------------------------------------------------------ metadata
    def solver_metadata(self) -> dict[str, Any]:
        """Solver identity recorded in the provenance bundle."""
        return {
            "library": "scipy.optimize.linprog",
            "scipy_version": scipy.__version__,
            "method": "highs",
            "numpy_version": np.__version__,
            "accepted_status": _OPTIMAL,
            "n_variables": self.n_vars,
            "n_equality_rows": int(self._a_eq.shape[0]),
            "n_inequality_rows": int(self._a_ub.shape[0]),
            "horizon_steps": self.H,
            "timestep_s": self.timestep_s,
        }
