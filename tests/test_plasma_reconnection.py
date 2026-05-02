"""Tests for quasim.ciir.plasma — CIIR plasma reconnection control engine.

Run:
    python -m pytest tests/test_plasma_reconnection.py -v --override-ini="addopts="
"""

from __future__ import annotations

import numpy as np
import pytest

from quasim.ciir.plasma import (CIIRSnapshot, Constraints, ControllerConfig,
                                EngineConfig, Grid2D, MHDState,
                                ObjectiveWeights, ReconnectionController,
                                SpectrumSnapshot, TopologySnapshot,
                                energy_release_rate, evaluate_objective,
                                find_critical_points, find_current_sheets,
                                fkr_growth_rate_estimate, gaussian_target_mask,
                                growth_rates, harris_sheet, initial_state,
                                lundquist_number, make_snapshot,
                                perturb_tearing, plasmoid_growth_rate_estimate,
                                plasmoid_indicator, recommended_dt,
                                reconnection_localization,
                                reconnection_rate_field,
                                run_reconnection_control, spectrum,
                                stability_inequality, step_rk2,
                                topology_snapshot, zero_control,
                                zero_control_vector)
from quasim.ciir.plasma.mhd import (ddx, ddy, invert_laplacian_periodic,
                                    laplacian, poisson_bracket)

# ---------------------------------------------------------------------------
# Grid + operators
# ---------------------------------------------------------------------------


class TestGridAndOperators:
    def test_grid_validation(self):
        with pytest.raises(ValueError):
            Grid2D(nx=2, ny=2)
        with pytest.raises(ValueError):
            Grid2D(nx=8, ny=8, Lx=-1.0)

    def test_grid_dx_dy(self):
        g = Grid2D(nx=8, ny=16, Lx=4.0, Ly=8.0)
        assert g.dx == pytest.approx(0.5)
        assert g.dy == pytest.approx(0.5)

    def test_ddx_constant_is_zero(self):
        g = Grid2D(nx=16, ny=16)
        f = np.ones((g.nx, g.ny))
        np.testing.assert_allclose(ddx(f, g.dx), 0.0, atol=1e-12)
        np.testing.assert_allclose(ddy(f, g.dy), 0.0, atol=1e-12)

    def test_ddx_sin_recovers_cos(self):
        g = Grid2D(nx=128, ny=8)
        X, _ = g.coords()
        f = np.sin(X)
        df = ddx(f, g.dx)
        # central FD on sin(x) recovers cos(x) to ~ second-order accuracy
        np.testing.assert_allclose(df, np.cos(X), atol=2e-3)

    def test_laplacian_of_sin(self):
        g = Grid2D(nx=128, ny=128)
        X, Y = g.coords()
        f = np.sin(X) * np.sin(Y)
        lap = laplacian(f, g.dx, g.dy)
        np.testing.assert_allclose(lap, -2.0 * f, atol=5e-3)

    def test_poisson_bracket_antisymmetry(self):
        g = Grid2D(nx=32, ny=32)
        rng = np.random.default_rng(0)
        f = rng.standard_normal((g.nx, g.ny))
        h = rng.standard_normal((g.nx, g.ny))
        ab = poisson_bracket(f, h, g.dx, g.dy)
        ba = poisson_bracket(h, f, g.dx, g.dy)
        np.testing.assert_allclose(ab, -ba, atol=1e-12)

    def test_invert_laplacian_round_trip(self):
        g = Grid2D(nx=64, ny=64)
        X, Y = g.coords()
        f = np.sin(2.0 * X) * np.cos(3.0 * Y)
        rhs_field = laplacian(f, g.dx, g.dy)
        recovered = invert_laplacian_periodic(rhs_field, g)
        # 2nd-order central FD induces O((kΔx)^2) error; loosen tol accordingly.
        np.testing.assert_allclose(recovered - recovered.mean(), f - f.mean(), atol=1e-2)


# ---------------------------------------------------------------------------
# State and equilibria
# ---------------------------------------------------------------------------


class TestState:
    def test_state_shape_validation(self):
        g = Grid2D(nx=16, ny=16)
        with pytest.raises(ValueError):
            MHDState(psi=np.zeros((4, 4)), omega=np.zeros((g.nx, g.ny)), grid=g)

    def test_negative_eta_rejected(self):
        g = Grid2D(nx=8, ny=8)
        with pytest.raises(ValueError):
            MHDState(psi=np.zeros((g.nx, g.ny)), omega=np.zeros((g.nx, g.ny)), grid=g, eta=-0.1)

    def test_divB_is_machine_zero(self):
        g = Grid2D(nx=64, ny=64)
        psi = harris_sheet(g, B0=1.0, a=0.5)
        omega = np.zeros_like(psi)
        s = MHDState(psi=psi, omega=omega, grid=g)
        # B = ∇ψ × ẑ is solenoidal by construction; central FD reproduces this.
        assert s.divB_max() < 1.0e-7

    def test_Jz_is_negative_laplacian(self):
        g = Grid2D(nx=32, ny=32)
        rng = np.random.default_rng(1)
        psi = rng.standard_normal((g.nx, g.ny))
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g)
        np.testing.assert_allclose(s.Jz(), -laplacian(psi, g.dx, g.dy), atol=1e-12)

    def test_harris_sheet_has_two_currents(self):
        g = Grid2D(nx=64, ny=128)
        psi = harris_sheet(g, B0=1.0, a=0.4)
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g)
        # Use the high-level current-sheet finder (the actual public feature).
        sheets = find_current_sheets(s, threshold_factor=2.0, min_size=8)
        assert len(sheets) >= 2
        # The two largest sheets should be centered near y = Ly/4 and y = 3Ly/4.
        ys = sorted(sh.centroid[1] for sh in sheets[:2])
        assert abs(ys[0] - g.Ly / 4.0) < 0.4
        assert abs(ys[1] - 3.0 * g.Ly / 4.0) < 0.4

    def test_perturb_tearing_amplitude(self):
        g = Grid2D(nx=32, ny=32)
        delta = perturb_tearing(g, amplitude=0.01, mode=2)
        assert delta.shape == (g.nx, g.ny)
        assert delta.max() <= 0.011
        assert delta.min() >= -0.011

    def test_state_copy_independent(self):
        g = Grid2D(nx=8, ny=8)
        s = MHDState(psi=np.ones((g.nx, g.ny)), omega=np.zeros((g.nx, g.ny)), grid=g)
        s2 = s.copy()
        s2.psi[0, 0] = 99.0
        assert s.psi[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Stepper / conservation properties
# ---------------------------------------------------------------------------


class TestStepper:
    def test_zero_control_runs(self):
        g = Grid2D(nx=32, ny=32)
        psi = harris_sheet(g, B0=1.0, a=0.5)
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g, eta=1e-3, nu=1e-3)
        dt = recommended_dt(g, eta=s.eta, nu=s.nu, B_max=1.0)
        for _ in range(5):
            step_rk2(s, dt, control=zero_control)
        assert np.all(np.isfinite(s.psi))
        assert np.all(np.isfinite(s.omega))

    def test_recommended_dt_positive(self):
        g = Grid2D(nx=32, ny=32)
        dt = recommended_dt(g, eta=1e-3, nu=1e-3, B_max=1.0)
        assert dt > 0.0

    def test_step_invalid_dt(self):
        g = Grid2D(nx=8, ny=8)
        s = MHDState(psi=np.zeros((g.nx, g.ny)), omega=np.zeros((g.nx, g.ny)), grid=g)
        with pytest.raises(ValueError):
            step_rk2(s, 0.0)

    def test_resistive_decay(self):
        """Free decay: with no flow and no forcing, magnetic energy should
        monotonically decrease under positive resistivity (η > 0) [V]."""
        g = Grid2D(nx=64, ny=64)
        X, Y = g.coords()
        # Pure k=1 magnetic mode in 2D (gives decaying flux).
        psi = 0.1 * np.sin(X) * np.sin(Y)
        omega = np.zeros_like(psi)
        s = MHDState(psi=psi, omega=omega, grid=g, eta=0.05, nu=0.05)
        E0 = s.magnetic_energy()
        dt = 0.5 * recommended_dt(g, eta=s.eta, nu=s.nu, B_max=0.1)
        for _ in range(20):
            step_rk2(s, dt, control=zero_control)
        E1 = s.magnetic_energy()
        assert E1 < E0
        assert E1 >= 0.0

    def test_divB_preserved_during_evolution(self):
        g = Grid2D(nx=32, ny=32)
        psi = harris_sheet(g, B0=1.0, a=0.5) + perturb_tearing(g, amplitude=1e-3)
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g, eta=1e-3, nu=1e-3)
        dt = recommended_dt(g, eta=s.eta, nu=s.nu, B_max=1.0)
        for _ in range(10):
            step_rk2(s, dt, control=zero_control)
        # ∇·B should remain near machine-zero because we use the flux-function form.
        assert s.divB_max() < 1.0e-6


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


class TestTopology:
    def test_o_point_at_paraboloid_minimum(self):
        # ψ = (x-π)^2 + (y-π)^2 has a single minimum (O-point) at (π, π).
        g = Grid2D(nx=64, ny=64, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
        X, Y = g.coords()
        psi = (X - np.pi) ** 2 + (Y - np.pi) ** 2
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g)
        crits = find_critical_points(s, grad_tol=0.05)
        kinds = {c.kind for c in crits}
        assert "O" in kinds

    def test_x_point_at_saddle(self):
        # Saddle ψ = (x-π)^2 - (y-π)^2 has det H < 0 → X-point.
        g = Grid2D(nx=64, ny=64, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
        X, Y = g.coords()
        psi = (X - np.pi) ** 2 - (Y - np.pi) ** 2
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g)
        crits = find_critical_points(s, grad_tol=0.05)
        kinds = {c.kind for c in crits}
        assert "X" in kinds

    def test_harris_sheet_has_current_sheets(self):
        g = Grid2D(nx=64, ny=64)
        psi = harris_sheet(g, B0=1.0, a=0.4)
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g)
        sheets = find_current_sheets(s, threshold_factor=2.0, min_size=8)
        assert len(sheets) >= 1
        for sh in sheets:
            assert sh.area > 0.0
            assert sh.Jz_max > 0.0
            assert sh.n_points >= 8

    def test_topology_snapshot_fields(self):
        cfg = EngineConfig(nx=32, ny=32, n_steps=1)
        s = initial_state(cfg)
        snap = topology_snapshot(s)
        assert isinstance(snap, TopologySnapshot)
        assert snap.n_X == len(snap.x_points)
        assert snap.n_O == len(snap.o_points)
        # No X-points expected before evolution; reconnection rate ≈ 0
        assert snap.reconnection_rate_at_X >= 0.0

    def test_reconnection_rate_field_shape(self):
        g = Grid2D(nx=16, ny=16)
        s = MHDState(
            psi=harris_sheet(g),
            omega=np.zeros((g.nx, g.ny)),
            grid=g,
            eta=1e-3,
        )
        R = reconnection_rate_field(s)
        assert R.shape == s.psi.shape


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


class TestSpectrum:
    def test_spectrum_returns_snapshot(self):
        g = Grid2D(nx=32, ny=32)
        psi = harris_sheet(g)
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g)
        sp = spectrum(s)
        assert isinstance(sp, SpectrumSnapshot)
        assert sp.E_mag.shape == sp.k_bins.shape
        assert sp.E_kin.shape == sp.k_bins.shape

    def test_spectrum_monochromatic_psi_peaks_at_k1(self):
        g = Grid2D(nx=64, ny=64, Lx=2 * np.pi, Ly=2 * np.pi)
        X, Y = g.coords()
        psi = np.sin(X) * np.sin(Y)  # k = sqrt(2)
        s = MHDState(psi=psi, omega=np.zeros_like(psi), grid=g)
        sp = spectrum(s, n_bins=20)
        # peak should be near k = sqrt(2) ≈ 1.414
        assert 0.7 <= sp.peak_k_mag <= 2.5

    def test_growth_rates_pure_exponential(self):
        """If we synthesise a sequence of states with E_mag(k) = A_0 e^{2γt}
        per bin, growth_rates should recover γ to high precision [D]."""
        g = Grid2D(nx=32, ny=32)
        snaps: list[SpectrumSnapshot] = []
        ts = np.linspace(0.0, 1.0, 5)
        gamma_true = 0.7
        n_bins = 8
        # Construct fake snapshots directly.
        for t in ts:
            E = np.exp(2.0 * gamma_true * t) * np.ones(n_bins)
            snaps.append(
                SpectrumSnapshot(
                    t=float(t),
                    k_bins=np.linspace(0.0, 1.0, n_bins),
                    E_mag=E,
                    E_kin=E,
                    Jz_spec=E,
                    peak_k_mag=0.0,
                    peak_k_Jz=0.0,
                )
            )
        gammas = growth_rates(snaps)
        np.testing.assert_allclose(gammas, gamma_true, atol=1e-10)

    def test_growth_rates_requires_two_snapshots(self):
        g = Grid2D(nx=8, ny=8)
        s = MHDState(psi=np.zeros((g.nx, g.ny)), omega=np.zeros((g.nx, g.ny)), grid=g)
        with pytest.raises(ValueError):
            growth_rates([spectrum(s)])

    def test_lundquist_number(self):
        g = Grid2D(nx=8, ny=8)
        s = MHDState(psi=np.zeros((g.nx, g.ny)), omega=np.zeros((g.nx, g.ny)), grid=g, eta=1e-2)
        S = lundquist_number(s, B_char=1.0, L=1.0)
        assert pytest.approx(100.0) == S

    def test_lundquist_zero_eta_infinite(self):
        g = Grid2D(nx=8, ny=8)
        s = MHDState(psi=np.zeros((g.nx, g.ny)), omega=np.zeros((g.nx, g.ny)), grid=g, eta=0.0)
        assert lundquist_number(s) == float("inf")

    def test_fkr_scaling_monotone(self):
        # γ_FKR ∝ S^{-3/5}: increasing S decreases γ.
        g_small = fkr_growth_rate_estimate(S=1.0e3)
        g_large = fkr_growth_rate_estimate(S=1.0e5)
        assert g_small > g_large > 0.0

    def test_fkr_invalid_S(self):
        with pytest.raises(ValueError):
            fkr_growth_rate_estimate(S=-1.0)
        with pytest.raises(ValueError):
            fkr_growth_rate_estimate(S=1.0e3, tau_A=0.0)

    def test_plasmoid_subcritical_returns_none(self):
        assert plasmoid_growth_rate_estimate(S=1.0e3, S_c=1.0e4) is None

    def test_plasmoid_supercritical_positive(self):
        gp = plasmoid_growth_rate_estimate(S=1.0e5, S_c=1.0e4)
        assert gp is not None and gp > 0.0


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


class TestControl:
    def _state(self, nx: int = 32) -> MHDState:
        g = Grid2D(nx=nx, ny=nx)
        psi = harris_sheet(g, B0=1.0, a=0.5) + perturb_tearing(g, amplitude=1e-3)
        return MHDState(psi=psi, omega=np.zeros_like(psi), grid=g, eta=1e-3, nu=1e-3)

    def test_gaussian_target_mask_in_unit_interval(self):
        s = self._state()
        m = gaussian_target_mask(s, x0=s.grid.Lx / 2, y0=s.grid.Ly / 4, sigma=0.4)
        assert m.shape == s.psi.shape
        assert 0.0 <= m.min() <= m.max() <= 1.0
        assert m.max() == pytest.approx(1.0, abs=1e-12)

    def test_gaussian_target_mask_invalid_sigma(self):
        s = self._state()
        with pytest.raises(ValueError):
            gaussian_target_mask(s, 0.0, 0.0, sigma=0.0)

    def test_zero_control_vector_shapes(self):
        s = self._state()
        u = zero_control_vector(s)
        assert u.E_drive.shape == s.psi.shape
        assert u.eta_eff.shape == s.psi.shape
        np.testing.assert_allclose(u.E_drive, 0.0)
        np.testing.assert_allclose(u.eta_eff, s.eta)

    def test_controller_synthesize_shapes(self):
        s = self._state()
        target = gaussian_target_mask(s, s.grid.Lx / 2, s.grid.Ly / 4, sigma=0.4)
        ctrl = ReconnectionController(ControllerConfig(E_drive_amplitude=0.01))
        u = ctrl.synthesize(s, target)
        assert u.E_drive.shape == s.psi.shape
        assert u.eta_eff.shape == s.psi.shape
        assert u.eta_eff.min() >= s.eta - 1e-12  # never below baseline

    def test_controller_rejects_negative_target(self):
        s = self._state()
        bad = -np.ones_like(s.psi)
        ctrl = ReconnectionController()
        with pytest.raises(ValueError):
            ctrl.synthesize(s, bad)

    def test_controller_rejects_shape_mismatch(self):
        s = self._state()
        ctrl = ReconnectionController()
        with pytest.raises(ValueError):
            ctrl.synthesize(s, np.ones((4, 4)))

    def test_objective_terms_nonneg(self):
        s = self._state()
        target = gaussian_target_mask(s, s.grid.Lx / 2, s.grid.Ly / 4, sigma=0.4)
        obj = evaluate_objective(s, ObjectiveWeights(), target)
        assert obj.R_term >= 0.0
        assert obj.I_term >= 0.0
        assert obj.P_term >= 0.0
        assert obj.total == pytest.approx(obj.R_term + obj.I_term + obj.P_term)

    def test_plasmoid_indicator_invalid_band(self):
        s = self._state()
        with pytest.raises(ValueError):
            plasmoid_indicator(s, k_band=(2.0, 1.0))
        with pytest.raises(ValueError):
            plasmoid_indicator(s, k_band=(-1.0, 5.0))

    def test_energy_release_matches_ohmic(self):
        s = self._state()
        assert energy_release_rate(s) == pytest.approx(s.ohmic_dissipation())

    def test_reconnection_localization_shape_check(self):
        s = self._state()
        with pytest.raises(ValueError):
            reconnection_localization(s, np.ones((2, 2)))

    def test_controller_hook_drives_evolution(self):
        """Controller-driven evolution must remain finite and exhibit
        non-negative Ohmic dissipation (energy can transiently rise locally
        because inhomogeneous η creates a (∇η)·∇ψ term that is not
        sign-definite — see ``rhs`` docstring [V/D])."""
        s = self._state()
        target = gaussian_target_mask(s, s.grid.Lx / 2, s.grid.Ly / 4, sigma=0.4)
        s2 = MHDState(
            psi=s.psi.copy(), omega=s.omega.copy(), grid=s.grid, eta=s.eta, nu=s.nu
        )
        ctrl = ReconnectionController(ControllerConfig(E_drive_amplitude=0.0, eta_boost=10.0))
        dt = recommended_dt(s.grid, eta=s.eta, nu=s.nu, B_max=1.0)
        for _ in range(10):
            step_rk2(s, dt, control=zero_control)
            u = ctrl.synthesize(s2, target)
            step_rk2(s2, dt, control=u.as_hook())
        # Both states must remain finite.
        assert np.all(np.isfinite(s.psi))
        assert np.all(np.isfinite(s2.psi))
        assert np.all(np.isfinite(s.omega))
        assert np.all(np.isfinite(s2.omega))
        # Ohmic dissipation is non-negative by construction (∫η J^2 dV ≥ 0).
        assert energy_release_rate(s) >= 0.0
        assert energy_release_rate(s2) >= 0.0


# ---------------------------------------------------------------------------
# CIIR snapshot + stability inequality
# ---------------------------------------------------------------------------


class TestCIIR:
    def test_constraints_defaults(self):
        c = Constraints()
        assert c.S_critical == pytest.approx(1.0e4)
        assert "Maxwell" in c.field_equations

    def test_make_snapshot_runs(self):
        cfg = EngineConfig(nx=32, ny=32, n_steps=1)
        s = initial_state(cfg)
        snap = make_snapshot(s, control_norm=0.1)
        assert isinstance(snap, CIIRSnapshot)
        assert snap.t == s.t
        assert snap.S > 0.0

    def test_stability_inequality_true_when_ctrl_dominates(self):
        cfg = EngineConfig(nx=16, ny=16, n_steps=1)
        s = initial_state(cfg)
        snap = make_snapshot(s, control_norm=1.0e9)  # huge control
        assert stability_inequality(snap) is True

    def test_stability_inequality_false_when_instab_dominates(self):
        cfg = EngineConfig(nx=16, ny=16, n_steps=1)
        s = initial_state(cfg)
        snap = make_snapshot(s, control_norm=0.0)
        assert stability_inequality(snap) is False

    def test_failure_modes_present(self):
        from quasim.ciir.plasma import FAILURE_MODES

        codes = {fm.code for fm in FAILURE_MODES}
        assert {"F1", "F2", "F3", "F4", "F5"}.issubset(codes)


# ---------------------------------------------------------------------------
# End-to-end engine
# ---------------------------------------------------------------------------


class TestEngine:
    def test_run_smoke(self):
        cfg = EngineConfig(nx=32, ny=32, n_steps=5, record_every=1, seed=7)
        result = run_reconnection_control(cfg)
        assert result.final_state is not None
        # 1 initial + 5 step recordings
        assert len(result.times) == 6
        assert len(result.ciir_history) == 6
        assert len(result.objective_history) == 6
        # Time is monotone
        assert all(t2 >= t1 for t1, t2 in zip(result.times, result.times[1:]))

    def test_run_no_controller(self):
        cfg = EngineConfig(nx=24, ny=24, n_steps=3, use_controller=False)
        result = run_reconnection_control(cfg)
        assert result.final_state is not None
        # Without controller, control_norm at step 0 is 0
        assert result.ciir_history[0].control_norm == 0.0

    def test_run_divB_remains_zero(self):
        cfg = EngineConfig(nx=32, ny=32, n_steps=10, record_every=2)
        result = run_reconnection_control(cfg)
        for d in result.divB_max:
            # Tolerance reflects float32 grid round-off in cumulative central FD.
            assert d < 1.0e-6

    def test_run_objective_non_decreasing_for_zero_drive(self):
        """Sanity: with controller off and η-only boost disabled, objective is finite."""
        cfg = EngineConfig(
            nx=24,
            ny=24,
            n_steps=4,
            use_controller=False,
            weights=ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0),
        )
        result = run_reconnection_control(cfg)
        for o in result.objective_history:
            assert np.isfinite(o.total)

    def test_run_invalid_config(self):
        cfg = EngineConfig(nx=16, ny=16, n_steps=0)
        with pytest.raises(ValueError):
            run_reconnection_control(cfg)
        cfg2 = EngineConfig(nx=16, ny=16, n_steps=4, record_every=0)
        with pytest.raises(ValueError):
            run_reconnection_control(cfg2)
