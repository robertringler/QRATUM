# Methods Documentation

## Overview

This simulation framework tests the conditional no-go theorem on backward lab-time signaling in compiled/analog realities.

## Core Diagnostics

### S_lab(θ): Lab-Time Signaling Diagnostic

**Definition:**

```
S_lab(θ) = sup_{O,I∈A} | ∂P(O_{t1}) / ∂I_{t2} |   for t1 < t2
```

**Physical Interpretation:**

- Measures whether future interventions at time t2 can causally influence past observables at t1 < t2
- S_lab = 0 indicates no backward signaling (causality preserved)
- S_lab > 0 would indicate acausal information flow

**Computational Method:**

1. Run twin simulations with I=0 and I=1 interventions at t2
2. Use identical seeds to ensure same initial conditions
3. Measure observable O at t1 < t2 for both runs
4. Compute |O_I1(t1) - O_I0(t1)| averaged over multiple samples
5. Statistical error estimated via bootstrapping

**Causality Audit:**

- Function computing O_{t1} cannot access data arrays indexed > t1
- Twin-run invariance verified: states identical before t2
- Postselection labels quarantined in separate files

### χ(θ): Effective-Time Loop Diagnostic

**Definition:**

```
χ(θ) = inf_{γ∈C} ∮_γ g_eff(θ)_{μν} dx^μ dx^ν
```

**Physical Interpretation:**

- Measures closed timelike curves in effective metric
- χ < 0 indicates effective-time loops (internal to effective metric)
- **Critical**: χ < 0 does NOT necessarily imply S_lab ≠ 0

**Computational Method:**

1. Reconstruct effective metric from dispersion relation ω(k)
2. Extract effective light speed: c_eff = dω/dk |_{k→0}
3. Compute loop functional with flow velocity parameter
4. χ ~ v² - c_eff² with dispersion corrections

**Validity Certificate:**

χ is only reported if all validity checks pass:

- Dispersion deviation < 0.10
- Attenuation margin > 1.5
- Reconstruction error < 0.10

## Model Classes

### M2: CPTP Process-Tensor Model

**Description:** Finite-dimensional quantum system with Lindblad master equation

**Dynamics:**

```
dρ/dt = -i[H, ρ] + Σ_k γ(L_k ρ L_k† - 1/2{L_k†L_k, ρ})
```

**Parameters:**

- dimension: Hilbert space dimension (2, 4)
- environment_coupling (γ): Coupling to environment (0.01, 0.1, 0.5)
- t1_t2_separation: Time steps between measurement and intervention (5, 10, 20)

**Interventions:**

- I=0: No intervention (identity)
- I=1: Apply unitary rotation at t2

**Observable:**

- Ground state population at t1: O = ⟨0|ρ(t1)|0⟩

### M3: Lattice Field Model

**Description:** 1+1D discrete lattice with tunable dispersion

**Dynamics:**

- Spectral evolution with dispersion relation ω(k)
- Flow term for effective light cone tilting
- Dissipation and momentum cutoff

**Parameters:**

- n_sites: Lattice sites (32, 64) - boundary parameter
- cutoff_scale: Momentum cutoff (10, 100, 1000) - bulk parameter
- dispersion_strength: Dispersion parameter (0.0, 0.5, 1.0, 2.0) - bulk parameter
- flow_velocity_ratio: v/c ratio (0.5, 0.7, 0.9, 0.95, 1.0)
- dissipation_rate: Energy dissipation (0.001, 0.01) - bulk parameter

**Interventions:**

- I=0: No intervention
- I=1: Apply localized Gaussian perturbation at t2

**Observable:**

- Boundary energy at t1: O = |ψ(x=0, t1)|² + |ψ(x=L, t1)|²

## Anti-Holographic Hypothesis

**Statement:** χ is more sensitive to bulk parameters than boundary parameters

**Bulk Parameters:** cutoff_scale, dispersion_strength, dissipation_rate
**Boundary Parameters:** n_sites (system size)

**Test Method:**

1. Compute finite-difference sensitivity: ∂χ/∂θ ≈ Δχ/Δθ
2. Bootstrap confidence intervals
3. Compare aggregate bulk vs boundary sensitivities

## Pre-Registered Hypotheses

**H1**: S_lab(θ) = 0 within statistical error for all θ satisfying assumptions

**H2**: χ(θ) may become negative in some parameter regimes

**H3**: Approaching χ < 0 triggers validity degradation or instability

**H4**: χ more sensitive to bulk parameters than boundary parameters

## Statistical Methods

**Significance Testing:**

- S_lab compared to estimated standard error
- Threshold: 5σ for claiming significance
- Two-sample t-test for I=0 vs I=1 comparison

**Bootstrap:**

- 100 bootstrap samples for sensitivity confidence intervals
- 95% confidence level

## Reproducibility

**Deterministic:**

- Fixed base seed: 42
- Seed sequences generated deterministically
- All random number generators seeded

**Auditable:**

- SHA256 hashes of config, code, results
- Audit log with causality checks
- Full parameter grid locked in configs/
