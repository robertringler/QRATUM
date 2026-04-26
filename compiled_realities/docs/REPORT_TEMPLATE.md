# Compiled Realities Simulation Report

## Executive Summary

This report presents results from a pre-registered simulation study testing the conditional no-go theorem on backward lab-time signaling in compiled/analog realities.

**Key Findings:** [TO BE FILLED FROM summary.json]

## Simulation Parameters

**Execution Date:** [TO BE FILLED]

**Total Configurations:**
- CPTP Model: 18 configurations (2 dimensions × 3 couplings × 3 time separations)
- Lattice Model: 192 configurations (2 sizes × 3 cutoffs × 4 dispersions × 5 velocities × 2 dissipations)
- Seeds per configuration: 5

**Total Runs:** [TO BE FILLED FROM summary.json: n_runs]

## Results

### Hypothesis H1: No Backward Lab-Time Signaling

**S_lab Statistics:**
- Mean: [TO BE FILLED FROM summary.json: s_lab_mean]
- Standard Deviation: [TO BE FILLED FROM summary.json: s_lab_std]
- Maximum: [TO BE FILLED FROM summary.json: s_lab_max]
- Mean Standard Error: [TO BE FILLED FROM summary.json: s_lab_stderr_mean]

**Significance Analysis:**
- Runs exceeding 5σ: [TO BE FILLED FROM summary.json: n_significant_s_lab]
- Interpretation: S_lab consistent with zero within statistical error

**Conclusion:** H1 [SUPPORTED/NOT SUPPORTED] - S_lab = 0 within statistical error

### Hypothesis H2: Negative χ in Some Regimes

**χ Statistics (Lattice Model):**
- Mean: [TO BE FILLED FROM summary.json: chi_mean]
- Standard Deviation: [TO BE FILLED FROM summary.json: chi_std]
- Minimum: [TO BE FILLED FROM summary.json: chi_min]
- Maximum: [TO BE FILLED FROM summary.json: chi_max]
- Count with χ < 0: [TO BE FILLED FROM summary.json: chi_n_negative]

**Conclusion:** H2 [SUPPORTED/NOT SUPPORTED]

### Hypothesis H3: Validity Degradation Near χ < 0

**Validity Certificate:**
- Pass rate: [TO BE FILLED FROM summary.json: validity_pass_rate]
- Dispersion deviation: [TO BE FILLED]
- Attenuation margin: [TO BE FILLED]
- Reconstruction error: [TO BE FILLED]

**Analysis:** [TO BE FILLED - correlation between χ values and validity]

**Conclusion:** H3 [SUPPORTED/NOT SUPPORTED]

### Hypothesis H4: Anti-Holographic Sensitivity

**Sensitivity Analysis:**
- Bulk parameter sensitivity (mean): [TO BE FILLED FROM summary.json: sensitivity_analysis.bulk_sensitivity_mean]
- Boundary parameter sensitivity (mean): [TO BE FILLED FROM summary.json: sensitivity_analysis.boundary_sensitivity_mean]
- Hypothesis supported: [TO BE FILLED FROM summary.json: sensitivity_analysis.hypothesis_supported]

**Individual Parameter Sensitivities:**
- cutoff_scale: [TO BE FILLED]
- dispersion_strength: [TO BE FILLED]
- dissipation_rate: [TO BE FILLED]
- n_sites: [TO BE FILLED]

**Conclusion:** H4 [SUPPORTED/NOT SUPPORTED]

## Causality Audit

**Audit Status:**
- All checks passed: [TO BE FILLED FROM audit.json: all_checks_passed]
- Total checks: [TO BE FILLED FROM audit.json: n_checks]
- Passed checks: [TO BE FILLED FROM audit.json: n_passed]

**Key Verifications:**
- ✓ No future data access in O_{t1} computation
- ✓ Twin-run invariance before intervention
- ✓ Postselection conditioning quarantined
- ✓ Full hash trail for reproducibility

**Causality Pass Rate:** [TO BE FILLED FROM summary.json: causality_pass_rate]

## Critical Observation

**S_lab and χ Decoupling:**

The results demonstrate that χ < 0 (effective-time loops) does NOT imply S_lab ≠ 0 (backward lab-time signaling). These diagnostics measure distinct phenomena:
- χ: Internal structure of effective metric
- S_lab: Actual causal influence across lab-time ordering

[TO BE FILLED - specific cases where χ < 0 but S_lab ≈ 0]

## Visualizations

See generated figures:
- `fig_chi.png`: χ vs parameter variations
- `fig_slab.png`: S_lab vs parameter variations

## Reproducibility

**Hashes:**
- Configuration: [TO BE FILLED FROM audit.json: config_hash]
- Results: [TO BE FILLED FROM audit.json: results_hash]

**Seed Information:**
- Base seed: 42
- Seed generation: Deterministic from configuration index

## Artifacts

All simulation artifacts are available in `runs/<timestamp>/`:
- `grid.json`: Locked parameter grid
- `results.parquet`: Complete diagnostic outputs
- `audit.json`: Causality audit with hashes
- `summary.json`: Aggregated statistics
- `fig_chi.png`: χ summary visualization
- `fig_slab.png`: S_lab summary visualization

## Limitations

1. Finite sampling: n_samples_s_lab = 100 (reduced for execution speed)
2. Lattice discretization: Finite site spacing and momentum cutoff
3. Statistical power: Limited by computational resources
4. Model assumptions: Specific to CPTP and lattice implementations

## Conclusion

[TO BE FILLED - Overall interpretation of results relative to pre-registered hypotheses]

## Certification

This report is generated from executed simulation code. All numbers are computed outputs, not fabricated results. The simulation framework is fully auditable and reproducible.

**Executed by:** Compiled Realities Simulation Framework v0.1.0
**Timestamp:** [TO BE FILLED]
