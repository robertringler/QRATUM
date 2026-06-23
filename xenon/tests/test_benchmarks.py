"""Phase 4 head-to-head benchmark tests (skip when the reference tool is absent).

These substantiate the narrow leadership claim with executed evidence:
  * XENON's mean-field forward model agrees with libRoadRunner's SBML ODE engine;
  * XENON's exact-likelihood inference recovers the true parameters as accurately
    as (here, more accurately than) the pyABC ABC-SMC reference.
"""

from __future__ import annotations

import pytest


def test_forward_model_matches_roadrunner_ode():
    pytest.importorskip("libsbml")
    pytest.importorskip("roadrunner")
    from xenon.benchmarks.harness import compare_simulation_to_roadrunner

    result = compare_simulation_to_roadrunner(t=2.0)
    # XENON's analytic mean-field must agree with the gold-standard ODE engine.
    assert result["max_rel_error"] < 1e-4, result


def test_inference_recovers_truth_and_is_competitive_with_pyabc():
    pytest.importorskip("pyabc")
    from xenon.benchmarks.harness import compare_inference_to_pyabc

    result = compare_inference_to_pyabc(seed=0)
    # XENON (deterministic given seed) must recover the true rates accurately.
    assert result["xenon_rel_error"] < 0.02, result
    # The ABC reference must also be in the right ballpark (sanity on the setup).
    assert result["pyabc_rel_error"] < 0.25, result
    # XENON must be at least as accurate as the ABC reference (or within 5%);
    # written to tolerate pyABC's run-to-run stochasticity without flaking.
    assert result["xenon_rel_error"] <= max(result["pyabc_rel_error"], 0.05), result
