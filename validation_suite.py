"""
QuASIM Validation Suite — CI Regression Tests
Author: Robert Ringler
"""

import json
import numpy as np
import pytest

TOL_BLOCH = 1e-3
TOL_SHOT = 1e-2
TOL_PURITY = 5e-3

def load_json(fname):
    with open(fname) as f:
        return json.load(f)

def rho_from_block(block):
    return np.array(block["rho_final_re"]) + 1j * np.array(block["rho_final_im"])

def bloch_from_rho(rho):
    return np.array([2*np.real(rho[0,1]), 2*np.imag(rho[0,1]), np.real(rho[0,0]-rho[1,1])])

def purity(rho):
    return np.real(np.trace(rho @ rho))

def validate_consistency(block):
    """Check Bloch↔ρ↔counts consistency"""
    rho = rho_from_block(block)
    bloch_calc = bloch_from_rho(rho)
    bloch_reported = np.array(block["bloch_final"])
    assert np.allclose(bloch_calc, bloch_reported, atol=TOL_BLOCH), f"Bloch mismatch {bloch_calc} vs {bloch_reported}"

    for axis, idx in zip("XYZ", range(3)):
        probs = (1 + bloch_calc[idx]) / 2
        vals = block["shot_counts"][axis]
        p_emp = vals["+1"] / (vals["+1"] + vals["-1"])
        assert abs(p_emp - probs) < TOL_SHOT, f"Shot mismatch on {axis}"

    p = purity(rho)
    if "Purity" in block:
        assert abs(p - block["Purity"]) < TOL_PURITY
    return True

@pytest.mark.parametrize("fname", [
    "default_sim.json",
    "ideal_sim.json",
    "expm_sim.json",
    "su2_sim.json",
])
def test_quasim_blocks(fname):
    block = load_json(fname)
    assert validate_consistency(block)
    assert 0.0 <= block["fidelity_mean"] <= 1.0
    assert block["fidelity_mean"] >= 0.99 if "ideal" in fname else block["fidelity_mean"] >= 0.97

def test_purity_monotonic():
    """Ensure noisy run has purity < ideal run"""
    noisy = load_json("default_sim.json")
    ideal = load_json("ideal_sim.json")
    p_noisy = purity(rho_from_block(noisy))
    p_ideal = purity(rho_from_block(ideal))
    assert p_noisy < p_ideal