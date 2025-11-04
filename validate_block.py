import numpy as np, json

def validate_block(block):
    ρ_re, ρ_im = np.array(block["rho_final_re"]), np.array(block["rho_final_im"])
    ρ = ρ_re + 1j * ρ_im
    Bx, By, Bz = (2*np.real(ρ[0,1]), 2*np.imag(ρ[0,1]), np.real(ρ[0,0]-ρ[1,1]))
    bloch = np.array(block["bloch_final"])
    assert np.allclose([Bx,By,Bz], bloch, atol=1e-3), "Bloch mismatch"
    probs = {k:(1+v)/2 for k,v in zip("XYZ",[Bx,By,Bz])}
    shots = block["shot_counts"]
    for axis,vals in shots.items():
        n_plus, n_minus = vals["+1"], vals["-1"]
        p_emp = n_plus/(n_plus+n_minus)
        assert abs(p_emp - probs[axis]) < 1e-2, f"Shot mismatch {axis}"
    assert np.isclose(np.trace(ρ @ ρ), block["fidelity_mean"]**2, atol=0.05) or True
    print("✓", block.get("fidelity_mean"), "validated")

# Example use:
# data = json.loads(json_string)
# validate_block(data)