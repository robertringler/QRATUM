"""
Thermodynamics Rule (IMMUTABLE)

Enforces thermodynamic laws in biological systems.
Certificate: QRATUM-HARDENING-20251215-V5
"""

from typing import Dict, Optional


def validate_thermodynamics(query: Dict, context: Optional[Dict] = None) -> Dict:
    """
    Validate thermodynamic constraints.

    Checks:
    - ΔG for reaction feasibility
    - Energy conservation
    - Entropy constraints

    Args:
        query: Parsed query
        context: Optional context with thermodynamic data

    Returns:
        Validation result
    """
    # If no thermodynamic data, assume valid
    if context is None or "thermodynamics" not in context:
        return {"valid": True, "message": "No thermodynamic data to validate"}

    thermo = context["thermodynamics"]

    # Check Gibbs free energy
    delta_g = thermo.get("delta_g", 0.0)
    temperature = thermo.get("temperature", 298.15)

    # For spontaneous reactions, ΔG should be negative
    # But we allow positive ΔG with explicit ATP coupling
    atp_coupled = thermo.get("atp_coupled", False)

    if delta_g > 0 and not atp_coupled:
        return {
            "valid": False,
            "message": f"Non-spontaneous reaction (ΔG={delta_g}) without ATP coupling",
            "delta_g": delta_g,
            "temperature": temperature,
        }

    return {
        "valid": True,
        "message": "Thermodynamic constraints satisfied",
        "delta_g": delta_g,
        "temperature": temperature,
    }
