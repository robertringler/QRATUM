"""
Mass Conservation Rule (IMMUTABLE)

Enforces mass conservation in biological reactions.
Certificate: QRATUM-HARDENING-20251215-V5
"""

from typing import Dict, Optional


def validate_mass_conservation(query: Dict, context: Optional[Dict] = None) -> Dict:
    """
    Validate mass conservation.
    
    For biological reactions: sum(reactant_masses) = sum(product_masses)
    
    Args:
        query: Parsed query
        context: Optional context with reaction data
        
    Returns:
        Validation result
    """
    # If no reaction data, assume valid
    if context is None or "reaction" not in context:
        return {
            "valid": True,
            "message": "No reaction data to validate"
        }
    
    reaction = context["reaction"]
    
    # Extract masses
    reactant_mass = reaction.get("reactant_mass", 0.0)
    product_mass = reaction.get("product_mass", 0.0)
    
    # Check conservation (with small tolerance for numerical precision)
    tolerance = 1e-6
    mass_diff = abs(reactant_mass - product_mass)
    
    if mass_diff > tolerance:
        return {
            "valid": False,
            "message": f"Mass not conserved: reactants={reactant_mass}, products={product_mass}, diff={mass_diff}",
            "reactant_mass": reactant_mass,
            "product_mass": product_mass,
            "difference": mass_diff
        }
    
    return {
        "valid": True,
        "message": "Mass conservation satisfied",
        "reactant_mass": reactant_mass,
        "product_mass": product_mass
    }
