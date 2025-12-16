"""
Constraint Validator for Symbolic Reasoning

Validates biological constraints (IMMUTABLE and TERMINAL).
Certificate: QRATUM-HARDENING-20251215-V5
"""

from typing import Dict, List, Optional


class ConstraintValidator:
    """
    Validates biological constraints.

    Enforces:
    - Mass conservation
    - Thermodynamic laws
    - Pathway logic
    """

    def __init__(self):
        """Initialize constraint validator."""
        self.violation_registry = []

    def validate_all(self, query: Dict, context: Optional[Dict] = None) -> Dict:
        """
        Validate all constraints.

        Args:
            query: Parsed query
            context: Optional context data

        Returns:
            Validation results with violations
        """
        violations = []
        critical_violations = []

        # Import constraint rules
        from .rules.mass_conservation import validate_mass_conservation
        from .rules.pathway_logic import validate_pathway_logic
        from .rules.thermodynamics import validate_thermodynamics

        # Check mass conservation (CRITICAL)
        mass_result = validate_mass_conservation(query, context)
        if not mass_result["valid"]:
            violation = {
                "constraint": "mass_conservation",
                "critical": True,
                "message": mass_result["message"],
            }
            violations.append(violation)
            critical_violations.append(violation)
            self.violation_registry.append(violation)

        # Check thermodynamics (CRITICAL)
        thermo_result = validate_thermodynamics(query, context)
        if not thermo_result["valid"]:
            violation = {
                "constraint": "thermodynamics",
                "critical": True,
                "message": thermo_result["message"],
            }
            violations.append(violation)
            critical_violations.append(violation)
            self.violation_registry.append(violation)

        # Check pathway logic (NON-CRITICAL)
        pathway_result = validate_pathway_logic(query, context)
        if not pathway_result["valid"]:
            violation = {
                "constraint": "pathway_logic",
                "critical": False,
                "message": pathway_result["message"],
            }
            violations.append(violation)
            self.violation_registry.append(violation)

        return {
            "valid": len(critical_violations) == 0,
            "violations": violations,
            "critical_violations": critical_violations,
            "checks": {
                "mass_conservation": mass_result,
                "thermodynamics": thermo_result,
                "pathway_logic": pathway_result,
            },
        }

    def get_violations(self) -> List[Dict]:
        """Get all recorded violations."""
        return self.violation_registry.copy()

    def clear_violations(self) -> None:
        """Clear violation registry."""
        self.violation_registry.clear()
