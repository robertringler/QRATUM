"""
Knowledge Base for Symbolic Reasoning

Stores biological facts, rules, and constraints.
Certificate: QRATUM-HARDENING-20251215-V5
"""

from typing import Dict, List


class KnowledgeBase:
    """
    Biological knowledge base with immutable constraints.
    """
    
    def __init__(self):
        """Initialize knowledge base."""
        self.facts = []
        self.rules = []
        self.constraints = self._load_constraints()
    
    def _load_constraints(self) -> List[Dict]:
        """Load immutable biological constraints."""
        return [
            {
                "id": "mass_conservation",
                "type": "conservation_law",
                "critical": True,
                "description": "Mass must be conserved in reactions"
            },
            {
                "id": "thermodynamics",
                "type": "physical_law",
                "critical": True,
                "description": "Reactions must obey thermodynamic laws"
            },
            {
                "id": "pathway_logic",
                "type": "biological_logic",
                "critical": False,
                "description": "Pathway dependencies must be satisfied"
            }
        ]
    
    def get_constraint(self, constraint_id: str) -> Dict:
        """Get constraint by ID."""
        for constraint in self.constraints:
            if constraint["id"] == constraint_id:
                return constraint
        return {}
    
    def add_fact(self, fact: Dict) -> None:
        """Add fact to knowledge base."""
        self.facts.append(fact)
    
    def add_rule(self, rule: Dict) -> None:
        """Add inference rule."""
        self.rules.append(rule)
    
    def query(self, query_type: str) -> List[Dict]:
        """Query knowledge base."""
        if query_type == "facts":
            return self.facts.copy()
        elif query_type == "rules":
            return self.rules.copy()
        elif query_type == "constraints":
            return self.constraints.copy()
        return []
