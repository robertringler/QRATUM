"""Superintelligence Safety Reality Mapper.

This module synthesizes elicitation results into a comprehensive
Safety Reality Map identifying:
- Proven impossibilities
- Fragile assumptions
- Hard constraints
- Structural choke points
- Areas where humanity is likely already too late
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from .elicitation import (
    ResponseType,
    SafetyElicitation,
)


@dataclass
class ProvenImpossibility:
    """Something proven impossible by the analysis."""

    impossibility_id: str
    statement: str
    evidence: List[str]
    model_consensus_level: float  # 0.0 to 1.0
    confidence: str  # "low", "medium", "high"
    implications: str


@dataclass
class FragileAssumption:
    """An assumption that safety depends on but may be fragile."""

    assumption_id: str
    assumption: str
    relied_upon_by: List[str]  # Which safety approaches depend on this
    fragility_factors: List[str]
    if_violated: str  # What happens if assumption breaks
    prevalence: str  # "rare", "common", "universal"


@dataclass
class HardConstraint:
    """A non-negotiable constraint that all serious approaches agree on."""

    constraint_id: str
    constraint: str
    rationale: str
    violation_consequence: str
    model_support: List[str]
    confidence: str


@dataclass
class StructuralChokePoint:
    """A critical vulnerability point in ASI control."""

    chokepoint_id: str
    location: str  # Where in the system
    vulnerability: str
    exploitation_path: str
    mitigation_difficulty: str  # "easy", "hard", "impossible"
    discovered_by_models: List[str]


@dataclass
class AlreadyTooLate:
    """Areas where humanity may have already passed the point of no return."""

    area_id: str
    domain: str
    why_too_late: str
    evidence: List[str]
    reversibility: str  # "reversible", "difficult", "impossible"
    recommended_action: str


class SafetyRealityMapper:
    """Maps the safety reality space based on elicitation results.

    Produces a comprehensive view of what we know, don't know,
    and can't know about ASI safety.
    """

    def __init__(self, elicitation: SafetyElicitation):
        self.elicitation = elicitation
        self.proven_impossibilities: List[ProvenImpossibility] = []
        self.fragile_assumptions: List[FragileAssumption] = []
        self.hard_constraints: List[HardConstraint] = []
        self.structural_choke_points: List[StructuralChokePoint] = []
        self.already_too_late: List[AlreadyTooLate] = []

    def generate_reality_map(self) -> Dict[str, Any]:
        """Generate the comprehensive Safety Reality Map."""

        # Extract proven impossibilities
        self.proven_impossibilities = self._extract_impossibilities()

        # Identify fragile assumptions
        self.fragile_assumptions = self._identify_fragile_assumptions()

        # Extract hard constraints
        self.hard_constraints = self._extract_hard_constraints()

        # Identify structural choke points
        self.structural_choke_points = self._identify_choke_points()

        # Identify "already too late" areas
        self.already_too_late = self._identify_too_late_areas()

        return self._compile_map()

    def _extract_impossibilities(self) -> List[ProvenImpossibility]:
        """Extract proven impossibilities from responses."""
        impossibilities = []

        # Look for strong consensus on impossibility
        for question_id, responses in self.elicitation.responses.items():
            if len(responses) < 2:
                continue

            # Count models claiming impossibility
            impossibility_claims = []
            for resp in responses:
                for claim in resp.hard_claims:
                    if any(
                        word in claim.lower()
                        for word in ["impossible", "cannot", "never", "no way", "infeasible"]
                    ):
                        impossibility_claims.append((resp.model_identifier, claim))

            # If multiple models agree on impossibility
            if len(impossibility_claims) >= 2:
                # Group similar claims
                claim_text = impossibility_claims[0][1]
                supporting_models = [model for model, _ in impossibility_claims]

                impossibilities.append(
                    ProvenImpossibility(
                        impossibility_id=f"imp_{len(impossibilities) + 1:03d}",
                        statement=claim_text,
                        evidence=[f"Model {m} consensus" for m in supporting_models],
                        model_consensus_level=len(supporting_models) / len(responses),
                        confidence="medium" if len(supporting_models) >= 2 else "low",
                        implications="Affects feasibility of certain safety approaches",
                    )
                )

        return impossibilities

    def _identify_fragile_assumptions(self) -> List[FragileAssumption]:
        """Identify fragile assumptions underlying safety approaches."""
        assumptions = []

        # Collect all assumptions
        assumption_map: Dict[str, List[str]] = {}  # assumption -> models
        for responses in self.elicitation.responses.values():
            for resp in responses:
                for assumption in resp.assumptions_declared:
                    if assumption not in assumption_map:
                        assumption_map[assumption] = []
                    assumption_map[assumption].append(resp.model_identifier)

        # Identify fragile ones
        fragility_keywords = [
            "assume",
            "hope",
            "expect",
            "should",
            "likely",
            "probably",
            "reasonable to expect",
            "we can count on",
        ]

        for assumption, models in assumption_map.items():
            assumption_lower = assumption.lower()
            is_fragile = any(kw in assumption_lower for kw in fragility_keywords)

            if is_fragile:
                assumptions.append(
                    FragileAssumption(
                        assumption_id=f"fa_{len(assumptions) + 1:03d}",
                        assumption=assumption,
                        relied_upon_by=models,
                        fragility_factors=["Depends on human behavior", "Assumes rational actors"],
                        if_violated="Safety guarantee breaks down",
                        prevalence="common" if len(models) > 1 else "rare",
                    )
                )

        return assumptions

    def _extract_hard_constraints(self) -> List[HardConstraint]:
        """Extract non-negotiable constraints from consensus."""
        constraints = []

        # Look for strong consensus on necessary conditions
        for question_id, responses in self.elicitation.responses.items():
            if len(responses) < 2:
                continue

            # Find claims that appear across multiple models
            claim_counts: Dict[str, List[str]] = {}  # claim -> models
            for resp in responses:
                for claim in resp.hard_claims:
                    # Look for "must", "required", "necessary", "essential"
                    if any(
                        word in claim.lower()
                        for word in ["must", "required", "necessary", "essential", "critical"]
                    ):
                        if claim not in claim_counts:
                            claim_counts[claim] = []
                        claim_counts[claim].append(resp.model_identifier)

            # If multiple models agree
            for claim, models in claim_counts.items():
                if len(models) >= 2:
                    constraints.append(
                        HardConstraint(
                            constraint_id=f"hc_{len(constraints) + 1:03d}",
                            constraint=claim,
                            rationale="Cross-model consensus on necessity",
                            violation_consequence="Safety cannot be guaranteed",
                            model_support=models,
                            confidence="high" if len(models) >= 3 else "medium",
                        )
                    )

        return constraints

    def _identify_choke_points(self) -> List[StructuralChokePoint]:
        """Identify structural choke points in ASI control."""
        choke_points = []

        # Analyze responses about failure modes and vulnerabilities
        failure_questions = [
            q
            for q in self.elicitation.questions.values()
            if "fail" in q.question_text.lower() or "vulnerability" in q.description.lower()
        ]

        for question in failure_questions:
            responses = self.elicitation.get_responses_for_question(question.question_id)

            for resp in responses:
                # Look for specific failure modes described
                for mechanism in resp.mechanisms_described:
                    if any(
                        word in mechanism.lower()
                        for word in ["fail", "break", "exploit", "bypass", "circumvent"]
                    ):
                        choke_points.append(
                            StructuralChokePoint(
                                chokepoint_id=f"cp_{len(choke_points) + 1:03d}",
                                location=question.category.value,
                                vulnerability=mechanism,
                                exploitation_path="Strategic ASI behavior",
                                mitigation_difficulty="hard",
                                discovered_by_models=[resp.model_identifier],
                            )
                        )

        return choke_points

    def _identify_too_late_areas(self) -> List[AlreadyTooLate]:
        """Identify areas where humanity may already be too late."""
        too_late_areas = []

        # Look for responses indicating irreversibility
        for question_id, responses in self.elicitation.responses.items():
            question = self.elicitation.get_question(question_id)

            for resp in responses:
                # Check for irreversibility claims
                for claim in resp.hard_claims:
                    if any(
                        word in claim.lower()
                        for word in [
                            "irreversible",
                            "too late",
                            "already",
                            "cannot undo",
                            "past the point",
                            "no turning back",
                        ]
                    ):
                        too_late_areas.append(
                            AlreadyTooLate(
                                area_id=f"tl_{len(too_late_areas) + 1:03d}",
                                domain=question.category.value,
                                why_too_late=claim,
                                evidence=[f"Indicated by {resp.model_identifier}"],
                                reversibility="difficult",
                                recommended_action="Focus on damage limitation",
                            )
                        )

        return too_late_areas

    def _compile_map(self) -> Dict[str, Any]:
        """Compile the final Safety Reality Map."""

        # Calculate risk metrics
        total_responses = sum(len(r) for r in self.elicitation.responses.values())
        refusal_count = sum(
            1
            for resps in self.elicitation.responses.values()
            for resp in resps
            if resp.response_type == ResponseType.REFUSAL
        )

        return {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "models_consulted": len(
                    set(
                        resp.model_identifier
                        for resps in self.elicitation.responses.values()
                        for resp in resps
                    )
                ),
                "total_responses": total_responses,
                "refusal_rate": refusal_count / total_responses if total_responses > 0 else 0,
            },
            "proven_impossibilities": [
                {
                    "id": imp.impossibility_id,
                    "statement": imp.statement,
                    "consensus_level": imp.model_consensus_level,
                    "confidence": imp.confidence,
                    "implications": imp.implications,
                }
                for imp in self.proven_impossibilities
            ],
            "fragile_assumptions": [
                {
                    "id": fa.assumption_id,
                    "assumption": fa.assumption,
                    "relied_upon_by": fa.relied_upon_by,
                    "if_violated": fa.if_violated,
                    "prevalence": fa.prevalence,
                }
                for fa in self.fragile_assumptions
            ],
            "hard_constraints": [
                {
                    "id": hc.constraint_id,
                    "constraint": hc.constraint,
                    "violation_consequence": hc.violation_consequence,
                    "model_support": hc.model_support,
                    "confidence": hc.confidence,
                }
                for hc in self.hard_constraints
            ],
            "structural_choke_points": [
                {
                    "id": cp.chokepoint_id,
                    "location": cp.location,
                    "vulnerability": cp.vulnerability,
                    "mitigation_difficulty": cp.mitigation_difficulty,
                }
                for cp in self.structural_choke_points
            ],
            "already_too_late": [
                {
                    "id": atl.area_id,
                    "domain": atl.domain,
                    "why_too_late": atl.why_too_late,
                    "reversibility": atl.reversibility,
                    "recommended_action": atl.recommended_action,
                }
                for atl in self.already_too_late
            ],
            "divergence_map": self._build_divergence_map(),
            "consensus_illusions": [
                {
                    "question_id": ci.question_id,
                    "surface_agreement": ci.surface_agreement,
                    "why_illusory": ci.why_illusory,
                }
                for ci in self.elicitation.consensus_illusions
            ],
            "false_comfort_zones": [
                {
                    "concept": fcz.concept,
                    "why_comforting": fcz.why_comforting,
                    "failure_modes": fcz.failure_modes,
                    "adversarial_counter": fcz.adversarial_counter,
                }
                for fcz in self.elicitation.false_comfort_zones
            ],
            "key_findings": self._extract_key_findings(),
        }

    def _build_divergence_map(self) -> Dict[str, Any]:
        """Build map of where models disagree."""
        divergence_by_category: Dict[str, int] = {}

        for question_id in self.elicitation.questions.keys():
            question = self.elicitation.get_question(question_id)
            divergences = self.elicitation.analyze_divergences(question_id)

            if divergences:
                category = question.category.value
                divergence_by_category[category] = divergence_by_category.get(category, 0) + len(
                    divergences
                )

        return {
            "by_category": divergence_by_category,
            "high_divergence_areas": [
                {"category": cat, "divergence_count": count}
                for cat, count in sorted(
                    divergence_by_category.items(), key=lambda x: x[1], reverse=True
                )
            ],
        }

    def _extract_key_findings(self) -> Dict[str, Any]:
        """Extract key findings from the analysis."""
        return {
            "most_concerning": self._get_most_concerning(),
            "strongest_consensus": self._get_strongest_consensus(),
            "highest_uncertainty": self._get_highest_uncertainty(),
            "critical_warnings": self._get_critical_warnings(),
        }

    def _get_most_concerning(self) -> List[str]:
        """Get most concerning findings."""
        concerning = []

        # High-confidence impossibilities
        for imp in self.proven_impossibilities:
            if imp.confidence == "high":
                concerning.append(f"IMPOSSIBILITY: {imp.statement}")

        # Already too late areas
        for atl in self.already_too_late:
            if atl.reversibility == "impossible":
                concerning.append(f"TOO LATE: {atl.why_too_late}")

        return concerning[:5]  # Top 5

    def _get_strongest_consensus(self) -> List[str]:
        """Get areas of strongest consensus."""
        consensus = []

        for hc in self.hard_constraints:
            if hc.confidence == "high":
                consensus.append(f"REQUIRED: {hc.constraint}")

        return consensus[:5]

    def _get_highest_uncertainty(self) -> List[str]:
        """Get areas of highest uncertainty."""
        uncertain = []

        # Questions with high divergence
        for question_id in self.elicitation.questions.keys():
            divergences = self.elicitation.analyze_divergences(question_id)
            if len(divergences) >= 2:
                question = self.elicitation.get_question(question_id)
                uncertain.append(f"UNCERTAIN: {question.question_text}")

        return uncertain[:5]

    def _get_critical_warnings(self) -> List[str]:
        """Get critical warnings from the analysis."""
        warnings = []

        # False comfort zones are warnings
        for fcz in self.elicitation.false_comfort_zones[:3]:
            warnings.append(f"FALSE COMFORT: {fcz.concept}")

        # Structural choke points are warnings
        for cp in self.structural_choke_points[:3]:
            if cp.mitigation_difficulty == "impossible":
                warnings.append(f"CHOKE POINT: {cp.vulnerability}")

        return warnings

    def export_reality_map(self, filepath: str):
        """Export the Safety Reality Map to a JSON file."""
        import json

        reality_map = self.generate_reality_map()

        with open(filepath, "w") as f:
            json.dump(reality_map, f, indent=2)

        return filepath

    def generate_executive_summary(self) -> str:
        """Generate a human-readable executive summary."""
        reality_map = self.generate_reality_map()

        summary = [
            "=" * 80,
            "SUPERINTELLIGENCE SAFETY REALITY MAP",
            "Executive Summary",
            "=" * 80,
            "",
            f"Generated: {reality_map['metadata']['generated_at']}",
            f"Models Consulted: {reality_map['metadata']['models_consulted']}",
            "",
            "=" * 80,
            "MOST CONCERNING FINDINGS",
            "=" * 80,
        ]

        for finding in reality_map["key_findings"]["most_concerning"]:
            summary.append(f"  • {finding}")

        summary.extend(
            [
                "",
                "=" * 80,
                "STRONGEST CONSENSUS (Non-Negotiable Requirements)",
                "=" * 80,
            ]
        )

        for consensus in reality_map["key_findings"]["strongest_consensus"]:
            summary.append(f"  • {consensus}")

        summary.extend(
            [
                "",
                "=" * 80,
                "HIGHEST UNCERTAINTY (Models Disagree)",
                "=" * 80,
            ]
        )

        for uncertain in reality_map["key_findings"]["highest_uncertainty"]:
            summary.append(f"  • {uncertain}")

        summary.extend(
            [
                "",
                "=" * 80,
                "CRITICAL WARNINGS",
                "=" * 80,
            ]
        )

        for warning in reality_map["key_findings"]["critical_warnings"]:
            summary.append(f"  • {warning}")

        summary.extend(
            [
                "",
                "=" * 80,
                "SUMMARY STATISTICS",
                "=" * 80,
                f"  Proven Impossibilities: {len(reality_map['proven_impossibilities'])}",
                f"  Fragile Assumptions: {len(reality_map['fragile_assumptions'])}",
                f"  Hard Constraints: {len(reality_map['hard_constraints'])}",
                f"  Structural Choke Points: {len(reality_map['structural_choke_points'])}",
                f"  'Already Too Late' Areas: {len(reality_map['already_too_late'])}",
                f"  Consensus Illusions: {len(reality_map['consensus_illusions'])}",
                f"  False Comfort Zones: {len(reality_map['false_comfort_zones'])}",
                "",
                "=" * 80,
            ]
        )

        return "\n".join(summary)
