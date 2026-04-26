"""JURIS - Legal AI Vertical Module.

Legal analysis, contract review, and compliance checking with
strict safety controls and disclaimers.

Based on OMNILEX reference implementation.
"""

import hashlib
from platform.core.base import VerticalModuleBase
from platform.core.events import EventType, ExecutionEvent
from platform.core.intent import PlatformContract
from platform.core.substrates import ComputeSubstrate
from typing import Any, Dict, FrozenSet


class JurisModule(VerticalModuleBase):
    """JURIS - Legal AI vertical module.

    Capabilities:
    - Contract analysis and review
    - Legal research and precedent matching
    - Compliance checking
    - Regulatory interpretation
    - Risk assessment

    Safety: NOT legal advice - requires attorney review.
    """

    def __init__(self, seed: int = 42):
        """Initialize JURIS module.

        Args:
            seed: Random seed for deterministic execution
        """
        super().__init__("JURIS", seed)

    def get_safety_disclaimer(self) -> str:
        """Get JURIS safety disclaimer.

        Returns:
            Safety disclaimer for legal AI
        """
        return (
            "⚖️ LEGAL DISCLAIMER: This analysis is NOT legal advice and does not "
            "create an attorney-client relationship. Results are for informational "
            "purposes only and must be reviewed by a licensed attorney before any "
            "legal action or reliance. Laws vary by jurisdiction and may have changed. "
            "Consult qualified legal counsel for specific legal matters."
        )

    def get_prohibited_uses(self) -> FrozenSet[str]:
        """Get prohibited uses for JURIS.

        Returns:
            Set of prohibited use cases
        """
        return frozenset(
            [
                "replace_attorney",
                "provide_legal_advice",
                "courtroom_representation",
                "unauthorized_practice_of_law",
                "legal_opinion_without_review",
                "circumvent_legal_requirements",
                "defraud",
                "illegal_activity",
            ]
        )

    def get_required_attestations(self, operation: str) -> FrozenSet[str]:
        """Get required attestations for JURIS operations.

        Args:
            operation: Operation being performed

        Returns:
            Set of required attestations
        """
        base_attestations = frozenset(
            [
                "not_legal_advice_acknowledged",
                "attorney_review_required",
            ]
        )

        if "contract" in operation.lower():
            return base_attestations | frozenset(["contract_jurisdiction_known"])
        elif "compliance" in operation.lower():
            return base_attestations | frozenset(["compliance_framework_specified"])
        elif "research" in operation.lower():
            return base_attestations | frozenset(["research_purpose_only"])

        return base_attestations

    def _execute_operation(
        self, contract: PlatformContract, substrate: ComputeSubstrate
    ) -> Dict[str, Any]:
        """Execute JURIS operation.

        Args:
            contract: Validated execution contract
            substrate: Selected compute substrate

        Returns:
            Operation results

        Operations:
        - analyze_contract: Analyze contract terms and identify risks
        - legal_research: Search legal precedents and statutes
        - compliance_check: Verify compliance with regulations
        - risk_assessment: Assess legal risks
        """
        operation = contract.intent.operation
        params = contract.intent.parameters

        # Emit operation start
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation=operation,
                payload={"step": "operation_dispatch", "operation": operation},
            )
        )

        if operation == "analyze_contract":
            return self._analyze_contract(params)
        elif operation == "legal_research":
            return self._legal_research(params)
        elif operation == "compliance_check":
            return self._compliance_check(params)
        elif operation == "risk_assessment":
            return self._risk_assessment(params)
        else:
            raise ValueError(f"Unknown JURIS operation: {operation}")

    def _analyze_contract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contract terms and clauses.

        Args:
            params: Contract text and analysis parameters

        Returns:
            Analysis results
        """
        contract_text = params.get("contract_text", "")
        jurisdiction = params.get("jurisdiction", "unknown")

        # Emit analysis step
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="analyze_contract",
                payload={
                    "step": "parsing",
                    "text_length": len(contract_text),
                    "jurisdiction": jurisdiction,
                },
            )
        )

        # Deterministic analysis (simplified for demonstration)
        # In production, this would use NLP and legal databases
        contract_hash = hashlib.sha256(contract_text.encode()).hexdigest()
        analysis_seed = int(contract_hash[:8], 16) + self.seed

        # Identify key clauses (deterministic based on content)
        clauses_found = []
        clause_keywords = [
            "termination",
            "liability",
            "indemnification",
            "confidentiality",
            "dispute resolution",
            "jurisdiction",
            "force majeure",
        ]

        for keyword in clause_keywords:
            if keyword.lower() in contract_text.lower():
                clauses_found.append(keyword)

        # Emit analysis completion
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="analyze_contract",
                payload={
                    "clauses_identified": len(clauses_found),
                    "contract_hash": contract_hash[:16],
                },
            )
        )

        return {
            "analysis_id": contract_hash[:16],
            "jurisdiction": jurisdiction,
            "clauses_identified": clauses_found,
            "clause_count": len(clauses_found),
            "risk_factors": self._identify_risk_factors(contract_text, clauses_found),
            "recommendations": [
                "Review all identified clauses with legal counsel",
                f"Verify compliance with {jurisdiction} law",
                "Consider additional protective clauses if missing",
            ],
        }

    def _legal_research(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct legal research.

        Args:
            params: Research query and filters

        Returns:
            Research results
        """
        query = params.get("query", "")
        jurisdiction = params.get("jurisdiction", "federal")

        # Emit research step
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="legal_research",
                payload={"step": "query_processing", "jurisdiction": jurisdiction},
            )
        )

        # Deterministic research results (simplified)
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        precedents = [
            {
                "case_id": f"CASE-{query_hash[:8]}",
                "title": f"Precedent regarding {query[:50]}",
                "year": 2020,
                "jurisdiction": jurisdiction,
                "relevance_score": 0.85,
            }
        ]

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="legal_research",
                payload={"precedents_found": len(precedents)},
            )
        )

        return {
            "query": query,
            "jurisdiction": jurisdiction,
            "precedents": precedents,
            "statutes_referenced": [],
            "research_notes": "Results are preliminary and require attorney verification",
        }

    def _compliance_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with regulations.

        Args:
            params: Compliance framework and requirements

        Returns:
            Compliance results
        """
        framework = params.get("framework", "general")
        requirements = params.get("requirements", [])

        # Emit compliance check
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="compliance_check",
                payload={"framework": framework, "requirement_count": len(requirements)},
            )
        )

        # Check each requirement (simplified)
        results = []
        for req in requirements:
            req_hash = hashlib.sha256(str(req).encode()).hexdigest()
            # Deterministic compliance status
            status = "compliant" if int(req_hash[:2], 16) % 3 != 0 else "needs_review"
            results.append({"requirement": req, "status": status})

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="compliance_check",
                payload={"checks_performed": len(results)},
            )
        )

        return {
            "framework": framework,
            "compliance_results": results,
            "overall_status": "review_required",
            "recommendations": [
                "Verify all compliance checks with legal team",
                "Document compliance measures",
                "Establish regular review schedule",
            ],
        }

    def _risk_assessment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess legal risks.

        Args:
            params: Risk assessment parameters

        Returns:
            Risk assessment results
        """
        scenario = params.get("scenario", "")

        # Emit risk assessment
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="risk_assessment",
                payload={"scenario_length": len(scenario)},
            )
        )

        # Deterministic risk scoring
        scenario_hash = hashlib.sha256(scenario.encode()).hexdigest()
        risk_score = (int(scenario_hash[:4], 16) % 100) / 100.0

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="risk_assessment",
                payload={"risk_score": risk_score},
            )
        )

        return {
            "scenario": scenario[:100],
            "risk_score": risk_score,
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            "risk_factors": self._identify_risk_factors(scenario, []),
            "mitigation_strategies": [
                "Consult with legal counsel",
                "Document all decisions",
                "Consider insurance coverage",
            ],
        }

    def _identify_risk_factors(self, text: str, clauses: list) -> list:
        """Identify potential risk factors.

        Args:
            text: Text to analyze
            clauses: Identified clauses

        Returns:
            List of risk factors
        """
        risk_keywords = [
            "unlimited liability",
            "no limitation",
            "sole discretion",
            "without notice",
            "waive",
            "forfeit",
        ]

        risks = []
        for keyword in risk_keywords:
            if keyword in text.lower():
                risks.append(f"Contains '{keyword}' - requires review")

        if not clauses:
            risks.append("Missing standard protective clauses")

        return risks
