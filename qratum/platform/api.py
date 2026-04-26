"""
QRATUM Platform REST API

Provides HTTP endpoints for all 14 verticals and cross-domain synthesis.
Built for sovereign, on-premises deployment.

Version: 1.0.0
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from qradle import DeterministicEngine, ExecutionContext
from qratum.platform.reasoning_engine import ReasoningStrategy, UnifiedReasoningEngine


# API Models
@dataclass
class APIRequest:
    """Base API request model."""

    vertical: str
    task: str
    parameters: Dict[str, Any]
    requester_id: str = "api_client"
    safety_level: str = "ROUTINE"
    authorized: bool = False


@dataclass
class APIResponse:
    """Base API response model."""

    success: bool
    data: Any
    execution_time: float
    output_hash: str
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None
    safety_disclaimer: Optional[str] = None


@dataclass
class SynthesisRequest:
    """Multi-vertical synthesis request."""

    query: str
    verticals: List[str]
    parameters: Optional[Dict[str, Any]] = None
    strategy: str = "deductive"
    requester_id: str = "api_client"


@dataclass
class SynthesisResponse:
    """Multi-vertical synthesis response."""

    success: bool
    chain_id: str
    query: str
    verticals_used: List[str]
    final_conclusion: Dict[str, Any]
    confidence: float
    provenance_hash: str
    reasoning_nodes_count: int
    execution_time: float


class QRATUMAPIService:
    """
    REST API service for QRATUM platform.

    Provides endpoints for:
    - All 14 vertical modules
    - Cross-domain synthesis
    - Reasoning chain verification
    - System health and statistics
    """

    # Vertical configurations
    VERTICALS = {
        "JURIS": {
            "name": "Legal & Compliance",
            "tasks": [
                "analyze_contract",
                "legal_reasoning",
                "predict_litigation",
                "check_compliance",
            ],
            "safety_disclaimer": "⚖️ LEGAL DISCLAIMER: This analysis is for informational purposes only.",
        },
        "VITRA": {
            "name": "Healthcare & Life Sciences",
            "tasks": [
                "diagnose_condition",
                "drug_interaction",
                "clinical_decision",
                "research_analysis",
            ],
            "safety_disclaimer": "🏥 MEDICAL DISCLAIMER: Not a substitute for professional medical advice.",
        },
        "ECORA": {
            "name": "Climate & Environment",
            "tasks": [
                "climate_model",
                "sustainability_analysis",
                "resource_optimization",
                "impact_assessment",
            ],
            "safety_disclaimer": "🌍 ENVIRONMENTAL ANALYSIS: Based on current scientific models.",
        },
        "CAPRA": {
            "name": "Finance & Economics",
            "tasks": [
                "risk_assessment",
                "fraud_detection",
                "market_analysis",
                "portfolio_optimization",
            ],
            "safety_disclaimer": "💰 FINANCIAL DISCLAIMER: Not financial advice. Consult professionals.",
        },
        "SENTRA": {
            "name": "Security & Defense",
            "tasks": [
                "threat_detection",
                "vulnerability_analysis",
                "strategic_planning",
                "risk_modeling",
            ],
            "safety_disclaimer": "🔒 SECURITY ANALYSIS: For authorized use only.",
        },
        "NEURA": {
            "name": "Cognitive Science & Psychology",
            "tasks": [
                "behavioral_analysis",
                "cognitive_modeling",
                "mental_health_support",
                "human_factors",
            ],
            "safety_disclaimer": "🧠 PSYCHOLOGICAL ANALYSIS: Not a substitute for mental health care.",
        },
        "FLUXA": {
            "name": "Supply Chain & Logistics",
            "tasks": [
                "optimize_route",
                "demand_forecast",
                "inventory_management",
                "supplier_analysis",
            ],
            "safety_disclaimer": "📦 LOGISTICS ANALYSIS: Operational recommendations only.",
        },
        "CHRONA": {
            "name": "Temporal Reasoning & Forecasting",
            "tasks": [
                "time_series_analysis",
                "predictive_modeling",
                "scenario_planning",
                "trend_detection",
            ],
            "safety_disclaimer": "⏰ TEMPORAL ANALYSIS: Predictions based on historical patterns.",
        },
        "GEONA": {
            "name": "Geospatial & Navigation",
            "tasks": [
                "spatial_analysis",
                "route_optimization",
                "terrain_modeling",
                "location_intelligence",
            ],
            "safety_disclaimer": "🗺️ GEOSPATIAL ANALYSIS: Verify critical navigation decisions.",
        },
        "FUSIA": {
            "name": "Energy & Materials",
            "tasks": [
                "grid_optimization",
                "materials_discovery",
                "energy_modeling",
                "fusion_research",
            ],
            "safety_disclaimer": "⚡ ENERGY ANALYSIS: Technical assessment only.",
        },
        "STRATA": {
            "name": "Social Systems & Policy",
            "tasks": [
                "policy_analysis",
                "social_impact",
                "governance_modeling",
                "equity_assessment",
            ],
            "safety_disclaimer": "🏛️ POLICY ANALYSIS: Informational analysis only.",
        },
        "VEXOR": {
            "name": "Adversarial & Game Theory",
            "tasks": [
                "strategic_game",
                "adversarial_reasoning",
                "negotiation_analysis",
                "game_theory_modeling",
            ],
            "safety_disclaimer": "♟️ STRATEGIC ANALYSIS: Theoretical modeling only.",
        },
        "COHORA": {
            "name": "Collaborative Intelligence",
            "tasks": [
                "multi_agent_coordination",
                "collective_decision",
                "swarm_optimization",
                "consensus_building",
            ],
            "safety_disclaimer": "🤝 COLLABORATIVE ANALYSIS: Coordination recommendations.",
        },
        "ORBIA": {
            "name": "Orbital & Space Systems",
            "tasks": [
                "orbital_mechanics",
                "satellite_ops",
                "space_mission_planning",
                "trajectory_optimization",
            ],
            "safety_disclaimer": "🚀 SPACE ANALYSIS: Mission-critical decisions require validation.",
        },
    }

    def __init__(self):
        """Initialize API service."""
        self.qradle_engine = DeterministicEngine()
        self.reasoning_engine = UnifiedReasoningEngine()

    def execute_vertical_task(self, request: APIRequest) -> APIResponse:
        """Execute a task on a specific vertical.

        Args:
            request: API request with vertical, task, and parameters

        Returns:
            APIResponse with results
        """
        start_time = datetime.now(timezone.utc)

        # Validate vertical
        if request.vertical not in self.VERTICALS:
            return APIResponse(
                success=False,
                data=None,
                execution_time=0.0,
                output_hash="",
                error=f"Unknown vertical: {request.vertical}",
            )

        vertical_config = self.VERTICALS[request.vertical]

        # Validate task
        if request.task not in vertical_config["tasks"]:
            return APIResponse(
                success=False,
                data=None,
                execution_time=0.0,
                output_hash="",
                error=f"Unknown task '{request.task}' for vertical '{request.vertical}'",
            )

        # Create execution context
        context = ExecutionContext(
            contract_id=f"{request.vertical}_{request.task}_{int(start_time.timestamp())}",
            parameters={"vertical": request.vertical, "task": request.task, **request.parameters},
            timestamp=start_time.isoformat(),
            safety_level=request.safety_level,
            authorized=request.authorized,
        )

        # Execute with QRADLE
        def task_executor(params):
            # Simulate vertical task execution
            # In production, this would call the actual vertical module
            return {
                "vertical": params["vertical"],
                "task": params["task"],
                "result": f"Executed {params['task']} on {params['vertical']}",
                "parameters": request.parameters,
                "insights": [
                    f"Insight 1 from {params['vertical']}",
                    f"Insight 2 from {params['vertical']}",
                ],
            }

        result = self.qradle_engine.execute_contract(context, task_executor)

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return APIResponse(
            success=result.success,
            data=result.output if result.success else None,
            execution_time=execution_time,
            output_hash=result.output_hash,
            checkpoint_id=result.checkpoint_id,
            error=result.error,
            safety_disclaimer=vertical_config["safety_disclaimer"],
        )

    def execute_synthesis(self, request: SynthesisRequest) -> SynthesisResponse:
        """Execute cross-domain synthesis across multiple verticals.

        Args:
            request: Synthesis request with query and verticals

        Returns:
            SynthesisResponse with synthesized results
        """
        start_time = datetime.now(timezone.utc)

        # Validate verticals
        invalid_verticals = [v for v in request.verticals if v not in self.VERTICALS]
        if invalid_verticals:
            return SynthesisResponse(
                success=False,
                chain_id="",
                query=request.query,
                verticals_used=[],
                final_conclusion={"error": f"Invalid verticals: {invalid_verticals}"},
                confidence=0.0,
                provenance_hash="",
                reasoning_nodes_count=0,
                execution_time=0.0,
            )

        # Parse reasoning strategy
        try:
            strategy = ReasoningStrategy(request.strategy.lower())
        except ValueError:
            strategy = ReasoningStrategy.DEDUCTIVE

        # Execute synthesis
        chain = self.reasoning_engine.synthesize(
            query=request.query,
            verticals=request.verticals,
            parameters=request.parameters,
            strategy=strategy,
        )

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return SynthesisResponse(
            success=True,
            chain_id=chain.chain_id,
            query=chain.query,
            verticals_used=chain.verticals_used,
            final_conclusion=chain.final_conclusion,
            confidence=chain.confidence,
            provenance_hash=chain.provenance_hash,
            reasoning_nodes_count=len(chain.nodes),
            execution_time=execution_time,
        )

    def get_vertical_info(self, vertical: str) -> Optional[Dict[str, Any]]:
        """Get information about a vertical module."""
        return self.VERTICALS.get(vertical)

    def list_verticals(self) -> List[Dict[str, Any]]:
        """List all available vertical modules."""
        return [{"vertical_id": v_id, **v_info} for v_id, v_info in self.VERTICALS.items()]

    def verify_reasoning_chain(self, chain_id: str) -> Dict[str, Any]:
        """Verify integrity of a reasoning chain."""
        is_valid = self.reasoning_engine.verify_reasoning_chain(chain_id)
        chain = self.reasoning_engine.get_reasoning_chain(chain_id)

        return {
            "chain_id": chain_id,
            "exists": chain is not None,
            "verified": is_valid,
            "provenance_hash": chain.provenance_hash if chain else None,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        qradle_stats = self.qradle_engine.get_stats()
        reasoning_stats = self.reasoning_engine.get_stats()

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verticals_available": len(self.VERTICALS),
            "qradle_executions": qradle_stats["total_executions"],
            "reasoning_chains": reasoning_stats["total_chains"],
            "merkle_chain_valid": qradle_stats["chain_valid"],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed system statistics."""
        return {
            "verticals": len(self.VERTICALS),
            "qradle": self.qradle_engine.get_stats(),
            "reasoning_engine": self.reasoning_engine.get_stats(),
        }


# Example API endpoint implementations (FastAPI/Flask would use these)


def create_api_routes():
    """
    Example of how to create API routes.

    In production, integrate with FastAPI, Flask, or other framework.
    """
    service = QRATUMAPIService()

    # Example routes:
    # POST /api/v1/vertical/execute
    # POST /api/v1/synthesis/execute
    # GET /api/v1/verticals
    # GET /api/v1/vertical/{vertical_id}
    # GET /api/v1/synthesis/verify/{chain_id}
    # GET /api/v1/health
    # GET /api/v1/stats

    return service
