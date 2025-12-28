"""
Continuous Optimization Module

Real-time monitoring, predictive simulation, emergent risk analysis,
and autonomous adjustment of nodes, contracts, policies, and token flows.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OptimizationType(Enum):
    """Types of optimization."""

    INFRASTRUCTURE = "infrastructure"
    ECONOMICS = "economics"
    GOVERNANCE = "governance"
    SECURITY = "security"
    PERFORMANCE = "performance"


class RiskLevel(Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot.

    Attributes:
        metric_id: Unique metric identifier
        name: Metric name
        value: Current value
        timestamp: Snapshot timestamp
        source: Data source
        metadata: Additional metadata
    """

    metric_id: str
    name: str
    value: float
    timestamp: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class OptimizationAction:
    """Recommended optimization action.

    Attributes:
        action_id: Unique action identifier
        action_type: Type of optimization
        description: Action description
        priority: Priority score (0-1)
        impact_estimate: Estimated impact percentage
        target_component: Target component for action
        parameters: Action parameters
        status: Action status
    """

    action_id: str
    action_type: OptimizationType
    description: str
    priority: float = 0.5
    impact_estimate: float = 0.0
    target_component: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def execute(self) -> dict[str, Any]:
        """Execute the optimization action.

        Returns:
            Execution result
        """
        self.status = "executed"
        return {
            "action_id": self.action_id,
            "status": self.status,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "impact_estimate": self.impact_estimate,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize action to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "priority": self.priority,
            "impact_estimate": self.impact_estimate,
            "target_component": self.target_component,
            "parameters": self.parameters,
            "status": self.status,
        }


@dataclass
class RiskAssessment:
    """Risk assessment result.

    Attributes:
        assessment_id: Unique assessment identifier
        risk_type: Type of risk
        level: Risk level
        probability: Probability (0-1)
        impact: Potential impact (0-1)
        description: Risk description
        mitigation_actions: Suggested mitigations
        timestamp: Assessment timestamp
    """

    assessment_id: str
    risk_type: str
    level: RiskLevel
    probability: float
    impact: float
    description: str
    mitigation_actions: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def get_risk_score(self) -> float:
        """Calculate risk score.

        Returns:
            Risk score (probability * impact)
        """
        return self.probability * self.impact

    def to_dict(self) -> dict[str, Any]:
        """Serialize assessment to dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "risk_type": self.risk_type,
            "level": self.level.value,
            "probability": self.probability,
            "impact": self.impact,
            "risk_score": self.get_risk_score(),
            "description": self.description,
            "mitigation_actions": self.mitigation_actions,
            "timestamp": self.timestamp,
        }


class RiskAnalyzer:
    """Emergent risk analysis system.

    Analyzes system state for emerging risks and provides
    risk assessments with mitigation recommendations.

    Attributes:
        analyzer_id: Unique analyzer identifier
        risk_thresholds: Thresholds for risk levels
        assessments: List of risk assessments
        monitoring_enabled: Whether monitoring is active
    """

    def __init__(self, analyzer_id: str | None = None) -> None:
        """Initialize risk analyzer.

        Args:
            analyzer_id: Optional analyzer ID
        """
        self.analyzer_id = analyzer_id or f"risk_{secrets.token_hex(6)}"
        self.risk_thresholds = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9,
        }
        self.assessments: list[RiskAssessment] = []
        self.monitoring_enabled = True

    def assess_infrastructure_risk(self, metrics: dict[str, float]) -> RiskAssessment:
        """Assess infrastructure-related risks.

        Args:
            metrics: Current infrastructure metrics

        Returns:
            Risk assessment
        """
        node_uptime = metrics.get("node_uptime_percent", 100)
        capacity_util = metrics.get("capacity_utilization", 0)
        error_rate = metrics.get("error_rate_percent", 0)

        # Calculate risk factors
        uptime_risk = max(0, (99 - node_uptime) / 10)
        capacity_risk = max(0, (capacity_util - 80) / 20) if capacity_util > 80 else 0
        error_risk = error_rate / 5  # 5% error rate = 1.0 risk

        probability = min(1.0, (uptime_risk + capacity_risk + error_risk) / 3)
        impact = 0.8 if capacity_util > 90 else 0.5

        level = self._determine_level(probability * impact)

        mitigations = []
        if uptime_risk > 0.5:
            mitigations.append("Add redundant nodes in affected regions")
        if capacity_risk > 0.5:
            mitigations.append("Scale up node capacity or add new nodes")
        if error_risk > 0.5:
            mitigations.append("Investigate error sources and deploy fixes")

        assessment = RiskAssessment(
            assessment_id=f"infra_risk_{len(self.assessments):04d}",
            risk_type="infrastructure",
            level=level,
            probability=probability,
            impact=impact,
            description=f"Infrastructure risk: uptime={node_uptime}%, capacity={capacity_util}%, errors={error_rate}%",
            mitigation_actions=mitigations,
        )
        self.assessments.append(assessment)
        return assessment

    def assess_economic_risk(self, metrics: dict[str, float]) -> RiskAssessment:
        """Assess economic-related risks.

        Args:
            metrics: Current economic metrics

        Returns:
            Risk assessment
        """
        token_volatility = metrics.get("token_volatility", 0)
        liquidity_ratio = metrics.get("liquidity_ratio", 1)
        revenue_growth = metrics.get("revenue_growth_percent", 0)

        # Calculate risk factors
        volatility_risk = min(1.0, token_volatility / 0.5)  # 50% volatility = max risk
        liquidity_risk = max(0, (0.3 - liquidity_ratio) / 0.3) if liquidity_ratio < 0.3 else 0
        growth_risk = max(0, -revenue_growth / 20) if revenue_growth < 0 else 0

        probability = min(1.0, (volatility_risk + liquidity_risk + growth_risk) / 3)
        impact = 0.7

        level = self._determine_level(probability * impact)

        mitigations = []
        if volatility_risk > 0.5:
            mitigations.append("Implement token stabilization mechanisms")
        if liquidity_risk > 0.5:
            mitigations.append("Increase liquidity incentives")
        if growth_risk > 0.5:
            mitigations.append("Review pricing and market strategy")

        assessment = RiskAssessment(
            assessment_id=f"econ_risk_{len(self.assessments):04d}",
            risk_type="economic",
            level=level,
            probability=probability,
            impact=impact,
            description=f"Economic risk: volatility={token_volatility:.2%}, liquidity={liquidity_ratio:.2f}",
            mitigation_actions=mitigations,
        )
        self.assessments.append(assessment)
        return assessment

    def assess_security_risk(self, metrics: dict[str, float]) -> RiskAssessment:
        """Assess security-related risks.

        Args:
            metrics: Current security metrics

        Returns:
            Risk assessment
        """
        threat_level = metrics.get("threat_level", 0)
        anomaly_score = metrics.get("anomaly_score", 0)
        incident_count = metrics.get("recent_incidents", 0)

        probability = min(1.0, (threat_level + anomaly_score) / 2)
        impact = 0.9  # Security risks have high impact

        level = self._determine_level(probability * impact)

        mitigations = []
        if threat_level > 0.5:
            mitigations.append("Enable enhanced threat monitoring")
        if anomaly_score > 0.5:
            mitigations.append("Investigate anomalous behavior patterns")
        if incident_count > 0:
            mitigations.append("Review and patch recent incident vectors")

        assessment = RiskAssessment(
            assessment_id=f"sec_risk_{len(self.assessments):04d}",
            risk_type="security",
            level=level,
            probability=probability,
            impact=impact,
            description=f"Security risk: threat={threat_level:.2f}, anomalies={anomaly_score:.2f}",
            mitigation_actions=mitigations,
        )
        self.assessments.append(assessment)
        return assessment

    def _determine_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score.

        Args:
            risk_score: Calculated risk score

        Returns:
            Risk level
        """
        if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def get_summary(self) -> dict[str, Any]:
        """Get risk analysis summary.

        Returns:
            Summary of all assessments
        """
        by_level: dict[str, int] = {level.value: 0 for level in RiskLevel}
        for assessment in self.assessments:
            by_level[assessment.level.value] += 1

        avg_score = (
            sum(a.get_risk_score() for a in self.assessments) / len(self.assessments)
            if self.assessments
            else 0
        )

        return {
            "analyzer_id": self.analyzer_id,
            "total_assessments": len(self.assessments),
            "by_level": by_level,
            "average_risk_score": avg_score,
            "monitoring_enabled": self.monitoring_enabled,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize analyzer to dictionary."""
        return {
            "analyzer_id": self.analyzer_id,
            "assessments": [a.to_dict() for a in self.assessments[-20:]],
            "summary": self.get_summary(),
        }


class PredictiveSimulator:
    """Predictive simulation engine.

    Runs simulations to predict system behavior and optimize parameters.

    Attributes:
        simulator_id: Unique simulator identifier
        simulations: List of simulation runs
        current_state: Current system state snapshot
    """

    def __init__(self, simulator_id: str | None = None) -> None:
        """Initialize predictive simulator.

        Args:
            simulator_id: Optional simulator ID
        """
        self.simulator_id = simulator_id or f"sim_{secrets.token_hex(6)}"
        self.simulations: list[dict[str, Any]] = []
        self.current_state: dict[str, float] = {}

    def run_monte_carlo(
        self,
        scenario: str,
        iterations: int = 1000,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation.

        Args:
            scenario: Scenario name
            iterations: Number of iterations
            parameters: Simulation parameters

        Returns:
            Simulation results
        """
        params = parameters or {}
        results = []

        for _ in range(iterations):
            # Simulate outcome based on parameters
            base_value = params.get("base_value", 100)
            volatility = params.get("volatility", 0.2)

            # Random walk simulation
            value = base_value * (1 + random.gauss(0, volatility))
            results.append(value)

        # Calculate statistics
        avg_result = sum(results) / len(results)
        min_result = min(results)
        max_result = max(results)
        std_dev = (sum((r - avg_result) ** 2 for r in results) / len(results)) ** 0.5

        simulation = {
            "simulation_id": f"mc_{len(self.simulations):04d}",
            "type": "monte_carlo",
            "scenario": scenario,
            "iterations": iterations,
            "parameters": params,
            "results": {
                "mean": avg_result,
                "min": min_result,
                "max": max_result,
                "std_dev": std_dev,
                "percentile_5": sorted(results)[int(iterations * 0.05)],
                "percentile_95": sorted(results)[int(iterations * 0.95)],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.simulations.append(simulation)
        return simulation

    def run_system_dynamics(
        self,
        scenario: str,
        time_steps: int = 100,
        initial_state: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run system dynamics simulation.

        Args:
            scenario: Scenario name
            time_steps: Number of time steps
            initial_state: Initial system state

        Returns:
            Simulation results
        """
        state = initial_state or {
            "users": 1000,
            "revenue": 10000,
            "capacity": 100,
            "load": 0.5,
        }

        trajectory = [dict(state)]

        for t in range(time_steps):
            # Growth dynamics
            growth_rate = 0.05 * (1 - state["load"])  # Growth slows as load increases
            state["users"] *= 1 + growth_rate

            # Revenue dynamics
            state["revenue"] = state["users"] * 10  # $10 per user

            # Load dynamics
            state["load"] = min(1.0, state["users"] / (state["capacity"] * 1000))

            # Capacity adjustment (auto-scaling)
            if state["load"] > 0.8:
                state["capacity"] *= 1.1

            trajectory.append(dict(state))

        simulation = {
            "simulation_id": f"sd_{len(self.simulations):04d}",
            "type": "system_dynamics",
            "scenario": scenario,
            "time_steps": time_steps,
            "initial_state": initial_state,
            "final_state": state,
            "trajectory_summary": {
                "users_growth": trajectory[-1]["users"] / trajectory[0]["users"],
                "revenue_growth": trajectory[-1]["revenue"] / trajectory[0]["revenue"],
                "peak_load": max(t["load"] for t in trajectory),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.simulations.append(simulation)
        return simulation

    def predict_scaling_requirements(
        self,
        current_load: float,
        growth_rate: float,
        target_headroom: float = 0.2,
    ) -> dict[str, Any]:
        """Predict infrastructure scaling requirements.

        Args:
            current_load: Current system load (0-1)
            growth_rate: Expected growth rate
            target_headroom: Target capacity headroom

        Returns:
            Scaling predictions
        """
        predictions = []

        load = current_load
        for month in range(1, 13):
            load *= 1 + growth_rate / 12
            required_capacity = load / (1 - target_headroom)
            scale_factor = max(1, required_capacity / current_load)

            predictions.append(
                {
                    "month": month,
                    "predicted_load": min(1.0, load),
                    "required_capacity_factor": scale_factor,
                    "action_needed": scale_factor > 1.2,
                }
            )

        return {
            "current_load": current_load,
            "growth_rate": growth_rate,
            "target_headroom": target_headroom,
            "predictions": predictions,
            "scaling_recommended_month": next(
                (p["month"] for p in predictions if p["action_needed"]), None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator to dictionary."""
        return {
            "simulator_id": self.simulator_id,
            "simulations": self.simulations[-10:],
            "total_simulations": len(self.simulations),
        }


@dataclass
class PolicyAdjustment:
    """Policy adjustment recommendation.

    Attributes:
        adjustment_id: Unique adjustment identifier
        policy_name: Name of policy to adjust
        current_value: Current policy value
        recommended_value: Recommended new value
        rationale: Adjustment rationale
        impact_estimate: Estimated impact
    """

    adjustment_id: str
    policy_name: str
    current_value: Any
    recommended_value: Any
    rationale: str
    impact_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize adjustment to dictionary."""
        return {
            "adjustment_id": self.adjustment_id,
            "policy_name": self.policy_name,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "rationale": self.rationale,
            "impact_estimate": self.impact_estimate,
        }


class PolicyAdjuster:
    """Autonomous policy adjustment system.

    Analyzes system metrics and recommends policy changes.

    Attributes:
        adjuster_id: Unique adjuster identifier
        policies: Current policy values
        adjustments: List of adjustment recommendations
        auto_apply: Whether to auto-apply adjustments
    """

    def __init__(self, adjuster_id: str | None = None) -> None:
        """Initialize policy adjuster.

        Args:
            adjuster_id: Optional adjuster ID
        """
        self.adjuster_id = adjuster_id or f"policy_{secrets.token_hex(6)}"
        self.policies: dict[str, Any] = {
            "consensus_threshold": 67,
            "transaction_fee_rate": 0.001,
            "node_scale_threshold": 0.8,
            "security_alert_threshold": 0.7,
            "data_retention_days": 90,
        }
        self.adjustments: list[PolicyAdjustment] = []
        self.auto_apply = False

    def analyze_and_recommend(self, metrics: dict[str, float]) -> list[PolicyAdjustment]:
        """Analyze metrics and recommend policy adjustments.

        Args:
            metrics: Current system metrics

        Returns:
            List of recommended adjustments
        """
        recommendations = []

        # Consensus threshold adjustment
        validator_participation = metrics.get("validator_participation", 1.0)
        if validator_participation < 0.8:
            adj = PolicyAdjustment(
                adjustment_id=f"adj_{len(self.adjustments):04d}",
                policy_name="consensus_threshold",
                current_value=self.policies["consensus_threshold"],
                recommended_value=max(51, self.policies["consensus_threshold"] - 5),
                rationale="Low validator participation requires lower threshold",
                impact_estimate=0.3,
            )
            recommendations.append(adj)

        # Transaction fee adjustment
        network_load = metrics.get("network_load", 0.5)
        if network_load > 0.9:
            adj = PolicyAdjustment(
                adjustment_id=f"adj_{len(self.adjustments):04d}",
                policy_name="transaction_fee_rate",
                current_value=self.policies["transaction_fee_rate"],
                recommended_value=self.policies["transaction_fee_rate"] * 1.5,
                rationale="High network load requires fee increase to reduce demand",
                impact_estimate=0.4,
            )
            recommendations.append(adj)
        elif network_load < 0.3:
            adj = PolicyAdjustment(
                adjustment_id=f"adj_{len(self.adjustments):04d}",
                policy_name="transaction_fee_rate",
                current_value=self.policies["transaction_fee_rate"],
                recommended_value=self.policies["transaction_fee_rate"] * 0.8,
                rationale="Low network load allows fee reduction to attract usage",
                impact_estimate=0.2,
            )
            recommendations.append(adj)

        # Node scale threshold adjustment
        capacity_util = metrics.get("capacity_utilization", 0.5)
        if capacity_util > 0.9:
            adj = PolicyAdjustment(
                adjustment_id=f"adj_{len(self.adjustments):04d}",
                policy_name="node_scale_threshold",
                current_value=self.policies["node_scale_threshold"],
                recommended_value=0.7,
                rationale="Lower scale threshold for earlier capacity expansion",
                impact_estimate=0.5,
            )
            recommendations.append(adj)

        self.adjustments.extend(recommendations)
        return recommendations

    def apply_adjustment(self, adjustment_id: str) -> bool:
        """Apply a policy adjustment.

        Args:
            adjustment_id: Adjustment to apply

        Returns:
            True if applied successfully
        """
        adj = next((a for a in self.adjustments if a.adjustment_id == adjustment_id), None)
        if adj and adj.policy_name in self.policies:
            self.policies[adj.policy_name] = adj.recommended_value
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize adjuster to dictionary."""
        return {
            "adjuster_id": self.adjuster_id,
            "policies": self.policies,
            "adjustments": [a.to_dict() for a in self.adjustments[-10:]],
            "auto_apply": self.auto_apply,
        }


class ContinuousOptimizer:
    """Main continuous optimization engine.

    Coordinates risk analysis, simulation, and policy adjustment
    for real-time system optimization.

    Attributes:
        optimizer_id: Unique optimizer identifier
        risk_analyzer: Risk analysis component
        simulator: Predictive simulation component
        policy_adjuster: Policy adjustment component
        optimization_actions: Pending optimization actions
        metrics_history: Historical metrics data
    """

    def __init__(self, optimizer_id: str | None = None) -> None:
        """Initialize continuous optimizer.

        Args:
            optimizer_id: Optional optimizer ID
        """
        self.optimizer_id = optimizer_id or f"opt_{secrets.token_hex(6)}"
        self.risk_analyzer = RiskAnalyzer(analyzer_id=f"risk_{self.optimizer_id}")
        self.simulator = PredictiveSimulator(simulator_id=f"sim_{self.optimizer_id}")
        self.policy_adjuster = PolicyAdjuster(adjuster_id=f"policy_{self.optimizer_id}")
        self.optimization_actions: list[OptimizationAction] = []
        self.metrics_history: list[MetricSnapshot] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def collect_metrics(self, metrics: dict[str, float]) -> None:
        """Collect and store metrics.

        Args:
            metrics: Current metrics to collect
        """
        for name, value in metrics.items():
            snapshot = MetricSnapshot(
                metric_id=f"metric_{len(self.metrics_history):06d}",
                name=name,
                value=value,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.metrics_history.append(snapshot)

    def run_optimization_cycle(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Run a complete optimization cycle.

        Args:
            metrics: Current system metrics

        Returns:
            Optimization cycle results
        """
        # Collect metrics
        self.collect_metrics(metrics)

        # Run risk assessments
        infra_risk = self.risk_analyzer.assess_infrastructure_risk(metrics)
        econ_risk = self.risk_analyzer.assess_economic_risk(metrics)
        sec_risk = self.risk_analyzer.assess_security_risk(metrics)

        # Run predictive simulation
        sim_result = self.simulator.run_system_dynamics(
            scenario="optimization_cycle",
            time_steps=50,
            initial_state={
                "users": metrics.get("active_users", 1000),
                "revenue": metrics.get("revenue", 10000),
                "capacity": metrics.get("capacity", 100),
                "load": metrics.get("network_load", 0.5),
            },
        )

        # Get policy recommendations
        policy_recs = self.policy_adjuster.analyze_and_recommend(metrics)

        # Generate optimization actions
        actions = self._generate_actions(infra_risk, econ_risk, sec_risk, sim_result, policy_recs)
        self.optimization_actions.extend(actions)

        return {
            "optimizer_id": self.optimizer_id,
            "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
            "risks": {
                "infrastructure": infra_risk.to_dict(),
                "economic": econ_risk.to_dict(),
                "security": sec_risk.to_dict(),
            },
            "simulation": sim_result,
            "policy_recommendations": [r.to_dict() for r in policy_recs],
            "optimization_actions": [a.to_dict() for a in actions],
        }

    def _generate_actions(
        self,
        infra_risk: RiskAssessment,
        econ_risk: RiskAssessment,
        sec_risk: RiskAssessment,
        sim_result: dict[str, Any],
        policy_recs: list[PolicyAdjustment],
    ) -> list[OptimizationAction]:
        """Generate optimization actions from analysis.

        Args:
            infra_risk: Infrastructure risk assessment
            econ_risk: Economic risk assessment
            sec_risk: Security risk assessment
            sim_result: Simulation result
            policy_recs: Policy recommendations

        Returns:
            List of optimization actions
        """
        actions = []

        # Infrastructure actions
        if infra_risk.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            for mitigation in infra_risk.mitigation_actions:
                actions.append(
                    OptimizationAction(
                        action_id=f"action_{len(self.optimization_actions) + len(actions):04d}",
                        action_type=OptimizationType.INFRASTRUCTURE,
                        description=mitigation,
                        priority=infra_risk.get_risk_score(),
                        impact_estimate=0.3,
                    )
                )

        # Security actions (always high priority)
        if sec_risk.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            for mitigation in sec_risk.mitigation_actions:
                actions.append(
                    OptimizationAction(
                        action_id=f"action_{len(self.optimization_actions) + len(actions):04d}",
                        action_type=OptimizationType.SECURITY,
                        description=mitigation,
                        priority=1.0,  # Security is always top priority
                        impact_estimate=0.5,
                    )
                )

        # Economic actions
        if econ_risk.level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            for mitigation in econ_risk.mitigation_actions:
                actions.append(
                    OptimizationAction(
                        action_id=f"action_{len(self.optimization_actions) + len(actions):04d}",
                        action_type=OptimizationType.ECONOMICS,
                        description=mitigation,
                        priority=econ_risk.get_risk_score(),
                        impact_estimate=0.25,
                    )
                )

        # Governance actions from policy recommendations
        for rec in policy_recs:
            actions.append(
                OptimizationAction(
                    action_id=f"action_{len(self.optimization_actions) + len(actions):04d}",
                    action_type=OptimizationType.GOVERNANCE,
                    description=rec.rationale,
                    priority=rec.impact_estimate,
                    impact_estimate=rec.impact_estimate,
                    target_component=rec.policy_name,
                    parameters={
                        "current": rec.current_value,
                        "recommended": rec.recommended_value,
                    },
                )
            )

        return actions

    def get_statistics(self) -> dict[str, Any]:
        """Get optimizer statistics.

        Returns:
            Optimizer statistics
        """
        actions_by_type: dict[str, int] = {t.value: 0 for t in OptimizationType}
        for action in self.optimization_actions:
            actions_by_type[action.action_type.value] += 1

        return {
            "optimizer_id": self.optimizer_id,
            "total_metrics_collected": len(self.metrics_history),
            "total_risk_assessments": len(self.risk_analyzer.assessments),
            "total_simulations": len(self.simulator.simulations),
            "total_policy_adjustments": len(self.policy_adjuster.adjustments),
            "total_optimization_actions": len(self.optimization_actions),
            "actions_by_type": actions_by_type,
            "risk_summary": self.risk_analyzer.get_summary(),
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize optimizer to dictionary."""
        return {
            "optimizer_id": self.optimizer_id,
            "risk_analyzer": self.risk_analyzer.to_dict(),
            "simulator": self.simulator.to_dict(),
            "policy_adjuster": self.policy_adjuster.to_dict(),
            "optimization_actions": [a.to_dict() for a in self.optimization_actions[-20:]],
            "metrics_history_count": len(self.metrics_history),
            "statistics": self.get_statistics(),
            "created_at": self.created_at,
        }
