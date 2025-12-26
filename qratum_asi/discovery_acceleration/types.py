"""Discovery Acceleration Type Definitions.

Centralized type definitions for the Discovery Acceleration Module.
Extracted from workflows.py for better organization and type safety.

Version: 1.0.0
Status: Production Ready
QuASIM: v2025.12.26
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiscoveryType(Enum):
    """Types of discoveries to accelerate."""

    COMPLEX_DISEASE_GENETICS = "complex_disease_genetics"
    PERSONALIZED_DRUG_DESIGN = "personalized_drug_design"
    CLIMATE_GENE_CONNECTIONS = "climate_gene_connections"
    NATURAL_DRUG_DISCOVERY = "natural_drug_discovery"
    ECONOMIC_BIOLOGICAL_MODEL = "economic_biological_model"
    ANTI_AGING_PATHWAYS = "anti_aging_pathways"


class WorkflowStage(Enum):
    """Stages in a discovery workflow."""

    INITIALIZATION = "initialization"
    INPUT_VALIDATION = "input_validation"
    ZK_PROOF_GENERATION = "zk_proof_generation"
    DETERMINISTIC_PROCESSING = "deterministic_processing"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    CROSS_VERTICAL_SYNTHESIS = "cross_vertical_synthesis"
    VALIDATION = "validation"
    ROLLBACK_POINT = "rollback_point"
    OUTPUT_GENERATION = "output_generation"
    PROVENANCE_CHAIN = "provenance_chain"


@dataclass
class WorkflowArtifact:
    """Artifact produced by a workflow stage.

    Attributes:
        artifact_id: Unique identifier
        stage: Workflow stage that produced this artifact
        data_hash: SHA3-256 hash of artifact data
        merkle_root: Merkle root for this artifact
        timestamp: Creation timestamp
        metadata: Additional metadata
    """

    artifact_id: str
    stage: WorkflowStage
    data_hash: str
    merkle_root: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize artifact."""
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage.value,
            "data_hash": self.data_hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class RollbackPoint:
    """Rollback point for workflow recovery.

    Attributes:
        rollback_id: Unique identifier
        stage: Stage where rollback point was created
        state_snapshot: Snapshot of workflow state
        merkle_root: Merkle root at this point
        timestamp: Creation timestamp
    """

    rollback_id: str
    stage: WorkflowStage
    state_snapshot: dict[str, Any]
    merkle_root: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize rollback point."""
        return {
            "rollback_id": self.rollback_id,
            "stage": self.stage.value,
            "state_hash": hashlib.sha3_256(
                json.dumps(self.state_snapshot, sort_keys=True).encode()
            ).hexdigest(),
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
        }


@dataclass
class DiscoveryResult:
    """Result of a discovery workflow execution.

    Attributes:
        workflow_id: Workflow identifier
        discovery_type: Type of discovery
        success: Whether workflow succeeded
        insights: Generated insights
        hypotheses: Generated hypotheses with confidence scores
        provenance_chain: Merkle chain proof
        artifacts: Produced artifacts
        rollback_points: Available rollback points
        execution_time_seconds: Total execution time
        timestamp: Completion timestamp
    """

    workflow_id: str
    discovery_type: DiscoveryType
    success: bool
    insights: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    provenance_chain: str
    artifacts: list[WorkflowArtifact]
    rollback_points: list[RollbackPoint]
    execution_time_seconds: float
    timestamp: str
    projections: dict[str, Any] = field(default_factory=dict)
    compliance_mapping: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""
        return {
            "workflow_id": self.workflow_id,
            "discovery_type": self.discovery_type.value,
            "success": self.success,
            "insights": self.insights,
            "hypotheses": self.hypotheses,
            "provenance_chain": self.provenance_chain,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "rollback_points": [r.to_dict() for r in self.rollback_points],
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp,
            "projections": self.projections,
            "compliance_mapping": self.compliance_mapping,
        }


@dataclass
class DiscoveryProjection:
    """Quantitative projection for a discovery type.

    Attributes:
        discovery_type: Type of discovery
        discovery_probability: Estimated probability of breakthrough (0-1)
        time_savings_factor: Speed multiplier vs legacy methods
        risk_mitigation_score: Safety score from trajectory monitoring (0-1)
        estimated_timeline_months: Estimated timeline in months
        legacy_timeline_months: Legacy method timeline in months
        additional_metrics: Additional type-specific metrics
    """

    discovery_type: DiscoveryType
    discovery_probability: float
    time_savings_factor: float
    risk_mitigation_score: float
    estimated_timeline_months: int
    legacy_timeline_months: int
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize projection."""
        return {
            "discovery_type": self.discovery_type.value,
            "discovery_probability": self.discovery_probability,
            "time_savings_factor": self.time_savings_factor,
            "risk_mitigation_score": self.risk_mitigation_score,
            "estimated_timeline_months": self.estimated_timeline_months,
            "legacy_timeline_months": self.legacy_timeline_months,
            **self.additional_metrics,
        }


@dataclass
class TimelineSimulation:
    """Timeline simulation result.

    Attributes:
        discovery_type: Type of discovery
        parameters: Simulation parameters
        baseline_months: Baseline estimated timeline
        optimistic_months: Optimistic scenario timeline
        pessimistic_months: Pessimistic scenario timeline
        confidence_interval: Confidence interval (low, high)
        risk_factors: Identified risk factors
    """

    discovery_type: DiscoveryType
    parameters: dict[str, Any]
    baseline_months: float
    optimistic_months: float
    pessimistic_months: float
    confidence_interval: tuple[float, float]
    risk_factors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulation."""
        return {
            "discovery_type": self.discovery_type.value,
            "parameters": self.parameters,
            "baseline_months": self.baseline_months,
            "optimistic_months": self.optimistic_months,
            "pessimistic_months": self.pessimistic_months,
            "confidence_interval": self.confidence_interval,
            "risk_factors": self.risk_factors,
        }


@dataclass
class RiskAssessment:
    """Risk assessment for a workflow.

    Attributes:
        workflow_id: Workflow identifier
        overall_risk_score: Overall risk score (0-1, lower is better)
        vulnerability_score: Vulnerability detection score
        trajectory_compliance: Trajectory compliance score
        risk_factors: Identified risk factors with severity
        mitigation_recommendations: Recommended mitigations
    """

    workflow_id: str
    overall_risk_score: float
    vulnerability_score: float
    trajectory_compliance: float
    risk_factors: list[dict[str, Any]]
    mitigation_recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize assessment."""
        return {
            "workflow_id": self.workflow_id,
            "overall_risk_score": self.overall_risk_score,
            "vulnerability_score": self.vulnerability_score,
            "trajectory_compliance": self.trajectory_compliance,
            "risk_factors": self.risk_factors,
            "mitigation_recommendations": self.mitigation_recommendations,
        }


@dataclass
class ComplianceMapping:
    """Compliance mapping for a discovery type.

    Attributes:
        discovery_type: Type of discovery
        frameworks: Regulatory frameworks mapped
        status: Overall compliance status
        controls: Required controls
        audit_requirements: Audit requirements
    """

    discovery_type: DiscoveryType
    frameworks: dict[str, dict[str, Any]]
    status: str
    controls: list[str]
    audit_requirements: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize mapping."""
        return {
            "discovery_type": self.discovery_type.value,
            "frameworks": self.frameworks,
            "status": self.status,
            "controls": self.controls,
            "audit_requirements": self.audit_requirements,
        }


@dataclass
class ComplianceArtifact:
    """Runtime compliance artifact.

    Attributes:
        artifact_id: Unique identifier
        contract_id: Associated contract identifier
        framework: Regulatory framework
        evidence: Compliance evidence
        timestamp: Creation timestamp
        merkle_root: Merkle root for provenance
    """

    artifact_id: str
    contract_id: str
    framework: str
    evidence: dict[str, Any]
    timestamp: str
    merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize artifact."""
        return {
            "artifact_id": self.artifact_id,
            "contract_id": self.contract_id,
            "framework": self.framework,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
        }


@dataclass
class ComplianceValidationResult:
    """Compliance validation result.

    Attributes:
        workflow_id: Workflow identifier
        is_compliant: Whether workflow is compliant
        validated_frameworks: Validated frameworks
        violations: Detected violations
        recommendations: Recommendations for compliance
    """

    workflow_id: str
    is_compliant: bool
    validated_frameworks: list[str]
    violations: list[dict[str, Any]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""
        return {
            "workflow_id": self.workflow_id,
            "is_compliant": self.is_compliant,
            "validated_frameworks": self.validated_frameworks,
            "violations": self.violations,
            "recommendations": self.recommendations,
        }


# Type aliases for enhanced type safety
WorkflowID = str
ArtifactID = str
RollbackID = str
ContractID = str
FrameworkName = str
