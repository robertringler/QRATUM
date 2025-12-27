"""Type definitions for the Reinjection Module.

Defines core types for discovery reinjection into the QRATUM knowledge base,
including scoring, validation, and audit structures.

Key Types:
- ReinjectionScore: Mutual information and cross-impact scoring
- ReinjectionCandidate: Discovery candidate for reinjection
- ReinjectionResult: Result of a reinjection operation
- AuditRecord: Immutable audit record for compliance
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ReinjectionStatus(Enum):
    """Status of a reinjection operation."""

    PENDING = "pending"
    VALIDATING = "validating"
    SANDBOX_TESTING = "sandbox_testing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    FAILED = "failed"


class ValidationLevel(Enum):
    """Level of validation required for reinjection."""

    BASIC = "basic"  # Automated checks only
    STANDARD = "standard"  # Automated + sandbox testing
    ENHANCED = "enhanced"  # Standard + dual-control approval
    CRITICAL = "critical"  # Enhanced + board-level review


class DiscoveryDomain(Enum):
    """Domains for discovery classification."""

    BIODISCOVERY = "biodiscovery"
    GENOMICS = "genomics"
    DRUG_DISCOVERY = "drug_discovery"
    CLIMATE_BIOLOGY = "climate_biology"
    LONGEVITY = "longevity"
    NEURAL = "neural"
    ECONOMIC_BIOLOGICAL = "economic_biological"
    CROSS_VERTICAL = "cross_vertical"


@dataclass
class ReinjectionScore:
    """Scoring for a reinjection candidate.

    Attributes:
        mutual_information: Information gain from this discovery (0-1)
        cross_impact: Impact across multiple verticals (0-1)
        confidence: Confidence in the discovery (0-1)
        novelty: How novel is this finding (0-1)
        entropy_reduction: Expected entropy reduction (0-1)
        compression_efficiency: How well this compresses the hypothesis space (0-1)
        composite_score: Weighted composite of all scores
    """

    mutual_information: float
    cross_impact: float
    confidence: float
    novelty: float = 0.0
    entropy_reduction: float = 0.0
    compression_efficiency: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def composite_score(self) -> float:
        """Compute weighted composite score."""
        weights = {
            "mutual_information": 0.25,
            "cross_impact": 0.20,
            "confidence": 0.25,
            "novelty": 0.10,
            "entropy_reduction": 0.10,
            "compression_efficiency": 0.10,
        }
        return (
            weights["mutual_information"] * self.mutual_information
            + weights["cross_impact"] * self.cross_impact
            + weights["confidence"] * self.confidence
            + weights["novelty"] * self.novelty
            + weights["entropy_reduction"] * self.entropy_reduction
            + weights["compression_efficiency"] * self.compression_efficiency
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize score."""
        return {
            "mutual_information": self.mutual_information,
            "cross_impact": self.cross_impact,
            "confidence": self.confidence,
            "novelty": self.novelty,
            "entropy_reduction": self.entropy_reduction,
            "compression_efficiency": self.compression_efficiency,
            "composite_score": self.composite_score,
            "timestamp": self.timestamp,
        }


@dataclass
class ReinjectionCandidate:
    """Candidate discovery for reinjection.

    Attributes:
        candidate_id: Unique identifier
        discovery_id: ID of the source discovery
        domain: Discovery domain
        description: Human-readable description
        data_payload: Discovery data to reinject
        score: Reinjection scoring
        target_priors: Target priors to update
        source_workflow_id: Source workflow that produced this
        provenance_hash: Hash of provenance chain
    """

    candidate_id: str
    discovery_id: str
    domain: DiscoveryDomain
    description: str
    data_payload: dict[str, Any]
    score: ReinjectionScore
    target_priors: list[str]
    source_workflow_id: str
    provenance_hash: str
    status: ReinjectionStatus = ReinjectionStatus.PENDING
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute hash of candidate content."""
        content = {
            "candidate_id": self.candidate_id,
            "discovery_id": self.discovery_id,
            "domain": self.domain.value,
            "description": self.description,
            "data_payload": self.data_payload,
            "target_priors": self.target_priors,
            "source_workflow_id": self.source_workflow_id,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize candidate."""
        return {
            "candidate_id": self.candidate_id,
            "discovery_id": self.discovery_id,
            "domain": self.domain.value,
            "description": self.description,
            "data_payload": self.data_payload,
            "score": self.score.to_dict(),
            "target_priors": self.target_priors,
            "source_workflow_id": self.source_workflow_id,
            "provenance_hash": self.provenance_hash,
            "status": self.status.value,
            "validation_level": self.validation_level.value,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "content_hash": self.compute_hash(),
        }


@dataclass
class SandboxResult:
    """Result of sandbox testing.

    Attributes:
        sandbox_id: Unique sandbox session ID
        candidate_id: Tested candidate ID
        success: Whether sandbox test succeeded
        fidelity_score: Measured fidelity improvement
        rollback_tested: Whether rollback was successfully tested
        side_effects: Any detected side effects
        execution_time_ms: Sandbox execution time
    """

    sandbox_id: str
    candidate_id: str
    success: bool
    fidelity_score: float
    rollback_tested: bool
    side_effects: list[str]
    execution_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize sandbox result."""
        return {
            "sandbox_id": self.sandbox_id,
            "candidate_id": self.candidate_id,
            "success": self.success,
            "fidelity_score": self.fidelity_score,
            "rollback_tested": self.rollback_tested,
            "side_effects": self.side_effects,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
        }


@dataclass
class ReinjectionResult:
    """Result of a complete reinjection operation.

    Attributes:
        result_id: Unique result identifier
        candidate_id: Injected candidate ID
        status: Final status
        validation_passed: Whether validation passed
        sandbox_result: Sandbox testing result
        approvers: List of approvers (for dual-control)
        committed_at: Commitment timestamp
        rollback_id: ID for potential rollback
        merkle_proof: Merkle proof of commitment
    """

    result_id: str
    candidate_id: str
    status: ReinjectionStatus
    validation_passed: bool
    sandbox_result: SandboxResult | None
    approvers: list[str]
    committed_at: str | None
    rollback_id: str | None
    merkle_proof: str
    audit_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""
        return {
            "result_id": self.result_id,
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "validation_passed": self.validation_passed,
            "sandbox_result": self.sandbox_result.to_dict() if self.sandbox_result else None,
            "approvers": self.approvers,
            "committed_at": self.committed_at,
            "rollback_id": self.rollback_id,
            "merkle_proof": self.merkle_proof,
            "audit_hash": self.audit_hash,
            "created_at": self.created_at,
            "metrics": self.metrics,
        }


@dataclass
class AuditRecord:
    """Immutable audit record for compliance.

    Attributes:
        audit_id: Unique audit identifier
        operation_type: Type of operation audited
        actor_id: ID of actor performing operation
        candidate_id: Related candidate ID
        status: Operation status
        details: Detailed audit information
        merkle_root: Merkle root at time of audit
        compliance_tags: Applicable compliance frameworks
    """

    audit_id: str
    operation_type: str
    actor_id: str
    candidate_id: str
    status: str
    details: dict[str, Any]
    merkle_root: str
    compliance_tags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_hash(self) -> str:
        """Compute immutable hash of audit record."""
        content = {
            "audit_id": self.audit_id,
            "operation_type": self.operation_type,
            "actor_id": self.actor_id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "details": self.details,
            "merkle_root": self.merkle_root,
            "compliance_tags": self.compliance_tags,
            "timestamp": self.timestamp,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit record."""
        return {
            "audit_id": self.audit_id,
            "operation_type": self.operation_type,
            "actor_id": self.actor_id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "details": self.details,
            "merkle_root": self.merkle_root,
            "compliance_tags": self.compliance_tags,
            "timestamp": self.timestamp,
            "record_hash": self.compute_hash(),
        }
