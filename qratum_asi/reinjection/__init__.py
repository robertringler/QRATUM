"""QRATUM Reinjection Module.

Implements the discovery reinjection feedback loop for QRATUM ASI.
Enables validated discoveries to be reinjected into biodiscovery library
priors with full provenance tracking and dual-control governance.

Version: 1.0.0
Status: Production Ready
QuASIM: v2025.12.26

Key Features:
- Validation with domain-specific rules
- Z1 sandbox testing with rollback verification
- Dual-control authorization via QIL contracts
- Z2 commitment with full Merkle provenance
- Comprehensive audit report generation
- Full compliance mapping (GDPR, HIPAA, 21 CFR Part 11, Nagoya)

Usage:
    from qratum_asi.reinjection import (
        ReinjectionEngine,
        ReinjectionCandidate,
        DiscoveryDomain,
    )

    # Initialize engine
    engine = ReinjectionEngine()

    # Create candidate
    candidate = engine.create_candidate(
        discovery_id="disc_001",
        domain=DiscoveryDomain.BIODISCOVERY,
        description="High-confidence natural compound discovery",
        data_payload={"compound_data": {...}},
        mutual_information=0.75,
        cross_impact=0.65,
        confidence=0.85,
        target_priors=["compound_affinity"],
        source_workflow_id="wf_001",
    )

    # Execute full reinjection cycle
    result = engine.execute_full_cycle(
        candidate=candidate,
        approvers=["reviewer_1", "reviewer_2"],
        auto_approve=False,  # Set to True only for testing
    )
"""

# Type definitions
# Audit
from qratum_asi.reinjection.audit import (
    DOMAIN_COMPLIANCE_FRAMEWORKS,
    AuditReport,
    AuditReportGenerator,
    ComplianceCheck,
)

# Autonomous orchestrator
from qratum_asi.reinjection.autonomous_orchestrator import (
    CROSS_VERTICAL_DEPENDENCIES,
    ArtifactSensitivity,
    AutonomousReinjectionOrchestrator,
    DiscoveryArtifact,
    PropagationResult,
    PropagationTarget,
    ReinjectionStatusSummary,
    SystemState,
    create_artifact_from_discovery_result,
)

# Contracts
from qratum_asi.reinjection.contracts import (
    ApprovalRecord,
    ReinjectionContract,
    ReinjectionContractStatus,
    create_reinjection_contract,
)

# Core engine
from qratum_asi.reinjection.engine import (
    ReinjectionCycleResult,
    ReinjectionEngine,
    create_synthetic_discovery_candidate,
)

# Mapper
from qratum_asi.reinjection.mapper import (
    DOMAIN_PRIOR_TYPES,
    DiscoveryPriorMapper,
    MappingResult,
    PriorUpdate,
)

# Sandbox
from qratum_asi.reinjection.sandbox import (
    RollbackTest,
    SandboxOrchestrator,
    SandboxState,
)

# SOI Telemetry integration
from qratum_asi.reinjection.soi_telemetry import (
    EvolutionDataPoint,
    OptionalityMetrics,
    SOIReinjectionTelemetry,
    TelemetryEvent,
)
from qratum_asi.reinjection.types import (
    AuditRecord,
    DiscoveryDomain,
    ReinjectionCandidate,
    ReinjectionResult,
    ReinjectionScore,
    ReinjectionStatus,
    SandboxResult,
    ValidationLevel,
)

# Validator
from qratum_asi.reinjection.validator import (
    DOMAIN_CONFIDENCE_THRESHOLDS,
    MINIMUM_COMPOSITE_SCORE,
    MINIMUM_MUTUAL_INFORMATION,
    ReinjectionValidator,
    ValidationResult,
)

__version__ = "1.0.0"

__all__ = [
    # Type definitions
    "AuditRecord",
    "DiscoveryDomain",
    "ReinjectionCandidate",
    "ReinjectionResult",
    "ReinjectionScore",
    "ReinjectionStatus",
    "SandboxResult",
    "ValidationLevel",
    # Validator
    "ReinjectionValidator",
    "ValidationResult",
    "DOMAIN_CONFIDENCE_THRESHOLDS",
    "MINIMUM_COMPOSITE_SCORE",
    "MINIMUM_MUTUAL_INFORMATION",
    # Mapper
    "DiscoveryPriorMapper",
    "MappingResult",
    "PriorUpdate",
    "DOMAIN_PRIOR_TYPES",
    # Sandbox
    "SandboxOrchestrator",
    "SandboxState",
    "RollbackTest",
    # Contracts
    "ReinjectionContract",
    "ReinjectionContractStatus",
    "ApprovalRecord",
    "create_reinjection_contract",
    # Audit
    "AuditReport",
    "AuditReportGenerator",
    "ComplianceCheck",
    "DOMAIN_COMPLIANCE_FRAMEWORKS",
    # Core engine
    "ReinjectionEngine",
    "ReinjectionCycleResult",
    "create_synthetic_discovery_candidate",
    # Autonomous orchestrator
    "AutonomousReinjectionOrchestrator",
    "DiscoveryArtifact",
    "PropagationResult",
    "PropagationTarget",
    "ReinjectionStatusSummary",
    "ArtifactSensitivity",
    "SystemState",
    "CROSS_VERTICAL_DEPENDENCIES",
    "create_artifact_from_discovery_result",
    # SOI Telemetry
    "SOIReinjectionTelemetry",
    "TelemetryEvent",
    "OptionalityMetrics",
    "EvolutionDataPoint",
]
