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
    ReinjectionValidator,
    ValidationResult,
    DOMAIN_CONFIDENCE_THRESHOLDS,
    MINIMUM_COMPOSITE_SCORE,
    MINIMUM_MUTUAL_INFORMATION,
)

# Mapper
from qratum_asi.reinjection.mapper import (
    DiscoveryPriorMapper,
    MappingResult,
    PriorUpdate,
    DOMAIN_PRIOR_TYPES,
)

# Sandbox
from qratum_asi.reinjection.sandbox import (
    SandboxOrchestrator,
    SandboxState,
    RollbackTest,
)

# Contracts
from qratum_asi.reinjection.contracts import (
    ReinjectionContract,
    ReinjectionContractStatus,
    ApprovalRecord,
    create_reinjection_contract,
)

# Audit
from qratum_asi.reinjection.audit import (
    AuditReport,
    AuditReportGenerator,
    ComplianceCheck,
    DOMAIN_COMPLIANCE_FRAMEWORKS,
)

# Core engine
from qratum_asi.reinjection.engine import (
    ReinjectionEngine,
    ReinjectionCycleResult,
    create_synthetic_discovery_candidate,
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
]
