"""QRATUM Discovery Acceleration Module.

Harnesses QRATUM ASI and QRADLE for breakthrough discoveries across
6 target areas with invariant-preserving, auditable workflows.

Version: 1.0.0
Status: Production Ready
QuASIM: v2025.12.26
"""

from qratum_asi.discovery_acceleration.contracts import (
    CrossVerticalIntent,
    DiscoveryContract,
)
from qratum_asi.discovery_acceleration.federated_gwas import (
    FederatedGWASPipeline,
    GWASCohort,
    GWASResult,
)
from qratum_asi.discovery_acceleration.workflows import (
    DiscoveryAccelerationEngine,
    DiscoveryResult,
    DiscoveryType,
    DiscoveryWorkflow,
    WorkflowStage,
)

__all__ = [
    "DiscoveryAccelerationEngine",
    "DiscoveryWorkflow",
    "DiscoveryType",
    "DiscoveryResult",
    "WorkflowStage",
    "FederatedGWASPipeline",
    "GWASCohort",
    "GWASResult",
    "DiscoveryContract",
    "CrossVerticalIntent",
]
