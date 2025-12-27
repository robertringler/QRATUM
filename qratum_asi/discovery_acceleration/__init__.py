"""QRATUM Discovery Acceleration Module.

Harnesses QRATUM ASI and QRADLE for breakthrough discoveries across
6 target areas with invariant-preserving, auditable workflows.

Version: 1.0.0
Status: Production Ready
QuASIM: v2025.12.26
"""

# Import from workflows module file (workflows.py)
from qratum_asi.discovery_acceleration.compliance_mapper import (
    ComplianceMapper,
)
from qratum_asi.discovery_acceleration.contracts import (
    CrossVerticalIntent,
    DiscoveryContract,
)
from qratum_asi.discovery_acceleration.federated_gwas import (
    FederatedGWASPipeline,
    GWASCohort,
    GWASResult,
)
from qratum_asi.discovery_acceleration.pipelines.climate_gene import (
    ClimateGenePipeline,
)
from qratum_asi.discovery_acceleration.pipelines.economic_bio import (
    EconomicBioPipeline,
)
from qratum_asi.discovery_acceleration.pipelines.longevity import (
    LongevityPipeline,
)
from qratum_asi.discovery_acceleration.pipelines.natural_compound import (
    NaturalCompoundPipeline,
)

# Import from pipelines package (pipelines/)
from qratum_asi.discovery_acceleration.pipelines.personalized_drug import (
    PersonalizedDrugPipeline,
)
from qratum_asi.discovery_acceleration.projections import (
    DiscoveryProjectionsEngine,
)
from qratum_asi.discovery_acceleration.types import (
    ComplianceArtifact,
    ComplianceMapping,
    ComplianceValidationResult,
    DiscoveryProjection,
    RiskAssessment,
    RollbackPoint,
    TimelineSimulation,
    WorkflowArtifact,
)
from qratum_asi.discovery_acceleration.workflows import (
    DiscoveryAccelerationEngine,
    DiscoveryResult,
    DiscoveryType,
    DiscoveryWorkflow,
    WorkflowStage,
)

__all__ = [
    # Core workflow components
    "DiscoveryAccelerationEngine",
    "DiscoveryWorkflow",
    "DiscoveryType",
    "DiscoveryResult",
    "WorkflowStage",
    # Federated GWAS
    "FederatedGWASPipeline",
    "GWASCohort",
    "GWASResult",
    # Contracts
    "DiscoveryContract",
    "CrossVerticalIntent",
    # Type definitions
    "WorkflowArtifact",
    "RollbackPoint",
    "DiscoveryProjection",
    "TimelineSimulation",
    "RiskAssessment",
    "ComplianceMapping",
    "ComplianceArtifact",
    "ComplianceValidationResult",
    # Projections and compliance
    "DiscoveryProjectionsEngine",
    "ComplianceMapper",
    # Discovery pipelines
    "PersonalizedDrugPipeline",
    "ClimateGenePipeline",
    "NaturalCompoundPipeline",
    "EconomicBioPipeline",
    "LongevityPipeline",
]
