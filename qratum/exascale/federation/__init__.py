"""
Federated Control Plane

Byzantine-fault-tolerant coordination enabling multi-site coherent federation
with sovereign isolation and air-gap compatible deployment.

Components:
- control_plane: Federated control orchestration
- bft_hotstuff: BFT-HotStuff consensus protocol
- sovereign: Air-gap deployment capabilities
"""

from .control_plane import (
    FederationControlPlane,
    Site,
    FederatedJob,
    GlobalState,
    SiteStatus,
    FederationMode,
    FederationBuilder,
)
from .bft_hotstuff import (
    BFTHotStuffConsensus,
    BFTReplica,
    Proposal,
    QuorumCertificate,
    MessageType,
    ProposalStatus,
)
from .sovereign import (
    SovereignDeployment,
    AirGapDeployment,
    SecurePackage,
    TransferManifest,
    AirGapDeploymentBuilder,
    DeploymentVerifier,
    IsolationLevel,
    TransferMethod,
    PackageType,
)

__all__ = [
    # Control Plane
    "FederationControlPlane",
    "Site",
    "FederatedJob",
    "GlobalState",
    "SiteStatus",
    "FederationMode",
    "FederationBuilder",
    # BFT-HotStuff
    "BFTHotStuffConsensus",
    "BFTReplica",
    "Proposal",
    "QuorumCertificate",
    "MessageType",
    "ProposalStatus",
    # Sovereign
    "SovereignDeployment",
    "AirGapDeployment",
    "SecurePackage",
    "TransferManifest",
    "AirGapDeploymentBuilder",
    "DeploymentVerifier",
    "IsolationLevel",
    "TransferMethod",
    "PackageType",
]
