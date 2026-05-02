"""
Federated Control Plane

Byzantine-fault-tolerant coordination enabling multi-site coherent federation
with sovereign isolation and air-gap compatible deployment.

Components:
- control_plane: Federated control orchestration
- bft_hotstuff: BFT-HotStuff consensus protocol
- sovereign: Air-gap deployment capabilities
"""

from .bft_hotstuff import (BFTHotStuffConsensus, BFTReplica, MessageType,
                           Proposal, ProposalStatus, QuorumCertificate)
from .control_plane import (FederatedJob, FederationBuilder,
                            FederationControlPlane, FederationMode,
                            GlobalState, Site, SiteStatus)
from .sovereign import (AirGapDeployment, AirGapDeploymentBuilder,
                        DeploymentVerifier, IsolationLevel, PackageType,
                        SecurePackage, SovereignDeployment, TransferManifest,
                        TransferMethod)

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
