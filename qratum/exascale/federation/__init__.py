"""
Federated Control Plane

Byzantine-fault-tolerant coordination enabling multi-site coherent federation
with sovereign isolation and air-gap compatible deployment.

Components:
- control_plane: Federated control orchestration
- bft_hotstuff: BFT-HotStuff consensus protocol
- sovereign: Air-gap deployment capabilities
"""

from .control_plane import FederatedControlPlane, SiteCoordinator
from .bft_hotstuff import BFTHotStuff, ConsensusNode
from .sovereign import SovereignDeployment, AirGapController

__all__ = [
    "FederatedControlPlane",
    "SiteCoordinator",
    "BFTHotStuff",
    "ConsensusNode",
    "SovereignDeployment",
    "AirGapController",
]
