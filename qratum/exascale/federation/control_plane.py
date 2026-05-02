"""
Multi-Site Coherent Federation Control Plane

Orchestrates QRATUM ExaScale federation across multiple geographic sites
with:
- Coherent global state management
- Site-level fault isolation
- Deterministic cross-site execution
- BFT consensus for coordination
- Quantum-safe communication

Federation Topology:
- 5 geographic sites (US-East, US-West, EU, Asia-Pacific, Middle East)
- 625 nodes per site
- 3,125 nodes total
- Full mesh inter-site connectivity (25 Tbps AetherFabric-X)

Control Plane Functions:
1. Global state synchronization
2. Cross-site job scheduling
3. Federated consensus (BFT-HotStuff)
4. Site failover and recovery
5. Network partition handling
6. Quantum state distribution

Design Principles:
- CAP theorem: Choose consistency + partition tolerance (CP)
- Eventually consistent for non-critical data
- Strongly consistent for deterministic execution
- Byzantine fault tolerance for adversarial environments
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from qratum.exascale.federation.bft_hotstuff import BFTHotStuffConsensus


class SiteStatus(Enum):
    """Site operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    PARTITIONED = "partitioned"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class FederationMode(Enum):
    """Federation operational modes"""
    NORMAL = "normal"  # All sites operational
    DEGRADED = "degraded"  # Some sites degraded
    EMERGENCY = "emergency"  # Critical failures
    SPLIT_BRAIN = "split_brain"  # Network partition


class StateConsistencyLevel(Enum):
    """State consistency requirements"""
    STRONG = "strong"  # Linearizable (consensus required)
    EVENTUAL = "eventual"  # Eventually consistent
    CAUSAL = "causal"  # Causally consistent
    READ_YOUR_WRITES = "read_your_writes"  # Session consistency


class CrossSiteOperation(Enum):
    """Cross-site operations"""
    STATE_SYNC = "state_sync"  # State synchronization
    JOB_MIGRATION = "job_migration"  # Job migration between sites
    DATA_REPLICATION = "data_replication"  # Data replication
    CONSENSUS_VOTE = "consensus_vote"  # Consensus voting
    HEALTH_CHECK = "health_check"  # Health monitoring


@dataclass
class Site:
    """
    Geographic site in federation
    
    Attributes:
        site_id: Unique site identifier
        location: Geographic location
        num_nodes: Number of compute nodes
        status: Current site status
        connected_sites: Set of sites this site can reach
        last_heartbeat_ns: Last heartbeat timestamp
        state_version: Current state version number
        state_hash: Hash of current state
    """
    site_id: str
    location: str
    num_nodes: int = 625  # 625 nodes per site
    status: SiteStatus = SiteStatus.HEALTHY
    connected_sites: Set[str] = field(default_factory=set)
    last_heartbeat_ns: int = 0
    state_version: int = 0
    state_hash: str = ""

    def __post_init__(self):
        """Initialize site"""
        if self.last_heartbeat_ns == 0:
            object.__setattr__(self, 'last_heartbeat_ns', int(time.time() * 1e9))

        if not self.state_hash:
            object.__setattr__(self, 'state_hash', self.compute_state_hash())

    def compute_state_hash(self) -> str:
        """
        Compute hash of site state
        
        Returns:
            SHA-256 hash as hex string
        """
        data = (
            f"{self.site_id}:{self.location}:{self.num_nodes}:"
            f"{self.status.value}:{self.state_version}"
        )
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def send_heartbeat(self) -> None:
        """Update heartbeat timestamp"""
        object.__setattr__(self, 'last_heartbeat_ns', int(time.time() * 1e9))

    def is_healthy(self, max_age_ms: float = 5000.0) -> bool:
        """
        Check if site is healthy based on heartbeat
        
        Args:
            max_age_ms: Maximum heartbeat age in milliseconds
            
        Returns:
            True if site is healthy
        """
        if self.status != SiteStatus.HEALTHY:
            return False

        current_ns = int(time.time() * 1e9)
        age_ms = (current_ns - self.last_heartbeat_ns) / 1_000_000.0

        return age_ms <= max_age_ms

    def can_reach(self, site_id: str) -> bool:
        """Check if this site can reach another site"""
        return site_id in self.connected_sites

    def update_state(self, new_state_hash: str) -> None:
        """
        Update site state
        
        Args:
            new_state_hash: Hash of new state
        """
        object.__setattr__(self, 'state_version', self.state_version + 1)
        object.__setattr__(self, 'state_hash', new_state_hash)


@dataclass
class FederatedJob:
    """
    Job spanning multiple sites
    
    Attributes:
        job_id: Unique job identifier
        primary_site: Primary execution site
        replica_sites: Replica sites for fault tolerance
        required_consistency: Consistency level required
        state_hash: Hash of job state
        status: Current job status
        created_ns: Job creation timestamp
    """
    job_id: str
    primary_site: str
    replica_sites: List[str] = field(default_factory=list)
    required_consistency: StateConsistencyLevel = StateConsistencyLevel.STRONG
    state_hash: str = ""
    status: str = "pending"
    created_ns: int = 0

    def __post_init__(self):
        """Initialize job"""
        if self.created_ns == 0:
            object.__setattr__(self, 'created_ns', int(time.time() * 1e9))

    def get_all_sites(self) -> List[str]:
        """Get all sites involved in job"""
        return [self.primary_site] + self.replica_sites

    def can_execute_on_sites(self, available_sites: Set[str]) -> bool:
        """
        Check if job can execute with available sites
        
        Args:
            available_sites: Set of available site IDs
            
        Returns:
            True if enough sites available
        """
        required_sites = set(self.get_all_sites())
        return required_sites.issubset(available_sites)


@dataclass
class GlobalState:
    """
    Global federation state
    
    Represents the consistent state across all sites.
    
    Attributes:
        version: Global state version
        state_hash: Hash of global state
        site_states: Map of site_id -> state_hash
        last_consensus_ns: Last consensus timestamp
        pending_updates: Pending state updates
    """
    version: int = 0
    state_hash: str = ""
    site_states: Dict[str, str] = field(default_factory=dict)
    last_consensus_ns: int = 0
    pending_updates: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize global state"""
        if not self.state_hash:
            object.__setattr__(self, 'state_hash', self.compute_global_hash())

    def compute_global_hash(self) -> str:
        """
        Compute hash of global state
        
        Returns:
            SHA-256 hash as hex string
        """
        # Hash all site states in deterministic order
        site_data = ";".join(
            f"{site_id}:{state_hash}"
            for site_id, state_hash in sorted(self.site_states.items())
        )
        data = f"{self.version}:{site_data}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def update_site_state(self, site_id: str, state_hash: str) -> None:
        """
        Update state for specific site
        
        Args:
            site_id: Site identifier
            state_hash: New state hash
        """
        self.site_states[site_id] = state_hash
        self.pending_updates.append((site_id, state_hash))

    def commit_updates(self) -> None:
        """Commit pending updates to global state"""
        object.__setattr__(self, 'version', self.version + 1)
        object.__setattr__(self, 'state_hash', self.compute_global_hash())
        object.__setattr__(self, 'last_consensus_ns', int(time.time() * 1e9))
        self.pending_updates.clear()

    def verify_consistency(self) -> bool:
        """
        Verify global state consistency
        
        Returns:
            True if all site states are consistent
        """
        computed_hash = self.compute_global_hash()
        return computed_hash == self.state_hash


@dataclass
class FederationControlPlane:
    """
    Multi-Site Federation Control Plane
    
    Orchestrates federation across geographic sites with:
    - Global state management
    - BFT consensus coordination
    - Site health monitoring
    - Cross-site job scheduling
    - Network partition handling
    
    Attributes:
        federation_id: Unique federation identifier
        sites: Map of site_id -> Site
        global_state: Global federation state
        consensus: BFT-HotStuff consensus engine
        mode: Current operational mode
        jobs: Map of job_id -> FederatedJob
        active_operations: Currently active cross-site operations
    """
    federation_id: str
    sites: Dict[str, Site] = field(default_factory=dict)
    global_state: GlobalState = field(default_factory=GlobalState)
    consensus: Optional[BFTHotStuffConsensus] = None
    mode: FederationMode = FederationMode.NORMAL
    jobs: Dict[str, FederatedJob] = field(default_factory=dict)
    active_operations: Dict[str, CrossSiteOperation] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize control plane"""
        # Initialize consensus if not provided
        if self.consensus is None:
            # Use all nodes across all sites for consensus
            total_nodes = sum(site.num_nodes for site in self.sites.values())
            if total_nodes >= 4:  # Minimum for BFT
                object.__setattr__(
                    self,
                    'consensus',
                    BFTHotStuffConsensus(total_replicas=min(total_nodes, 3125))
                )

    def add_site(self, site: Site) -> None:
        """
        Add site to federation
        
        Args:
            site: Site to add
        """
        self.sites[site.site_id] = site
        self.global_state.update_site_state(site.site_id, site.state_hash)

    def update_site_connectivity(self, site_id: str,
                                 connected_sites: Set[str]) -> None:
        """
        Update connectivity information for site
        
        Args:
            site_id: Site identifier
            connected_sites: Set of reachable site IDs
        """
        if site_id in self.sites:
            self.sites[site_id].connected_sites = connected_sites

    def monitor_sites(self) -> Dict[str, SiteStatus]:
        """
        Monitor health of all sites
        
        Returns:
            Map of site_id -> status
        """
        site_statuses = {}
        healthy_count = 0

        for site_id, site in self.sites.items():
            if site.is_healthy():
                site_statuses[site_id] = SiteStatus.HEALTHY
                healthy_count += 1
            else:
                # Check if partitioned
                reachable = len(site.connected_sites)
                total_sites = len(self.sites)

                if reachable < total_sites / 2:
                    site.status = SiteStatus.PARTITIONED
                    site_statuses[site_id] = SiteStatus.PARTITIONED
                else:
                    site.status = SiteStatus.DEGRADED
                    site_statuses[site_id] = SiteStatus.DEGRADED

        # Update federation mode
        if healthy_count == len(self.sites):
            self.mode = FederationMode.NORMAL
        elif healthy_count >= len(self.sites) / 2:
            self.mode = FederationMode.DEGRADED
        else:
            self.mode = FederationMode.EMERGENCY

        return site_statuses

    def submit_federated_job(self, job: FederatedJob) -> str:
        """
        Submit job spanning multiple sites
        
        Args:
            job: Federated job to submit
            
        Returns:
            Job ID
        """
        # Verify sites are available
        available_sites = set(
            site_id for site_id, site in self.sites.items()
            if site.status == SiteStatus.HEALTHY
        )

        if not job.can_execute_on_sites(available_sites):
            raise RuntimeError(f"Insufficient sites for job {job.job_id}")

        self.jobs[job.job_id] = job
        job.status = "submitted"

        return job.job_id

    def synchronize_state(self, site_id: str, new_state_hash: str) -> bool:
        """
        Synchronize state for site (requires consensus)
        
        Args:
            site_id: Site identifier
            new_state_hash: New state hash
            
        Returns:
            True if synchronization successful
        """
        if site_id not in self.sites:
            return False

        # Update site state
        site = self.sites[site_id]
        site.update_state(new_state_hash)

        # Update global state
        self.global_state.update_site_state(site_id, new_state_hash)

        # Run consensus for strong consistency
        if self.consensus:
            proposal_id = self.consensus.submit_proposal(
                state_hash=self.global_state.compute_global_hash(),
                payload={
                    'operation': 'state_sync',
                    'site_id': site_id,
                    'new_state_hash': new_state_hash,
                }
            )

            if proposal_id:
                # Run consensus
                success = self.consensus.run_consensus_round(proposal_id)
                if success:
                    self.global_state.commit_updates()
                    return True

        return False

    def handle_partition(self, partitioned_sites: Set[str]) -> Dict[str, Any]:
        """
        Handle network partition
        
        Args:
            partitioned_sites: Set of partitioned site IDs
            
        Returns:
            Recovery plan
        """
        # Identify majority partition
        majority_size = len(self.sites) / 2 + 1
        connected_sites = set(self.sites.keys()) - partitioned_sites

        if len(connected_sites) >= majority_size:
            # Majority partition continues operation
            recovery_plan = {
                'strategy': 'majority_continues',
                'active_sites': list(connected_sites),
                'paused_sites': list(partitioned_sites),
                'action': 'continue_on_majority',
            }
        else:
            # No majority, pause all operations
            recovery_plan = {
                'strategy': 'pause_all',
                'active_sites': [],
                'paused_sites': list(self.sites.keys()),
                'action': 'wait_for_quorum',
            }
            self.mode = FederationMode.EMERGENCY

        return recovery_plan

    def migrate_job(self, job_id: str, from_site: str, to_site: str) -> bool:
        """
        Migrate job between sites
        
        Args:
            job_id: Job identifier
            from_site: Source site
            to_site: Destination site
            
        Returns:
            True if migration successful
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]

        # Verify destination site is healthy
        if to_site not in self.sites or self.sites[to_site].status != SiteStatus.HEALTHY:
            return False

        # Update job assignment
        if job.primary_site == from_site:
            job.primary_site = to_site
        elif from_site in job.replica_sites:
            job.replica_sites.remove(from_site)
            job.replica_sites.append(to_site)

        return True

    def get_federation_statistics(self) -> Dict[str, Any]:
        """
        Get federation statistics
        
        Returns:
            Dictionary of statistics
        """
        # Site statistics
        site_stats = {}
        for status in SiteStatus:
            count = sum(1 for s in self.sites.values() if s.status == status)
            site_stats[status.value] = count

        # Job statistics
        job_stats = {}
        for job in self.jobs.values():
            job_stats[job.status] = job_stats.get(job.status, 0) + 1

        # Consensus statistics
        consensus_stats = {}
        if self.consensus:
            consensus_stats = self.consensus.get_consensus_statistics()

        return {
            'federation_id': self.federation_id,
            'mode': self.mode.value,
            'total_sites': len(self.sites),
            'site_statistics': site_stats,
            'total_nodes': sum(s.num_nodes for s in self.sites.values()),
            'global_state_version': self.global_state.version,
            'global_state_hash': self.global_state.state_hash,
            'total_jobs': len(self.jobs),
            'job_statistics': job_stats,
            'active_operations': len(self.active_operations),
            'consensus': consensus_stats,
        }

    def verify_global_consistency(self) -> bool:
        """
        Verify global state consistency across all sites
        
        Returns:
            True if globally consistent
        """
        if not self.global_state.verify_consistency():
            return False

        # Verify each site state matches global state
        for site_id, expected_hash in self.global_state.site_states.items():
            if site_id in self.sites:
                actual_hash = self.sites[site_id].state_hash
                if actual_hash != expected_hash:
                    return False

        return True


class FederationBuilder:
    """
    Builder for standard federation configurations
    """

    @staticmethod
    def create_standard_federation(
        federation_id: str = "qratum_exascale_federation"
    ) -> FederationControlPlane:
        """
        Create standard 5-site federation
        
        Args:
            federation_id: Federation identifier
            
        Returns:
            FederationControlPlane instance
        """
        control_plane = FederationControlPlane(federation_id=federation_id)

        # Define standard sites
        sites_config = [
            ("us-east", "Virginia, USA"),
            ("us-west", "Oregon, USA"),
            ("eu", "Frankfurt, Germany"),
            ("asia-pacific", "Tokyo, Japan"),
            ("middle-east", "Dubai, UAE"),
        ]

        # Create sites
        for site_id, location in sites_config:
            site = Site(
                site_id=site_id,
                location=location,
                num_nodes=625,  # 625 nodes per site
            )

            # Full mesh connectivity (all sites can reach all sites)
            site.connected_sites = set(s[0] for s in sites_config if s[0] != site_id)

            control_plane.add_site(site)

        # Commit initial global state
        control_plane.global_state.commit_updates()

        return control_plane
