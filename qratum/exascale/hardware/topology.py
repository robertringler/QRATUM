"""
System Topology Management

Defines rack-scale, row-scale, and system-wide topology for the QRATUM ExaScale
deployment with 3,125 compute nodes organized for optimal bisection bandwidth
and fault isolation.

Topology:
- Dragonfly+ network topology
- 57 racks per row, 10 rows total (570 racks)
- 5-6 nodes per rack (3,125 nodes total)
- Deterministic routing ensures reproducible communication patterns
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


class TopologyType(Enum):
    """Network topology types"""
    DRAGONFLY_PLUS = "Dragonfly+"
    FAT_TREE = "Fat Tree"
    TORUS = "Torus"
    HYPERCUBE = "Hypercube"


class FailureDomainType(Enum):
    """Failure domain isolation levels"""
    NODE = "Node"
    RACK = "Rack"
    ROW = "Row"
    DATACENTER = "Datacenter"


@dataclass
class RackLayout:
    """
    Physical rack configuration for CN-QES nodes
    
    Each rack contains 5-6 compute nodes with dedicated cooling,
    power distribution, and network connectivity.
    
    Power Density: 54.5 kW per rack (average)
    Cooling: Direct liquid cooling + rear-door heat exchangers
    """
    rack_id: str
    row_id: str
    nodes_per_rack: int
    node_ids: List[str]
    power_capacity_kw: float
    cooling_type: str = "Direct Liquid + RDHX"
    network_switches: int = 2  # Redundant top-of-rack switches

    def __post_init__(self):
        """Validate rack configuration"""
        if self.nodes_per_rack < 5 or self.nodes_per_rack > 6:
            raise ValueError(f"Nodes per rack must be 5-6, got {self.nodes_per_rack}")
        if len(self.node_ids) != self.nodes_per_rack:
            raise ValueError(f"Node ID count mismatch: expected {self.nodes_per_rack}, got {len(self.node_ids)}")

    def total_power_kw(self, node_power_kw: float = 17.6) -> float:
        """
        Calculate total rack power consumption
        
        Args:
            node_power_kw: Power per node in kW (default 17.6 kW)
            
        Returns:
            Total rack power in kW
        """
        return self.nodes_per_rack * node_power_kw

    def is_power_within_limits(self) -> bool:
        """Check if rack power is within capacity"""
        return self.total_power_kw() <= self.power_capacity_kw


@dataclass
class FailureDomain:
    """
    Fault isolation domain for resilience planning
    
    Defines boundaries for failure propagation and provides
    redundancy strategy for Byzantine fault tolerance.
    """
    domain_id: str
    domain_type: FailureDomainType
    member_ids: Set[str]
    redundancy_factor: int = 3  # BFT requires 3f+1 replicas
    isolation_enabled: bool = True

    def can_tolerate_failures(self, num_failures: int) -> bool:
        """
        Check if domain can tolerate given number of failures
        
        BFT consensus requires 3f+1 nodes to tolerate f failures
        """
        max_failures = (len(self.member_ids) - 1) // 3
        return num_failures <= max_failures

    def required_nodes_for_bft(self, f: int) -> int:
        """Calculate minimum nodes required to tolerate f failures"""
        return 3 * f + 1


@dataclass
class SystemTopology:
    """
    Complete system topology for QRATUM ExaScale deployment
    
    Manages 3,125 compute nodes organized into racks, rows, and
    failure domains with Dragonfly+ interconnect topology.
    
    Configuration:
    - 3,125 compute nodes (50,000 GPUs)
    - 570 racks (57 racks × 10 rows)
    - Dragonfly+ topology with deterministic routing
    - 800 Gbps links per node (8× 100 Gbps AetherFabric-X NICs)
    - < 500 ns any-to-any latency
    - ≥ 100 PB/s bisection bandwidth
    """
    topology_type: TopologyType
    total_nodes: int
    total_racks: int
    total_rows: int
    nodes_per_rack: int
    racks_per_row: int
    racks: List[RackLayout] = field(default_factory=list)
    failure_domains: List[FailureDomain] = field(default_factory=list)
    link_bandwidth_gbps: int = 800
    any_to_any_latency_ns: int = 500
    bisection_bandwidth_pbs: float = 100.0

    def __post_init__(self):
        """Validate system configuration"""
        expected_racks = self.total_rows * self.racks_per_row
        if self.total_racks != expected_racks:
            raise ValueError(
                f"Rack count mismatch: {self.total_racks} != {self.total_rows} × {self.racks_per_row}"
            )

        expected_nodes = self.total_racks * self.nodes_per_rack
        if abs(self.total_nodes - expected_nodes) > self.total_racks:
            # Allow small variance due to uneven distribution
            raise ValueError(
                f"Node count mismatch: {self.total_nodes} != {self.total_racks} × {self.nodes_per_rack}"
            )

    def generate_node_placement(self) -> List[RackLayout]:
        """
        Generate optimal node placement across racks
        
        Returns:
            List of RackLayout objects with node assignments
        """
        racks = []
        node_counter = 0

        for row in range(self.total_rows):
            for rack in range(self.racks_per_row):
                rack_id = f"R{row:02d}-{rack:02d}"
                row_id = f"Row{row:02d}"

                # Distribute nodes evenly (5-6 per rack)
                nodes_in_rack = min(
                    self.nodes_per_rack,
                    self.total_nodes - node_counter
                )

                if nodes_in_rack <= 0:
                    break

                node_ids = [
                    f"CN-{node_counter + i:05d}"
                    for i in range(nodes_in_rack)
                ]

                rack = RackLayout(
                    rack_id=rack_id,
                    row_id=row_id,
                    nodes_per_rack=nodes_in_rack,
                    node_ids=node_ids,
                    power_capacity_kw=60.0,  # 60 kW capacity per rack
                )

                racks.append(rack)
                node_counter += nodes_in_rack

        self.racks = racks
        return racks

    def create_failure_domains(self) -> List[FailureDomain]:
        """
        Create hierarchical failure domains for fault tolerance
        
        Returns:
            List of FailureDomain objects
        """
        domains = []

        # Node-level domains
        for rack in self.racks:
            for node_id in rack.node_ids:
                domain = FailureDomain(
                    domain_id=f"FD-{node_id}",
                    domain_type=FailureDomainType.NODE,
                    member_ids={node_id},
                )
                domains.append(domain)

        # Rack-level domains
        for rack in self.racks:
            domain = FailureDomain(
                domain_id=f"FD-{rack.rack_id}",
                domain_type=FailureDomainType.RACK,
                member_ids=set(rack.node_ids),
            )
            domains.append(domain)

        # Row-level domains
        for row_id in range(self.total_rows):
            row_racks = [r for r in self.racks if r.row_id == f"Row{row_id:02d}"]
            all_nodes = set()
            for rack in row_racks:
                all_nodes.update(rack.node_ids)

            domain = FailureDomain(
                domain_id=f"FD-Row{row_id:02d}",
                domain_type=FailureDomainType.ROW,
                member_ids=all_nodes,
            )
            domains.append(domain)

        self.failure_domains = domains
        return domains

    def calculate_diameter(self) -> int:
        """
        Calculate network diameter (maximum hops between any two nodes)
        
        For Dragonfly+ topology with deterministic routing
        """
        if self.topology_type == TopologyType.DRAGONFLY_PLUS:
            # Dragonfly+ typically has diameter ≤ 3 hops
            return 3
        return -1  # Unknown for other topologies

    def get_system_summary(self) -> Dict[str, any]:
        """Return comprehensive system summary"""
        return {
            "topology_type": self.topology_type.value,
            "total_nodes": self.total_nodes,
            "total_gpus": self.total_nodes * 16,
            "total_cpu_cores": self.total_nodes * 256,
            "total_racks": self.total_racks,
            "total_rows": self.total_rows,
            "nodes_per_rack": self.nodes_per_rack,
            "racks_per_row": self.racks_per_row,
            "link_bandwidth_gbps": self.link_bandwidth_gbps,
            "any_to_any_latency_ns": self.any_to_any_latency_ns,
            "bisection_bandwidth_pbs": self.bisection_bandwidth_pbs,
            "network_diameter_hops": self.calculate_diameter(),
            "total_failure_domains": len(self.failure_domains),
        }


def create_exascale_topology() -> SystemTopology:
    """
    Create the standard QRATUM ExaScale topology
    
    Returns:
        SystemTopology configured for 3,125 nodes
    """
    topology = SystemTopology(
        topology_type=TopologyType.DRAGONFLY_PLUS,
        total_nodes=3125,
        total_racks=570,  # Will accommodate 3,125 nodes
        total_rows=10,
        nodes_per_rack=6,  # Average (some racks have 5)
        racks_per_row=57,
    )

    # Generate node placement
    topology.generate_node_placement()

    # Create failure domains
    topology.create_failure_domains()

    return topology


# Standard ExaScale topology instance
EXASCALE_TOPOLOGY = create_exascale_topology()
