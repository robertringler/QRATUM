"""AION-SIR Hyperedge Definitions.

Defines the hyperedge types for the AION semantic hypergraph:
- DataFlow: Data dependency edges
- ControlFlow: Control flow edges
- EffectEdge: Side effect ordering edges
- ParallelEdge: Parallelism annotations

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import uuid4

from .vertices import HardwareAffinity, Vertex


class EdgeType(Enum):
    """Enumeration of hyperedge types in AION-SIR."""

    DATA_FLOW = auto()
    CONTROL_FLOW = auto()
    EFFECT_EDGE = auto()
    PARALLEL_EDGE = auto()
    MEMORY_EDGE = auto()
    REGION_EDGE = auto()


class ParallelismKind(Enum):
    """Types of parallelism for parallel edges."""

    SIMD = auto()
    SIMT = auto()  # GPU warps
    THREAD_LEVEL = auto()
    TASK_LEVEL = auto()
    PIPELINE = auto()
    DATAFLOW = auto()


class ControlFlowKind(Enum):
    """Types of control flow edges."""

    SEQUENTIAL = auto()
    BRANCH = auto()
    LOOP_ENTRY = auto()
    LOOP_BACK = auto()
    LOOP_EXIT = auto()
    CALL = auto()
    RETURN = auto()
    EXCEPTION = auto()


@dataclass
class EdgeMetadata:
    """Metadata attached to hyperedges.

    Attributes:
        weight: Edge weight for scheduling
        latency: Estimated latency in cycles
        bandwidth: Required memory bandwidth
        hardware_affinity: Target hardware preference
        critical_path: Whether edge is on critical path
    """

    weight: float = 1.0
    latency: int = 0
    bandwidth: int = 0
    hardware_affinity: HardwareAffinity = HardwareAffinity.ANY
    critical_path: bool = False


@dataclass
class HyperEdge:
    """A hyperedge in the AION-SIR hypergraph.

    Hyperedges connect multiple vertices and represent:
    - DataFlow: Data dependencies between operations
    - ControlFlow: Execution order constraints
    - EffectEdge: Side effect ordering
    - ParallelEdge: Parallelism annotations

    Attributes:
        id: Unique edge identifier
        edge_type: Type of the hyperedge
        sources: Source vertices
        targets: Target vertices
        attributes: Additional edge attributes
        metadata: Edge metadata
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    edge_type: EdgeType = EdgeType.DATA_FLOW
    sources: list[Vertex] = field(default_factory=list)
    targets: list[Vertex] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: EdgeMetadata = field(default_factory=EdgeMetadata)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperEdge):
            return False
        return self.id == other.id

    @staticmethod
    def data_flow(
        source: Vertex | list[Vertex],
        target: Vertex | list[Vertex],
        port: int = 0,
        affinity: HardwareAffinity = HardwareAffinity.ANY,
    ) -> HyperEdge:
        """Create a data flow hyperedge."""
        sources = [source] if isinstance(source, Vertex) else source
        targets = [target] if isinstance(target, Vertex) else target
        return HyperEdge(
            edge_type=EdgeType.DATA_FLOW,
            sources=sources,
            targets=targets,
            attributes={"port": port},
            metadata=EdgeMetadata(hardware_affinity=affinity),
        )

    @staticmethod
    def control_flow(
        source: Vertex | list[Vertex],
        target: Vertex | list[Vertex],
        kind: ControlFlowKind = ControlFlowKind.SEQUENTIAL,
        condition: str | None = None,
    ) -> HyperEdge:
        """Create a control flow hyperedge."""
        sources = [source] if isinstance(source, Vertex) else source
        targets = [target] if isinstance(target, Vertex) else target
        return HyperEdge(
            edge_type=EdgeType.CONTROL_FLOW,
            sources=sources,
            targets=targets,
            attributes={"kind": kind.name, "condition": condition},
        )

    @staticmethod
    def effect_edge(
        source: Vertex | list[Vertex],
        target: Vertex | list[Vertex],
        effect_ordering: str = "seq",
    ) -> HyperEdge:
        """Create an effect ordering hyperedge.

        Args:
            source: Source vertex or vertices
            target: Target vertex or vertices
            effect_ordering: Ordering constraint ("seq", "par", "atomic")
        """
        sources = [source] if isinstance(source, Vertex) else source
        targets = [target] if isinstance(target, Vertex) else target
        return HyperEdge(
            edge_type=EdgeType.EFFECT_EDGE,
            sources=sources,
            targets=targets,
            attributes={"ordering": effect_ordering},
        )

    @staticmethod
    def parallel_edge(
        vertices: list[Vertex],
        kind: ParallelismKind,
        simd_width: int = 1,
        num_threads: int = 1,
        warp_size: int = 32,
        affinity: HardwareAffinity = HardwareAffinity.ANY,
    ) -> HyperEdge:
        """Create a parallelism annotation hyperedge.

        Args:
            vertices: Vertices that can execute in parallel
            kind: Type of parallelism
            simd_width: SIMD vector width
            num_threads: Number of threads
            warp_size: GPU warp size
            affinity: Target hardware
        """
        return HyperEdge(
            edge_type=EdgeType.PARALLEL_EDGE,
            sources=[],
            targets=vertices,
            attributes={
                "kind": kind.name,
                "simd_width": simd_width,
                "num_threads": num_threads,
                "warp_size": warp_size,
            },
            metadata=EdgeMetadata(hardware_affinity=affinity),
        )

    @staticmethod
    def memory_edge(
        source: Vertex,
        target: Vertex,
        access_type: str = "read",  # "read", "write", "atomic"
        region: str = "heap",
    ) -> HyperEdge:
        """Create a memory access edge."""
        return HyperEdge(
            edge_type=EdgeType.MEMORY_EDGE,
            sources=[source],
            targets=[target],
            attributes={"access_type": access_type, "region": region},
        )

    @staticmethod
    def region_edge(
        source: Vertex,
        target: Vertex,
        source_region: str,
        target_region: str,
        transfer_type: str = "copy",  # "copy", "move", "borrow"
    ) -> HyperEdge:
        """Create a region transfer edge for memory management."""
        return HyperEdge(
            edge_type=EdgeType.REGION_EDGE,
            sources=[source],
            targets=[target],
            attributes={
                "source_region": source_region,
                "target_region": target_region,
                "transfer_type": transfer_type,
            },
        )

    def all_vertices(self) -> list[Vertex]:
        """Return all vertices connected by this edge."""
        return self.sources + self.targets

    def is_parallel(self) -> bool:
        """Check if this edge represents parallel execution."""
        return self.edge_type == EdgeType.PARALLEL_EDGE

    def is_data_flow(self) -> bool:
        """Check if this is a data flow edge."""
        return self.edge_type == EdgeType.DATA_FLOW

    def is_control_flow(self) -> bool:
        """Check if this is a control flow edge."""
        return self.edge_type == EdgeType.CONTROL_FLOW

    def serialize(self) -> dict[str, Any]:
        """Serialize edge to dictionary for .aion_sir section."""
        return {
            "id": self.id,
            "type": self.edge_type.name,
            "sources": [v.id for v in self.sources],
            "targets": [v.id for v in self.targets],
            "attributes": self.attributes,
            "metadata": {
                "weight": self.metadata.weight,
                "latency": self.metadata.latency,
                "bandwidth": self.metadata.bandwidth,
                "hardware_affinity": self.metadata.hardware_affinity.name,
                "critical_path": self.metadata.critical_path,
            },
        }


@dataclass
class EdgeGroup:
    """A group of related hyperedges for fusion analysis.

    Used to identify patterns like:
    - PythonCall → RustFFI → CUDA pipeline
    - Sequential memory operations for coalescing
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    edges: list[HyperEdge] = field(default_factory=list)
    pattern: str = ""  # Pattern identifier for fusion
    fusible: bool = False

    def add_edge(self, edge: HyperEdge) -> None:
        """Add an edge to the group."""
        self.edges.append(edge)

    def vertices(self) -> set[Vertex]:
        """Get all vertices in the edge group."""
        result = set()
        for edge in self.edges:
            result.update(edge.all_vertices())
        return result

    def can_fuse_with(self, other: EdgeGroup) -> bool:
        """Check if this group can be fused with another."""
        if not self.fusible or not other.fusible:
            return False
        # Check if groups share vertices (potential fusion point)
        return bool(self.vertices() & other.vertices())
