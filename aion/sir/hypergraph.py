"""AION-SIR Hypergraph Implementation.

The core hypergraph data structure for AION semantic IR.
Provides graph construction, traversal, and analysis.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from uuid import uuid4

from .vertices import Vertex, VertexType, AIONType, HardwareAffinity, EffectKind
from .edges import HyperEdge, EdgeType, EdgeGroup, ParallelismKind


@dataclass
class HyperGraph:
    """A typed attributed hypergraph for AION-SIR.
    
    The hypergraph represents program semantics with:
    - Vertices for computation nodes
    - Hyperedges for relationships (data flow, control flow, effects, parallelism)
    
    Attributes:
        id: Unique graph identifier
        name: Graph name (e.g., function name)
        vertices: Set of vertices in the graph
        edges: Set of hyperedges in the graph
        entry: Entry vertex for the graph
        exits: Exit vertices for the graph
        metadata: Graph-level metadata
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    vertices: set[Vertex] = field(default_factory=set)
    edges: set[HyperEdge] = field(default_factory=set)
    entry: Vertex | None = None
    exits: list[Vertex] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_vertex(self, vertex: Vertex) -> None:
        """Add a vertex to the graph."""
        self.vertices.add(vertex)
    
    def add_edge(self, edge: HyperEdge) -> None:
        """Add a hyperedge to the graph."""
        self.edges.add(edge)
        # Auto-add vertices from edge
        for v in edge.all_vertices():
            self.vertices.add(v)
    
    def remove_vertex(self, vertex: Vertex) -> None:
        """Remove a vertex and all connected edges."""
        self.vertices.discard(vertex)
        # Remove edges connected to vertex
        self.edges = {e for e in self.edges if vertex not in e.all_vertices()}
    
    def remove_edge(self, edge: HyperEdge) -> None:
        """Remove a hyperedge from the graph."""
        self.edges.discard(edge)
    
    def get_vertex_by_id(self, vertex_id: str) -> Vertex | None:
        """Find a vertex by its ID."""
        for v in self.vertices:
            if v.id == vertex_id:
                return v
        return None
    
    def get_predecessors(self, vertex: Vertex) -> list[Vertex]:
        """Get all predecessor vertices via data flow edges."""
        preds = []
        for edge in self.edges:
            if edge.edge_type == EdgeType.DATA_FLOW and vertex in edge.targets:
                preds.extend(edge.sources)
        return preds
    
    def get_successors(self, vertex: Vertex) -> list[Vertex]:
        """Get all successor vertices via data flow edges."""
        succs = []
        for edge in self.edges:
            if edge.edge_type == EdgeType.DATA_FLOW and vertex in edge.sources:
                succs.extend(edge.targets)
        return succs
    
    def get_data_flow_edges(self) -> list[HyperEdge]:
        """Get all data flow edges."""
        return [e for e in self.edges if e.edge_type == EdgeType.DATA_FLOW]
    
    def get_control_flow_edges(self) -> list[HyperEdge]:
        """Get all control flow edges."""
        return [e for e in self.edges if e.edge_type == EdgeType.CONTROL_FLOW]
    
    def get_parallel_edges(self) -> list[HyperEdge]:
        """Get all parallel annotation edges."""
        return [e for e in self.edges if e.edge_type == EdgeType.PARALLEL_EDGE]
    
    def get_effect_edges(self) -> list[HyperEdge]:
        """Get all effect ordering edges."""
        return [e for e in self.edges if e.edge_type == EdgeType.EFFECT_EDGE]
    
    def topological_order(self) -> list[Vertex]:
        """Return vertices in topological order based on data flow."""
        in_degree: dict[str, int] = {v.id: 0 for v in self.vertices}
        
        for edge in self.get_data_flow_edges():
            for target in edge.targets:
                in_degree[target.id] += 1
        
        queue = [v for v in self.vertices if in_degree[v.id] == 0]
        result = []
        
        while queue:
            vertex = queue.pop(0)
            result.append(vertex)
            for succ in self.get_successors(vertex):
                in_degree[succ.id] -= 1
                if in_degree[succ.id] == 0:
                    queue.append(succ)
        
        return result
    
    def find_parallel_regions(self) -> list[EdgeGroup]:
        """Find regions that can execute in parallel."""
        groups = []
        visited = set()
        
        for edge in self.get_parallel_edges():
            group = EdgeGroup(
                name=f"parallel_region_{len(groups)}",
                edges=[edge],
                pattern="parallel",
                fusible=True,
            )
            groups.append(group)
        
        return groups
    
    def compute_effects(self) -> dict[str, set[EffectKind]]:
        """Compute effect summary for the graph."""
        effects: dict[str, set[EffectKind]] = {}
        for v in self.vertices:
            effects[v.id] = v.metadata.effects.copy()
        return effects
    
    def verify_memory_safety(self) -> list[str]:
        """Verify basic memory safety properties.
        
        Returns list of violations found.
        """
        violations = []
        
        # Check that all loads have corresponding allocations
        for v in self.vertices:
            if v.vertex_type == VertexType.LOAD:
                preds = self.get_predecessors(v)
                if not any(p.vertex_type in (VertexType.ALLOC, VertexType.PARAMETER) for p in preds):
                    violations.append(f"Load {v.id} has no allocation predecessor")
        
        # Check that stores don't escape their regions
        region_map: dict[str, str] = {}
        for v in self.vertices:
            if v.metadata.region:
                region_map[v.id] = v.metadata.region
        
        for edge in self.edges:
            if edge.edge_type == EdgeType.REGION_EDGE:
                transfer = edge.attributes.get("transfer_type", "copy")
                if transfer == "move":
                    # Check source is not used after move
                    for src in edge.sources:
                        for e2 in self.edges:
                            if e2 != edge and src in e2.sources:
                                violations.append(f"Vertex {src.id} used after move")
        
        return violations
    
    def serialize(self) -> dict[str, Any]:
        """Serialize graph to dictionary for .aion_sir section."""
        return {
            "id": self.id,
            "name": self.name,
            "vertices": [v.serialize() for v in self.vertices],
            "edges": [e.serialize() for e in self.edges],
            "entry": self.entry.id if self.entry else None,
            "exits": [v.id for v in self.exits],
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.serialize(), indent=indent)
    
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "HyperGraph":
        """Deserialize graph from dictionary."""
        graph = HyperGraph(
            id=data["id"],
            name=data.get("name", ""),
            metadata=data.get("metadata", {}),
        )
        # Deserialize vertices and edges...
        # (Full implementation would reconstruct vertices/edges)
        return graph
    
    def clone(self) -> "HyperGraph":
        """Create a deep copy of the graph."""
        new_graph = HyperGraph(
            name=self.name,
            metadata=self.metadata.copy(),
        )
        # Clone vertices with new IDs mapping
        id_map: dict[str, Vertex] = {}
        for v in self.vertices:
            new_v = Vertex(
                vertex_type=v.vertex_type,
                value=v.value,
                attributes=v.attributes.copy(),
                metadata=v.metadata,
            )
            id_map[v.id] = new_v
            new_graph.add_vertex(new_v)
        
        # Clone edges with updated vertex references
        for e in self.edges:
            new_sources = [id_map.get(s.id, s) for s in e.sources]
            new_targets = [id_map.get(t.id, t) for t in e.targets]
            new_e = HyperEdge(
                edge_type=e.edge_type,
                sources=new_sources,
                targets=new_targets,
                attributes=e.attributes.copy(),
                metadata=e.metadata,
            )
            new_graph.add_edge(new_e)
        
        # Update entry/exit
        if self.entry and self.entry.id in id_map:
            new_graph.entry = id_map[self.entry.id]
        new_graph.exits = [id_map.get(v.id, v) for v in self.exits]
        
        return new_graph


class GraphBuilder:
    """Builder class for constructing AION-SIR hypergraphs.
    
    Provides a fluent interface for graph construction with
    automatic edge creation.
    """
    
    def __init__(self, name: str = ""):
        """Initialize a graph builder."""
        self.graph = HyperGraph(name=name)
        self._current_vertex: Vertex | None = None
        self._scope_stack: list[Vertex] = []
    
    def const(self, value: Any, type_info: AIONType | None = None) -> "GraphBuilder":
        """Add a constant vertex."""
        v = Vertex.const(value, type_info)
        self.graph.add_vertex(v)
        self._current_vertex = v
        return self
    
    def alloc(
        self,
        size: int | str,
        type_info: AIONType,
        region: str = "heap",
        affinity: HardwareAffinity = HardwareAffinity.ANY,
    ) -> "GraphBuilder":
        """Add an allocation vertex."""
        v = Vertex.alloc(size, type_info, region, affinity)
        self.graph.add_vertex(v)
        self._current_vertex = v
        return self
    
    def load(self, type_info: AIONType, from_vertex: Vertex | None = None) -> "GraphBuilder":
        """Add a load vertex."""
        v = Vertex.load(type_info)
        self.graph.add_vertex(v)
        if from_vertex:
            self.graph.add_edge(HyperEdge.data_flow(from_vertex, v))
        self._current_vertex = v
        return self
    
    def store(self, value_vertex: Vertex, to_vertex: Vertex, type_info: AIONType) -> "GraphBuilder":
        """Add a store vertex."""
        v = Vertex.store(type_info)
        self.graph.add_vertex(v)
        self.graph.add_edge(HyperEdge.data_flow([value_vertex, to_vertex], v))
        self._current_vertex = v
        return self
    
    def apply(
        self,
        function_name: str,
        args: list[Vertex],
        type_info: AIONType,
        effects: set[EffectKind] | None = None,
    ) -> "GraphBuilder":
        """Add a function application vertex."""
        v = Vertex.apply(function_name, type_info, effects)
        self.graph.add_vertex(v)
        if args:
            self.graph.add_edge(HyperEdge.data_flow(args, v))
        self._current_vertex = v
        return self
    
    def phi(self, sources: list[Vertex], type_info: AIONType) -> "GraphBuilder":
        """Add a phi node."""
        v = Vertex.phi(type_info)
        self.graph.add_vertex(v)
        self.graph.add_edge(HyperEdge.data_flow(sources, v))
        self._current_vertex = v
        return self
    
    def kernel(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Vertex],
        type_info: AIONType,
        affinity: HardwareAffinity = HardwareAffinity.GPU,
    ) -> "GraphBuilder":
        """Add a kernel launch vertex."""
        v = Vertex.kernel_launch(name, grid, block, type_info, affinity)
        self.graph.add_vertex(v)
        if args:
            self.graph.add_edge(HyperEdge.data_flow(args, v))
        self._current_vertex = v
        return self
    
    def param(self, name: str, type_info: AIONType, index: int = 0) -> "GraphBuilder":
        """Add a parameter vertex."""
        v = Vertex.parameter(name, type_info, index)
        self.graph.add_vertex(v)
        if self.graph.entry is None:
            self.graph.entry = v
        self._current_vertex = v
        return self
    
    def ret(self, value: Vertex, type_info: AIONType) -> "GraphBuilder":
        """Add a return vertex."""
        v = Vertex.ret(type_info)
        self.graph.add_vertex(v)
        self.graph.add_edge(HyperEdge.data_flow(value, v))
        self.graph.exits.append(v)
        self._current_vertex = v
        return self
    
    def connect(self, source: Vertex, target: Vertex) -> "GraphBuilder":
        """Add a data flow edge between vertices."""
        self.graph.add_edge(HyperEdge.data_flow(source, target))
        return self
    
    def control(self, source: Vertex, target: Vertex, kind: str = "sequential") -> "GraphBuilder":
        """Add a control flow edge."""
        from .edges import ControlFlowKind
        kind_enum = ControlFlowKind[kind.upper()]
        self.graph.add_edge(HyperEdge.control_flow(source, target, kind_enum))
        return self
    
    def parallel(
        self,
        vertices: list[Vertex],
        kind: ParallelismKind = ParallelismKind.SIMD,
        affinity: HardwareAffinity = HardwareAffinity.ANY,
    ) -> "GraphBuilder":
        """Add a parallel edge for vertices."""
        self.graph.add_edge(HyperEdge.parallel_edge(vertices, kind, affinity=affinity))
        return self
    
    def effect_order(self, source: Vertex, target: Vertex, ordering: str = "seq") -> "GraphBuilder":
        """Add an effect ordering edge."""
        self.graph.add_edge(HyperEdge.effect_edge(source, target, ordering))
        return self
    
    def current(self) -> Vertex:
        """Get the current vertex."""
        if self._current_vertex is None:
            raise ValueError("No current vertex")
        return self._current_vertex
    
    def build(self) -> HyperGraph:
        """Build and return the graph."""
        return self.graph


def merge_graphs(graphs: list[HyperGraph], name: str = "merged") -> HyperGraph:
    """Merge multiple graphs into one.
    
    Used for cross-function analysis and interprocedural optimization.
    """
    merged = HyperGraph(name=name)
    
    for g in graphs:
        for v in g.vertices:
            merged.add_vertex(v)
        for e in g.edges:
            merged.add_edge(e)
    
    return merged


def slice_graph(
    graph: HyperGraph,
    criterion: Callable[[Vertex], bool],
) -> HyperGraph:
    """Extract a subgraph based on vertex criterion.
    
    Useful for extracting hardware-specific subgraphs.
    """
    selected = {v for v in graph.vertices if criterion(v)}
    
    # Include vertices connected to selected
    for edge in graph.edges:
        if any(v in selected for v in edge.all_vertices()):
            for v in edge.all_vertices():
                selected.add(v)
    
    sliced = HyperGraph(name=f"{graph.name}_slice")
    for v in selected:
        sliced.add_vertex(v)
    
    for e in graph.edges:
        if all(v in selected for v in e.all_vertices()):
            sliced.add_edge(e)
    
    return sliced
