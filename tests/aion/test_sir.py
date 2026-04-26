"""Unit tests for AION-SIR hypergraph implementation."""

from __future__ import annotations

from aion.sir.edges import (
    ControlFlowKind,
    EdgeType,
    HyperEdge,
    ParallelismKind,
)
from aion.sir.hypergraph import GraphBuilder, HyperGraph
from aion.sir.vertices import (
    AIONType,
    EffectKind,
    HardwareAffinity,
    Provenance,
    Vertex,
    VertexType,
)


class TestVertex:
    """Tests for Vertex class."""

    def test_const_vertex(self) -> None:
        """Test constant vertex creation."""
        v = Vertex.const(42, AIONType.int())

        assert v.vertex_type == VertexType.CONST
        assert v.value == 42
        assert v.metadata.effects == {EffectKind.PURE}

    def test_alloc_vertex(self) -> None:
        """Test allocation vertex creation."""
        v = Vertex.alloc(
            size=64,
            type_info=AIONType.array(AIONType.float(), 8),
            region="heap",
            affinity=HardwareAffinity.GPU,
        )

        assert v.vertex_type == VertexType.ALLOC
        assert v.attributes["size"] == 64
        assert v.attributes["region"] == "heap"
        assert v.metadata.hardware_affinity == HardwareAffinity.GPU
        assert EffectKind.ALLOC in v.metadata.effects

    def test_kernel_launch_vertex(self) -> None:
        """Test kernel launch vertex creation."""
        v = Vertex.kernel_launch(
            kernel_name="matmul",
            grid_dim=(16, 16, 1),
            block_dim=(32, 32, 1),
            type_info=AIONType(kind="fn"),
            affinity=HardwareAffinity.GPU,
        )

        assert v.vertex_type == VertexType.KERNEL_LAUNCH
        assert v.attributes["kernel"] == "matmul"
        assert v.metadata.parallelism["grid_dim"] == (16, 16, 1)
        assert v.metadata.parallelism["block_dim"] == (32, 32, 1)

    def test_vertex_serialization(self) -> None:
        """Test vertex serialization."""
        v = Vertex.const(3.14, AIONType.float())
        data = v.serialize()

        assert data["type"] == "CONST"
        assert data["value"] == 3.14
        assert "metadata" in data

    def test_vertex_with_provenance(self) -> None:
        """Test vertex with provenance."""
        prov = Provenance(
            source_language="C",
            source_file="test.c",
            source_line=42,
            original_name="x",
        )
        v = Vertex.const(0, AIONType.int(), prov)

        assert v.metadata.provenance == prov
        assert v.metadata.provenance.source_language == "C"

    def test_vertex_with_affinity(self) -> None:
        """Test vertex affinity modification."""
        v = Vertex.const(0)
        v2 = v.with_affinity(HardwareAffinity.FPGA)

        assert v2.metadata.hardware_affinity == HardwareAffinity.FPGA
        assert v.metadata.hardware_affinity == HardwareAffinity.ANY  # Original unchanged


class TestHyperEdge:
    """Tests for HyperEdge class."""

    def test_data_flow_edge(self) -> None:
        """Test data flow edge creation."""
        src = Vertex.const(1)
        dst = Vertex.const(2)
        edge = HyperEdge.data_flow(src, dst)

        assert edge.edge_type == EdgeType.DATA_FLOW
        assert src in edge.sources
        assert dst in edge.targets

    def test_control_flow_edge(self) -> None:
        """Test control flow edge creation."""
        src = Vertex.const(0)
        dst = Vertex.const(1)
        edge = HyperEdge.control_flow(src, dst, ControlFlowKind.BRANCH)

        assert edge.edge_type == EdgeType.CONTROL_FLOW
        assert edge.attributes["kind"] == "BRANCH"

    def test_parallel_edge(self) -> None:
        """Test parallel edge creation."""
        vertices = [Vertex.const(i) for i in range(4)]
        edge = HyperEdge.parallel_edge(
            vertices,
            kind=ParallelismKind.SIMT,
            warp_size=32,
            affinity=HardwareAffinity.GPU,
        )

        assert edge.edge_type == EdgeType.PARALLEL_EDGE
        assert edge.attributes["kind"] == "SIMT"
        assert edge.attributes["warp_size"] == 32
        assert len(edge.targets) == 4

    def test_effect_edge(self) -> None:
        """Test effect ordering edge."""
        src = Vertex.store(AIONType.int(), "heap")
        dst = Vertex.load(AIONType.int(), "heap")
        edge = HyperEdge.effect_edge(src, dst, "seq")

        assert edge.edge_type == EdgeType.EFFECT_EDGE
        assert edge.attributes["ordering"] == "seq"

    def test_region_edge(self) -> None:
        """Test region transfer edge."""
        src = Vertex.alloc(8, AIONType.int(), "stack")
        dst = Vertex.alloc(8, AIONType.int(), "heap")
        edge = HyperEdge.region_edge(src, dst, "stack", "heap", "move")

        assert edge.edge_type == EdgeType.REGION_EDGE
        assert edge.attributes["transfer_type"] == "move"


class TestHyperGraph:
    """Tests for HyperGraph class."""

    def test_empty_graph(self) -> None:
        """Test empty graph creation."""
        g = HyperGraph(name="test")

        assert g.name == "test"
        assert len(g.vertices) == 0
        assert len(g.edges) == 0

    def test_add_vertex(self) -> None:
        """Test adding vertices."""
        g = HyperGraph()
        v = Vertex.const(42)
        g.add_vertex(v)

        assert v in g.vertices
        assert len(g.vertices) == 1

    def test_add_edge(self) -> None:
        """Test adding edges auto-adds vertices."""
        g = HyperGraph()
        v1 = Vertex.const(1)
        v2 = Vertex.const(2)
        edge = HyperEdge.data_flow(v1, v2)

        g.add_edge(edge)

        assert v1 in g.vertices
        assert v2 in g.vertices
        assert edge in g.edges

    def test_topological_order(self) -> None:
        """Test topological sorting."""
        g = HyperGraph()
        v1 = Vertex.const(1)
        v2 = Vertex.const(2)
        v3 = Vertex.apply("add", AIONType.int())

        g.add_vertex(v1)
        g.add_vertex(v2)
        g.add_vertex(v3)
        g.add_edge(HyperEdge.data_flow(v1, v3))
        g.add_edge(HyperEdge.data_flow(v2, v3))

        order = g.topological_order()

        # v3 should come after v1 and v2
        assert order.index(v3) > order.index(v1)
        assert order.index(v3) > order.index(v2)

    def test_get_predecessors(self) -> None:
        """Test getting predecessor vertices."""
        g = HyperGraph()
        v1 = Vertex.const(1)
        v2 = Vertex.const(2)
        g.add_edge(HyperEdge.data_flow(v1, v2))

        preds = g.get_predecessors(v2)

        assert v1 in preds

    def test_get_successors(self) -> None:
        """Test getting successor vertices."""
        g = HyperGraph()
        v1 = Vertex.const(1)
        v2 = Vertex.const(2)
        g.add_edge(HyperEdge.data_flow(v1, v2))

        succs = g.get_successors(v1)

        assert v2 in succs

    def test_clone_graph(self) -> None:
        """Test graph cloning."""
        g = HyperGraph(name="original")
        v1 = Vertex.const(1)
        v2 = Vertex.const(2)
        g.add_edge(HyperEdge.data_flow(v1, v2))

        clone = g.clone()

        assert clone.name == g.name
        assert len(clone.vertices) == len(g.vertices)
        assert len(clone.edges) == len(g.edges)
        # But different objects
        assert clone.vertices != g.vertices

    def test_serialization(self) -> None:
        """Test graph serialization."""
        g = HyperGraph(name="test")
        g.add_vertex(Vertex.const(42))

        data = g.serialize()

        assert data["name"] == "test"
        assert len(data["vertices"]) == 1

    def test_memory_safety_check(self) -> None:
        """Test memory safety verification."""
        g = HyperGraph()

        # Valid: alloc followed by load
        alloc = Vertex.alloc(8, AIONType.int())
        load = Vertex.load(AIONType.int())
        g.add_edge(HyperEdge.data_flow(alloc, load))

        violations = g.verify_memory_safety()

        assert len(violations) == 0


class TestGraphBuilder:
    """Tests for GraphBuilder fluent API."""

    def test_const_builder(self) -> None:
        """Test building with constants."""
        builder = GraphBuilder(name="test")
        builder.const(42, AIONType.int())
        graph = builder.build()

        assert len(graph.vertices) == 1

    def test_apply_builder(self) -> None:
        """Test building with function application."""
        builder = GraphBuilder()

        builder.const(1)
        v1 = builder.current()
        builder.const(2)
        v2 = builder.current()

        builder.apply("add", [v1, v2], AIONType.int())
        graph = builder.build()

        assert len(graph.vertices) == 3
        assert len(graph.get_data_flow_edges()) == 1

    def test_kernel_builder(self) -> None:
        """Test building kernel launches."""
        builder = GraphBuilder()

        builder.alloc(
            64, AIONType.tensor(AIONType.float(), [8, 8]), "gpu_global", HardwareAffinity.GPU
        )
        arr = builder.current()

        builder.kernel(
            "process",
            grid=(8, 8, 1),
            block=(8, 8, 1),
            args=[arr],
            type_info=AIONType(kind="unit"),
            affinity=HardwareAffinity.GPU,
        )

        graph = builder.build()
        kernels = [v for v in graph.vertices if v.vertex_type == VertexType.KERNEL_LAUNCH]

        assert len(kernels) == 1

    def test_parallel_builder(self) -> None:
        """Test building parallel regions."""
        builder = GraphBuilder()

        v1 = Vertex.const(1)
        v2 = Vertex.const(2)
        v3 = Vertex.const(3)

        for v in [v1, v2, v3]:
            builder.graph.add_vertex(v)

        builder.parallel([v1, v2, v3], ParallelismKind.SIMD, HardwareAffinity.CPU)
        graph = builder.build()

        parallel_edges = graph.get_parallel_edges()
        assert len(parallel_edges) == 1


class TestAIONType:
    """Tests for AIONType."""

    def test_int_type(self) -> None:
        """Test integer type creation."""
        t = AIONType.int(32)

        assert t.kind == "int"
        assert t.size == 32

    def test_float_type(self) -> None:
        """Test float type creation."""
        t = AIONType.float(64)

        assert t.kind == "float"
        assert t.size == 64

    def test_ptr_type(self) -> None:
        """Test pointer type creation."""
        t = AIONType.ptr(AIONType.int(), "heap")

        assert t.kind == "ptr"
        assert len(t.params) == 1
        assert "region=heap" in t.refinement

    def test_array_type(self) -> None:
        """Test array type creation."""
        t = AIONType.array(AIONType.float(), 100)

        assert t.kind == "array"
        assert "length=100" in t.refinement

    def test_tensor_type(self) -> None:
        """Test tensor type creation."""
        t = AIONType.tensor(AIONType.float(32), [32, 32, 3])

        assert t.kind == "tensor"
        assert "shape=[32,32,3]" in t.refinement

    def test_fn_type(self) -> None:
        """Test function type creation."""
        t = AIONType.fn([AIONType.int(), AIONType.int()], AIONType.int(), {EffectKind.PURE})

        assert t.kind == "fn"
        assert len(t.params) == 3  # 2 params + return
        assert EffectKind.PURE in t.effects
