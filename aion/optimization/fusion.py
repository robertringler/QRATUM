"""AION Kernel Fusion.

Cross-language kernel fusion for optimizing polyglot pipelines:
- Pattern detection (PythonCall → RustFFI → CUDA)
- Proof-preserving optimizations
- Hardware affinity-aware fusion

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from ..proof.synthesis import synthesize_proofs_for_rewrite
from ..proof.verifier import ProofTerm
from ..sir.edges import HyperEdge
from ..sir.hypergraph import HyperGraph
from ..sir.vertices import EffectKind, HardwareAffinity, Vertex, VertexType


class FusionPatternKind(Enum):
    """Types of fusion patterns."""

    PYTHON_RUST_FFI = auto()
    RUST_CUDA = auto()
    PYTHON_RUST_CUDA = auto()
    CPU_GPU_TRANSFER = auto()
    MEMORY_COALESCING = auto()
    LOOP_FUSION = auto()
    KERNEL_FUSION = auto()
    DATAFLOW_FUSION = auto()
    PIPELINE_FUSION = auto()


@dataclass
class FusionPattern:
    """A detected fusion pattern in the hypergraph.

    Attributes:
        kind: Type of fusion pattern
        vertices: Vertices involved in the pattern
        edges: Edges involved in the pattern
        estimated_speedup: Estimated speedup from fusion
        hardware_target: Target hardware for fused kernel
        constraints: Constraints that must be satisfied
    """

    kind: FusionPatternKind
    vertices: list[Vertex] = field(default_factory=list)
    edges: list[HyperEdge] = field(default_factory=list)
    estimated_speedup: float = 1.0
    hardware_target: HardwareAffinity = HardwareAffinity.ANY
    constraints: list[str] = field(default_factory=list)

    def can_fuse(self) -> bool:
        """Check if fusion is legal."""
        # Check for conflicting effects
        all_effects: set[EffectKind] = set()
        for v in self.vertices:
            all_effects.update(v.metadata.effects)

        # Can't fuse if there are conflicting writes
        write_regions: set[str | None] = set()
        for v in self.vertices:
            if EffectKind.WRITE in v.metadata.effects:
                write_regions.add(v.metadata.region)

        if len(write_regions) > 1 and None not in write_regions:
            return False

        return True


@dataclass
class FusionResult:
    """Result of kernel fusion."""

    success: bool
    fused_graph: HyperGraph | None = None
    fused_vertices: list[Vertex] = field(default_factory=list)
    removed_vertices: list[Vertex] = field(default_factory=list)
    proofs: list[ProofTerm] = field(default_factory=list)
    speedup_estimate: float = 1.0


class KernelFusion:
    """Kernel fusion optimizer.

    Fuses multiple operations into single kernels for:
    - Reduced memory traffic
    - Better cache utilization
    - Fewer kernel launches
    """

    def __init__(self) -> None:
        """Initialize kernel fusion optimizer."""
        self.patterns: list[FusionPattern] = []
        self.fusion_rules: dict[FusionPatternKind, callable] = {
            FusionPatternKind.PYTHON_RUST_FFI: self._fuse_python_rust,
            FusionPatternKind.RUST_CUDA: self._fuse_rust_cuda,
            FusionPatternKind.PYTHON_RUST_CUDA: self._fuse_python_rust_cuda,
            FusionPatternKind.CPU_GPU_TRANSFER: self._fuse_cpu_gpu_transfer,
            FusionPatternKind.MEMORY_COALESCING: self._fuse_memory_coalescing,
            FusionPatternKind.LOOP_FUSION: self._fuse_loops,
            FusionPatternKind.KERNEL_FUSION: self._fuse_kernels,
            FusionPatternKind.DATAFLOW_FUSION: self._fuse_dataflow,
            FusionPatternKind.PIPELINE_FUSION: self._fuse_pipeline,
        }

    def optimize(
        self,
        graph: HyperGraph,
        proofs: list[ProofTerm] | None = None,
    ) -> FusionResult:
        """Apply kernel fusion optimizations.

        Args:
            graph: Input hypergraph
            proofs: Existing proofs for the graph

        Returns:
            FusionResult with optimized graph
        """
        # Detect patterns
        self.patterns = detect_fusion_patterns(graph)

        if not self.patterns:
            return FusionResult(success=True, fused_graph=graph, proofs=proofs or [])

        # Clone graph for modification
        optimized = graph.clone()
        all_proofs = proofs.copy() if proofs else []
        removed_vertices: list[Vertex] = []
        fused_vertices: list[Vertex] = []
        total_speedup = 1.0

        # Apply fusion for each pattern
        for pattern in self.patterns:
            if not pattern.can_fuse():
                continue

            fusion_rule = self.fusion_rules.get(pattern.kind)
            if fusion_rule:
                result = fusion_rule(optimized, pattern)
                if result.success:
                    optimized = result.fused_graph or optimized
                    removed_vertices.extend(result.removed_vertices)
                    fused_vertices.extend(result.fused_vertices)
                    total_speedup *= result.speedup_estimate

                    # Synthesize new proofs
                    if proofs:
                        new_proofs = synthesize_proofs_for_rewrite(graph, optimized, all_proofs)
                        all_proofs = new_proofs

        return FusionResult(
            success=True,
            fused_graph=optimized,
            fused_vertices=fused_vertices,
            removed_vertices=removed_vertices,
            proofs=all_proofs,
            speedup_estimate=total_speedup,
        )

    def _fuse_python_rust(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Fuse Python → Rust FFI calls."""
        # Find Python call vertices followed by Rust FFI
        python_calls = [
            v
            for v in pattern.vertices
            if v.metadata.provenance and v.metadata.provenance.source_language == "Python"
        ]
        rust_calls = [
            v
            for v in pattern.vertices
            if v.metadata.provenance and v.metadata.provenance.source_language == "Rust"
        ]

        if not python_calls or not rust_calls:
            return FusionResult(success=False)

        # Create fused vertex
        from ..sir.vertices import AIONType, Provenance

        fused = Vertex.apply(
            function_name="fused_python_rust",
            type_info=AIONType(kind="fn"),
            effects={EffectKind.IO},
            provenance=Provenance(
                source_language="fused",
                transformation_chain=["python_rust_fusion"],
            ),
        )
        fused.metadata.hardware_affinity = HardwareAffinity.CPU

        # Update graph
        graph.add_vertex(fused)

        # Rewire edges
        for v in pattern.vertices:
            for pred in graph.get_predecessors(v):
                if pred not in pattern.vertices:
                    graph.add_edge(HyperEdge.data_flow(pred, fused))
            for succ in graph.get_successors(v):
                if succ not in pattern.vertices:
                    graph.add_edge(HyperEdge.data_flow(fused, succ))
            graph.remove_vertex(v)

        return FusionResult(
            success=True,
            fused_graph=graph,
            fused_vertices=[fused],
            removed_vertices=pattern.vertices,
            speedup_estimate=1.2,
        )

    def _fuse_rust_cuda(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Fuse Rust → CUDA kernel launch."""
        return self._generic_fuse(
            graph,
            pattern,
            "fused_rust_cuda",
            HardwareAffinity.GPU,
            speedup=1.5,
        )

    def _fuse_python_rust_cuda(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Fuse Python → Rust → CUDA pipeline into single GPU kernel."""
        return self._generic_fuse(
            graph,
            pattern,
            "fused_polyglot_kernel",
            HardwareAffinity.GPU,
            speedup=3.0,
        )

    def _fuse_cpu_gpu_transfer(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Eliminate unnecessary CPU-GPU transfers."""
        # Find consecutive transfers
        transfers = [
            v for v in pattern.vertices if "memcpy" in v.attributes.get("function", "").lower()
        ]

        if len(transfers) < 2:
            return FusionResult(success=False)

        # Keep only necessary transfers
        kept_transfers = [transfers[0], transfers[-1]]
        removed = [t for t in transfers if t not in kept_transfers]

        for v in removed:
            graph.remove_vertex(v)

        return FusionResult(
            success=True,
            fused_graph=graph,
            removed_vertices=removed,
            speedup_estimate=1.3,
        )

    def _fuse_memory_coalescing(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Coalesce memory accesses for better bandwidth."""
        # Find load/store vertices accessing same region
        loads = [v for v in pattern.vertices if v.vertex_type == VertexType.LOAD]

        if len(loads) < 2:
            return FusionResult(success=False)

        # Create coalesced load
        from ..sir.vertices import AIONType, Provenance

        coalesced = Vertex.apply(
            function_name="coalesced_load",
            type_info=AIONType.array(AIONType(kind="unit"), len(loads)),
            effects={EffectKind.READ},
            provenance=Provenance(
                source_language="optimization",
                transformation_chain=["memory_coalescing"],
            ),
        )

        graph.add_vertex(coalesced)

        for v in loads:
            graph.remove_vertex(v)

        return FusionResult(
            success=True,
            fused_graph=graph,
            fused_vertices=[coalesced],
            removed_vertices=loads,
            speedup_estimate=1.4,
        )

    def _fuse_loops(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Fuse adjacent loops with compatible bounds."""
        return self._generic_fuse(
            graph,
            pattern,
            "fused_loop",
            HardwareAffinity.ANY,
            speedup=1.25,
        )

    def _fuse_kernels(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Fuse multiple GPU kernels into one."""
        kernels = [v for v in pattern.vertices if v.vertex_type == VertexType.KERNEL_LAUNCH]

        if len(kernels) < 2:
            return FusionResult(success=False)

        # Get largest grid/block dims
        max_grid = (1, 1, 1)
        max_block = (1, 1, 1)
        for k in kernels:
            grid = k.metadata.parallelism.get("grid_dim", (1, 1, 1))
            block = k.metadata.parallelism.get("block_dim", (1, 1, 1))
            max_grid = tuple(max(a, b) for a, b in zip(max_grid, grid))
            max_block = tuple(max(a, b) for a, b in zip(max_block, block))

        from ..sir.vertices import AIONType, Provenance

        fused_kernel = Vertex.kernel_launch(
            kernel_name="fused_kernel",
            grid_dim=max_grid,  # type: ignore
            block_dim=max_block,  # type: ignore
            type_info=AIONType(kind="fn"),
            affinity=HardwareAffinity.GPU,
            provenance=Provenance(
                source_language="optimization",
                transformation_chain=["kernel_fusion"],
            ),
        )

        graph.add_vertex(fused_kernel)

        # Rewire
        for k in kernels:
            for pred in graph.get_predecessors(k):
                if pred not in kernels:
                    graph.add_edge(HyperEdge.data_flow(pred, fused_kernel))
            for succ in graph.get_successors(k):
                if succ not in kernels:
                    graph.add_edge(HyperEdge.data_flow(fused_kernel, succ))
            graph.remove_vertex(k)

        return FusionResult(
            success=True,
            fused_graph=graph,
            fused_vertices=[fused_kernel],
            removed_vertices=kernels,
            speedup_estimate=2.0,
        )

    def _fuse_dataflow(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Fuse dataflow operators (SQL optimization)."""
        return self._generic_fuse(
            graph,
            pattern,
            "fused_dataflow",
            HardwareAffinity.ANY,
            speedup=1.5,
        )

    def _fuse_pipeline(self, graph: HyperGraph, pattern: FusionPattern) -> FusionResult:
        """Fuse pipeline stages."""
        return self._generic_fuse(
            graph,
            pattern,
            "fused_pipeline",
            HardwareAffinity.ANY,
            speedup=1.3,
        )

    def _generic_fuse(
        self,
        graph: HyperGraph,
        pattern: FusionPattern,
        name: str,
        affinity: HardwareAffinity,
        speedup: float,
    ) -> FusionResult:
        """Generic fusion helper."""
        from ..sir.vertices import AIONType, Provenance

        fused = Vertex.apply(
            function_name=name,
            type_info=AIONType(kind="fn"),
            effects=set().union(*(v.metadata.effects for v in pattern.vertices)),
            provenance=Provenance(
                source_language="optimization",
                transformation_chain=[f"{name}_fusion"],
            ),
        )
        fused.metadata.hardware_affinity = affinity

        graph.add_vertex(fused)

        # Rewire edges
        for v in pattern.vertices:
            for pred in graph.get_predecessors(v):
                if pred not in pattern.vertices:
                    graph.add_edge(HyperEdge.data_flow(pred, fused))
            for succ in graph.get_successors(v):
                if succ not in pattern.vertices:
                    graph.add_edge(HyperEdge.data_flow(fused, succ))
            graph.remove_vertex(v)

        return FusionResult(
            success=True,
            fused_graph=graph,
            fused_vertices=[fused],
            removed_vertices=pattern.vertices,
            speedup_estimate=speedup,
        )


def detect_fusion_patterns(graph: HyperGraph) -> list[FusionPattern]:
    """Detect fusion opportunities in a hypergraph.

    Patterns detected:
    - PythonCall → RustFFI → CUDA
    - Sequential memory operations
    - Adjacent loop nests
    - Consecutive kernel launches

    Args:
        graph: Input hypergraph

    Returns:
        List of detected fusion patterns
    """
    patterns: list[FusionPattern] = []

    # Detect Python → Rust → CUDA pattern
    patterns.extend(_detect_polyglot_pipeline(graph))

    # Detect kernel fusion opportunities
    patterns.extend(_detect_kernel_fusion(graph))

    # Detect memory coalescing opportunities
    patterns.extend(_detect_memory_coalescing(graph))

    # Detect loop fusion opportunities
    patterns.extend(_detect_loop_fusion(graph))

    # Detect dataflow fusion (SQL)
    patterns.extend(_detect_dataflow_fusion(graph))

    return patterns


def _detect_polyglot_pipeline(graph: HyperGraph) -> list[FusionPattern]:
    """Detect cross-language pipeline patterns."""
    patterns = []

    # Find sequences of calls with different source languages
    visited: set[str] = set()

    for vertex in graph.vertices:
        if vertex.id in visited:
            continue

        prov = vertex.metadata.provenance
        if not prov:
            continue

        # Look for Python → Rust → CUDA sequence
        if prov.source_language == "Python":
            sequence = [vertex]
            current = vertex

            while True:
                succs = graph.get_successors(current)
                if not succs:
                    break

                next_vertex = None
                for s in succs:
                    s_prov = s.metadata.provenance
                    if s_prov:
                        if s_prov.source_language in ("Rust", "CUDA") and s.id not in visited:
                            next_vertex = s
                            break

                if next_vertex:
                    sequence.append(next_vertex)
                    visited.add(next_vertex.id)
                    current = next_vertex
                else:
                    break

            if len(sequence) >= 2:
                languages = [
                    v.metadata.provenance.source_language for v in sequence if v.metadata.provenance
                ]

                if "Python" in languages and "Rust" in languages and "CUDA" in languages:
                    kind = FusionPatternKind.PYTHON_RUST_CUDA
                    speedup = 3.5
                elif "Python" in languages and "Rust" in languages:
                    kind = FusionPatternKind.PYTHON_RUST_FFI
                    speedup = 1.3
                elif "Rust" in languages and "CUDA" in languages:
                    kind = FusionPatternKind.RUST_CUDA
                    speedup = 1.5
                else:
                    continue

                patterns.append(
                    FusionPattern(
                        kind=kind,
                        vertices=sequence,
                        estimated_speedup=speedup,
                        hardware_target=(
                            HardwareAffinity.GPU if "CUDA" in languages else HardwareAffinity.CPU
                        ),
                    )
                )

    return patterns


def _detect_kernel_fusion(graph: HyperGraph) -> list[FusionPattern]:
    """Detect GPU kernel fusion opportunities."""
    patterns = []

    # Find consecutive kernel launches
    kernels = [v for v in graph.vertices if v.vertex_type == VertexType.KERNEL_LAUNCH]

    if len(kernels) < 2:
        return patterns

    # Group kernels that can be fused
    groups: list[list[Vertex]] = []
    current_group: list[Vertex] = []

    for kernel in graph.topological_order():
        if kernel.vertex_type != VertexType.KERNEL_LAUNCH:
            continue

        if not current_group:
            current_group = [kernel]
        else:
            # Check if can fuse with current group
            last = current_group[-1]

            # Same hardware affinity and compatible parallelism
            if (
                kernel.metadata.hardware_affinity == last.metadata.hardware_affinity
                and _compatible_parallelism(kernel, last)
            ):
                current_group.append(kernel)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [kernel]

    if len(current_group) >= 2:
        groups.append(current_group)

    for group in groups:
        patterns.append(
            FusionPattern(
                kind=FusionPatternKind.KERNEL_FUSION,
                vertices=group,
                estimated_speedup=1.5 * len(group),
                hardware_target=group[0].metadata.hardware_affinity,
            )
        )

    return patterns


def _detect_memory_coalescing(graph: HyperGraph) -> list[FusionPattern]:
    """Detect memory coalescing opportunities."""
    patterns = []

    # Group loads by region
    loads_by_region: dict[str | None, list[Vertex]] = {}

    for v in graph.vertices:
        if v.vertex_type == VertexType.LOAD:
            region = v.metadata.region
            loads_by_region.setdefault(region, []).append(v)

    for region, loads in loads_by_region.items():
        if len(loads) >= 4:  # Worth coalescing
            patterns.append(
                FusionPattern(
                    kind=FusionPatternKind.MEMORY_COALESCING,
                    vertices=loads,
                    estimated_speedup=1.3,
                    hardware_target=HardwareAffinity.ANY,
                )
            )

    return patterns


def _detect_loop_fusion(graph: HyperGraph) -> list[FusionPattern]:
    """Detect loop fusion opportunities."""
    # Simplified - would analyze loop bounds
    return []


def _detect_dataflow_fusion(graph: HyperGraph) -> list[FusionPattern]:
    """Detect dataflow (SQL) fusion opportunities."""
    patterns = []

    # Find consecutive filter-project-aggregate chains
    sql_ops = [
        v
        for v in graph.vertices
        if v.attributes.get("operator") in ("FILTER", "PROJECT", "AGGREGATE", "TABLE_SCAN")
    ]

    if len(sql_ops) >= 3:
        patterns.append(
            FusionPattern(
                kind=FusionPatternKind.DATAFLOW_FUSION,
                vertices=sql_ops,
                estimated_speedup=2.0,
                hardware_target=HardwareAffinity.ANY,
            )
        )

    return patterns


def _compatible_parallelism(a: Vertex, b: Vertex) -> bool:
    """Check if two kernels have compatible parallelism for fusion."""
    a_grid = a.metadata.parallelism.get("grid_dim", (1, 1, 1))
    b_grid = b.metadata.parallelism.get("grid_dim", (1, 1, 1))

    a_block = a.metadata.parallelism.get("block_dim", (1, 1, 1))
    b_block = b.metadata.parallelism.get("block_dim", (1, 1, 1))

    # Compatible if same or one is subset
    return a_block == b_block or a_block == (1, 1, 1) or b_block == (1, 1, 1)


class CrossLanguageFuser:
    """Specialized fuser for cross-language pipelines.

    Handles the complex task of fusing operations from
    different source languages while preserving semantics.
    """

    def __init__(self) -> None:
        """Initialize cross-language fuser."""
        self.fusion = KernelFusion()

    def fuse_pipeline(
        self,
        python_graph: HyperGraph,
        rust_graph: HyperGraph,
        cuda_graph: HyperGraph | None = None,
    ) -> HyperGraph:
        """Fuse graphs from different languages.

        Args:
            python_graph: Graph from Python code
            rust_graph: Graph from Rust code
            cuda_graph: Optional graph from CUDA code

        Returns:
            Fused hypergraph
        """
        from ..sir.hypergraph import merge_graphs

        # Merge graphs
        graphs = [python_graph, rust_graph]
        if cuda_graph:
            graphs.append(cuda_graph)

        merged = merge_graphs(graphs, name="cross_language_fused")

        # Apply fusion
        result = self.fusion.optimize(merged)

        return result.fused_graph or merged

    def fuse_with_zero_copy(
        self,
        graph: HyperGraph,
    ) -> HyperGraph:
        """Apply zero-copy optimization to cross-language transfers.

        Eliminates unnecessary copies between language runtimes.

        Args:
            graph: Input graph

        Returns:
            Optimized graph with zero-copy transfers
        """
        optimized = graph.clone()

        # Find memory transfers
        transfers = [
            v
            for v in optimized.vertices
            if "memcpy" in v.attributes.get("function", "").lower()
            or "copy" in v.attributes.get("function", "").lower()
        ]

        # Remove redundant copies
        for transfer in transfers:
            # Check if source and destination can share memory
            preds = optimized.get_predecessors(transfer)
            succs = optimized.get_successors(transfer)

            if preds and succs:
                src_region = preds[0].metadata.region
                dst_region = succs[0].metadata.region if succs[0].metadata else None

                # If regions are compatible, bypass the transfer
                if src_region == dst_region:
                    # Connect predecessors directly to successors
                    for pred in preds:
                        for succ in succs:
                            optimized.add_edge(HyperEdge.data_flow(pred, succ))
                    optimized.remove_vertex(transfer)

        return optimized
