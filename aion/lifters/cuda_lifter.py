"""CUDA/OpenCL Lifter for AION.

Lifts CUDA and OpenCL kernels to AION-SIR hypergraph with:
- Grid/block/thread index mapping to vertices
- ParallelEdge with hardware affinity
- Warp-level synchronization
- Device memory region mapping

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ..memory.regions import Region
from ..sir.edges import HyperEdge, ParallelismKind
from ..sir.hypergraph import HyperGraph
from ..sir.vertices import (
    AIONType,
    EffectKind,
    HardwareAffinity,
    Provenance,
    Vertex,
)


class CUDANodeKind(Enum):
    """CUDA AST node kinds."""

    KERNEL_DEF = auto()
    DEVICE_FUNC = auto()
    HOST_FUNC = auto()
    KERNEL_LAUNCH = auto()
    SHARED_DECL = auto()
    GLOBAL_DECL = auto()
    CONSTANT_DECL = auto()
    SYNCTHREADS = auto()
    THREAD_IDX = auto()
    BLOCK_IDX = auto()
    BLOCK_DIM = auto()
    GRID_DIM = auto()
    ATOMIC_OP = auto()
    CUDA_MALLOC = auto()
    CUDA_FREE = auto()
    CUDA_MEMCPY = auto()
    WARP_VOTE = auto()
    WARP_SHUFFLE = auto()


@dataclass
class CUDAKernel:
    """CUDA kernel representation."""

    name: str
    params: list[tuple[str, str]]  # (name, type)
    grid_dim: tuple[int | str, int | str, int | str] = (1, 1, 1)
    block_dim: tuple[int | str, int | str, int | str] = (1, 1, 1)
    shared_mem_size: int = 0
    stream: int = 0


@dataclass
class CUDAASTNode:
    """CUDA AST node representation."""

    kind: CUDANodeKind
    value: Any = None
    children: list[CUDAASTNode] = field(default_factory=list)
    source_loc: tuple[str, int, int] = ("", 0, 0)
    name: str = ""
    type_info: str = ""


class CUDALifter:
    """Lifts CUDA code to AION-SIR.

    Handles:
    - __global__ kernel functions
    - __device__ functions
    - Grid/block/thread indexing
    - Shared memory declarations
    - Synchronization primitives
    - Memory operations (cudaMalloc, cudaMemcpy)
    """

    def __init__(self, source_file: str = "") -> None:
        """Initialize the CUDA lifter."""
        self.source_file = source_file
        self.source_language = "CUDA"
        self.graph = HyperGraph()
        self.kernels: dict[str, CUDAKernel] = {}
        self.variables: dict[str, Vertex] = {}
        self.current_kernel: str = ""

        # Initialize GPU regions
        self.regions = {
            "global": Region.gpu_global("global"),
            "shared": Region.gpu_shared("shared"),
            "stream0": Region.gpu_global("stream0", stream=0),
            "stream1": Region.gpu_global("stream1", stream=1),
        }

    def lift(self, ast: CUDAASTNode) -> HyperGraph:
        """Lift CUDA AST to AION-SIR hypergraph.

        Args:
            ast: Root of CUDA AST

        Returns:
            AION-SIR hypergraph with GPU annotations
        """
        self.graph = HyperGraph(name=self.source_file)
        self._lift_node(ast)
        return self.graph

    def lift_kernel(self, kernel: CUDAKernel, body: CUDAASTNode) -> HyperGraph:
        """Lift a single CUDA kernel.

        Args:
            kernel: Kernel metadata
            body: Kernel body AST

        Returns:
            AION-SIR hypergraph for the kernel
        """
        self.current_kernel = kernel.name
        self.kernels[kernel.name] = kernel

        provenance = Provenance(
            source_language="CUDA",
            source_file=self.source_file,
            original_name=kernel.name,
        )

        # Create kernel entry
        grid = tuple(int(d) if isinstance(d, int) else 1 for d in kernel.grid_dim)
        block = tuple(int(d) if isinstance(d, int) else 1 for d in kernel.block_dim)

        entry = Vertex.kernel_launch(
            kernel_name=kernel.name,
            grid_dim=grid,  # type: ignore
            block_dim=block,  # type: ignore
            type_info=AIONType(kind="fn"),
            affinity=HardwareAffinity.GPU,
            provenance=provenance,
        )
        self.graph.add_vertex(entry)
        self.graph.entry = entry

        # Create parameter vertices
        for i, (param_name, param_type) in enumerate(kernel.params):
            param = Vertex.parameter(
                name=param_name,
                type_info=self._cuda_type_to_aion(param_type),
                index=i,
                provenance=provenance,
            )
            self.graph.add_vertex(param)
            self.graph.add_edge(HyperEdge.data_flow(param, entry))
            self.variables[param_name] = param

        # Lift kernel body
        self._lift_node(body)

        # Add parallel edge for thread-level parallelism
        parallel_vertices = [entry]
        self.graph.add_edge(
            HyperEdge.parallel_edge(
                parallel_vertices,
                kind=ParallelismKind.SIMT,
                warp_size=32,
                affinity=HardwareAffinity.GPU,
            )
        )

        return self.graph

    def lift_from_source(self, source: str) -> HyperGraph:
        """Lift CUDA source code directly.

        Args:
            source: CUDA source code string

        Returns:
            AION-SIR hypergraph
        """
        ast = self._parse_cuda_source(source)
        return self.lift(ast)

    def _parse_cuda_source(self, source: str) -> CUDAASTNode:
        """Parse CUDA source to AST.

        Simplified parser for demonstration.
        """
        root = CUDAASTNode(kind=CUDANodeKind.HOST_FUNC, name="root")
        lines = source.strip().split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith("//"):
                continue

            # Kernel definition
            if "__global__" in line:
                kernel_node = self._parse_kernel_def(line, lines, i)
                root.children.append(kernel_node)
                # Skip kernel body
                brace_count = 1
                while i < len(lines) and brace_count > 0:
                    brace_count += lines[i].count("{") - lines[i].count("}")
                    i += 1

            # Kernel launch
            elif "<<<" in line and ">>>" in line:
                launch_node = self._parse_kernel_launch(line, i)
                root.children.append(launch_node)

            # cudaMalloc
            elif "cudaMalloc" in line:
                malloc_node = CUDAASTNode(
                    kind=CUDANodeKind.CUDA_MALLOC,
                    source_loc=(self.source_file, i, 0),
                )
                root.children.append(malloc_node)

            # cudaMemcpy
            elif "cudaMemcpy" in line:
                memcpy_node = CUDAASTNode(
                    kind=CUDANodeKind.CUDA_MEMCPY,
                    source_loc=(self.source_file, i, 0),
                )
                root.children.append(memcpy_node)

            # __shared__ declaration
            elif "__shared__" in line:
                shared_node = self._parse_shared_decl(line, i)
                root.children.append(shared_node)

        return root

    def _parse_kernel_def(self, line: str, lines: list[str], lineno: int) -> CUDAASTNode:
        """Parse kernel definition."""
        # Extract kernel name and params
        func_start = line.index("__global__") + 10
        rest = line[func_start:].strip()

        # Find function name
        if "(" in rest:
            name_part = rest[: rest.index("(")].strip()
            name = name_part.split()[-1] if " " in name_part else name_part
        else:
            name = "kernel"

        return CUDAASTNode(
            kind=CUDANodeKind.KERNEL_DEF,
            name=name,
            source_loc=(self.source_file, lineno, 0),
        )

    def _parse_kernel_launch(self, line: str, lineno: int) -> CUDAASTNode:
        """Parse kernel launch configuration."""
        # Extract kernel name
        kernel_name = line[: line.index("<<<")].strip().split()[-1]

        # Extract launch config
        config_start = line.index("<<<") + 3
        config_end = line.index(">>>")
        config = line[config_start:config_end]

        # Parse grid and block dims
        parts = config.split(",")
        grid_dim = (1, 1, 1)
        block_dim = (1, 1, 1)

        if len(parts) >= 2:
            try:
                grid_dim = (int(parts[0].strip()), 1, 1)
                block_dim = (int(parts[1].strip()), 1, 1)
            except ValueError:
                pass

        node = CUDAASTNode(
            kind=CUDANodeKind.KERNEL_LAUNCH,
            name=kernel_name,
            value={"grid": grid_dim, "block": block_dim},
            source_loc=(self.source_file, lineno, 0),
        )

        return node

    def _parse_shared_decl(self, line: str, lineno: int) -> CUDAASTNode:
        """Parse shared memory declaration."""
        # Extract variable name and type
        parts = line.replace("__shared__", "").strip().split()
        var_type = parts[0] if parts else "float"
        var_name = parts[-1].rstrip(";").split("[")[0] if parts else "shared_var"

        return CUDAASTNode(
            kind=CUDANodeKind.SHARED_DECL,
            name=var_name,
            type_info=var_type,
            source_loc=(self.source_file, lineno, 0),
        )

    def _lift_node(self, node: CUDAASTNode) -> Vertex | None:
        """Lift a CUDA AST node to AION-SIR vertex."""
        provenance = Provenance(
            source_language=self.source_language,
            source_file=node.source_loc[0],
            source_line=node.source_loc[1],
            source_column=node.source_loc[2],
            original_name=node.name,
        )

        if node.kind == CUDANodeKind.HOST_FUNC:
            for child in node.children:
                self._lift_node(child)
            return None

        elif node.kind == CUDANodeKind.KERNEL_DEF:
            return self._lift_kernel_def(node, provenance)

        elif node.kind == CUDANodeKind.KERNEL_LAUNCH:
            return self._lift_kernel_launch(node, provenance)

        elif node.kind == CUDANodeKind.SHARED_DECL:
            return self._lift_shared_decl(node, provenance)

        elif node.kind == CUDANodeKind.SYNCTHREADS:
            return self._lift_syncthreads(node, provenance)

        elif node.kind == CUDANodeKind.THREAD_IDX:
            return self._lift_thread_idx(node, provenance)

        elif node.kind == CUDANodeKind.CUDA_MALLOC:
            return self._lift_cuda_malloc(node, provenance)

        elif node.kind == CUDANodeKind.CUDA_MEMCPY:
            return self._lift_cuda_memcpy(node, provenance)

        elif node.kind == CUDANodeKind.ATOMIC_OP:
            return self._lift_atomic(node, provenance)

        elif node.kind == CUDANodeKind.WARP_SHUFFLE:
            return self._lift_warp_shuffle(node, provenance)

        return None

    def _lift_kernel_def(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift kernel definition."""
        self.current_kernel = node.name

        # Create kernel entry with SIMT parallelism
        entry = Vertex.kernel_launch(
            kernel_name=node.name,
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),  # Default block size
            type_info=AIONType(kind="fn"),
            affinity=HardwareAffinity.GPU,
            provenance=provenance,
        )
        self.graph.add_vertex(entry)

        # Add parallel edge for GPU parallelism
        self.graph.add_edge(
            HyperEdge.parallel_edge(
                [entry],
                kind=ParallelismKind.SIMT,
                warp_size=32,
                affinity=HardwareAffinity.GPU,
            )
        )

        # Lift children
        for child in node.children:
            v = self._lift_node(child)
            if v:
                self.graph.add_edge(HyperEdge.data_flow(entry, v))

        return entry

    def _lift_kernel_launch(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift kernel launch from host code."""
        config = node.value or {}
        grid = config.get("grid", (1, 1, 1))
        block = config.get("block", (256, 1, 1))

        launch = Vertex.kernel_launch(
            kernel_name=node.name,
            grid_dim=grid,
            block_dim=block,
            type_info=AIONType(kind="unit"),
            affinity=HardwareAffinity.GPU,
            provenance=provenance,
        )
        self.graph.add_vertex(launch)

        # Add parallel edge
        self.graph.add_edge(
            HyperEdge.parallel_edge(
                [launch],
                kind=ParallelismKind.SIMT,
                num_threads=grid[0] * block[0],
                warp_size=32,
                affinity=HardwareAffinity.GPU,
            )
        )

        return launch

    def _lift_shared_decl(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift shared memory declaration."""
        type_info = self._cuda_type_to_aion(node.type_info)

        alloc = Vertex.alloc(
            size=type_info.size or 4,
            type_info=type_info,
            region="shared",
            affinity=HardwareAffinity.GPU,
            provenance=provenance,
        )
        self.graph.add_vertex(alloc)
        self.variables[node.name] = alloc

        return alloc

    def _lift_syncthreads(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift __syncthreads() call."""
        sync = Vertex.apply(
            function_name="__syncthreads",
            type_info=AIONType(kind="unit"),
            effects={EffectKind.WARP_SYNC},
            provenance=provenance,
        )
        sync.metadata.hardware_affinity = HardwareAffinity.GPU
        self.graph.add_vertex(sync)

        return sync

    def _lift_thread_idx(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift threadIdx.x/y/z access."""
        type_info = AIONType.int(32)

        idx = Vertex.apply(
            function_name="threadIdx",
            type_info=type_info,
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        idx.attributes["dimension"] = node.value  # x, y, or z
        idx.metadata.hardware_affinity = HardwareAffinity.GPU
        self.graph.add_vertex(idx)

        return idx

    def _lift_cuda_malloc(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift cudaMalloc to GPU allocation."""
        type_info = AIONType.ptr(AIONType(kind="unit"), region="global")

        alloc = Vertex.alloc(
            size=0,  # Size would be parsed from call
            type_info=type_info,
            region="global",
            affinity=HardwareAffinity.GPU,
            provenance=provenance,
        )
        self.graph.add_vertex(alloc)

        return alloc

    def _lift_cuda_memcpy(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift cudaMemcpy for host-device transfer."""
        # Create transfer operation
        transfer = Vertex.apply(
            function_name="cudaMemcpy",
            type_info=AIONType(kind="unit"),
            effects={EffectKind.IO, EffectKind.WRITE},
            provenance=provenance,
        )
        self.graph.add_vertex(transfer)

        # Add region edge for the transfer
        # Would connect source and destination regions

        return transfer

    def _lift_atomic(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift atomic operations."""

        type_info = AIONType.int(32)

        atomic = Vertex.apply(
            function_name=f"atomic_{node.value}",
            type_info=type_info,
            effects={EffectKind.WRITE},
            provenance=provenance,
        )
        atomic.metadata.hardware_affinity = HardwareAffinity.GPU
        self.graph.add_vertex(atomic)

        return atomic

    def _lift_warp_shuffle(self, node: CUDAASTNode, provenance: Provenance) -> Vertex:
        """Lift warp shuffle operations."""
        type_info = AIONType.int(32)

        shuffle = Vertex.apply(
            function_name=f"__shfl_{node.value}",
            type_info=type_info,
            effects={EffectKind.WARP_SYNC},
            provenance=provenance,
        )
        shuffle.metadata.hardware_affinity = HardwareAffinity.GPU
        shuffle.metadata.parallelism["warp_level"] = True
        self.graph.add_vertex(shuffle)

        return shuffle

    def _cuda_type_to_aion(self, cuda_type: str) -> AIONType:
        """Convert CUDA type to AION type."""
        type_map = {
            "int": AIONType.int(32),
            "int2": AIONType.array(AIONType.int(32), 2),
            "int4": AIONType.array(AIONType.int(32), 4),
            "float": AIONType.float(32),
            "float2": AIONType.array(AIONType.float(32), 2),
            "float4": AIONType.array(AIONType.float(32), 4),
            "double": AIONType.float(64),
            "half": AIONType.float(16),
            "dim3": AIONType.array(AIONType.int(32), 3),
        }

        # Handle pointers
        if "*" in cuda_type:
            base = cuda_type.replace("*", "").strip()
            base_type = type_map.get(base, AIONType(kind="unit"))
            return AIONType.ptr(base_type, region="global")

        return type_map.get(cuda_type, AIONType(kind="unit"))


class OpenCLLifter:
    """Lifts OpenCL code to AION-SIR.

    Similar to CUDALifter but handles OpenCL specifics:
    - __kernel functions
    - work-item/work-group indexing
    - local/global memory qualifiers
    """

    def __init__(self, source_file: str = "") -> None:
        """Initialize OpenCL lifter."""
        self.source_file = source_file
        self.source_language = "OpenCL"
        self.graph = HyperGraph()
        self.variables: dict[str, Vertex] = {}

        # OpenCL memory spaces
        self.regions = {
            "global": Region.gpu_global("global"),
            "local": Region.gpu_shared("local"),
            "constant": Region(name="constant", kind=Region.gpu_global("constant").kind),
            "private": Region.stack("private"),
        }

    def lift_kernel(
        self,
        kernel_name: str,
        params: list[tuple[str, str]],
        body_source: str,
        global_size: tuple[int, int, int] = (1, 1, 1),
        local_size: tuple[int, int, int] = (1, 1, 1),
    ) -> HyperGraph:
        """Lift an OpenCL kernel.

        Args:
            kernel_name: Kernel function name
            params: List of (name, type) parameters
            body_source: Kernel body source code
            global_size: NDRange global work size
            local_size: NDRange local work size

        Returns:
            AION-SIR hypergraph
        """
        self.graph = HyperGraph(name=kernel_name)

        provenance = Provenance(
            source_language="OpenCL",
            source_file=self.source_file,
            original_name=kernel_name,
        )

        # Create kernel entry
        entry = Vertex.kernel_launch(
            kernel_name=kernel_name,
            grid_dim=global_size,
            block_dim=local_size,
            type_info=AIONType(kind="fn"),
            affinity=HardwareAffinity.GPU,
            provenance=provenance,
        )
        self.graph.add_vertex(entry)
        self.graph.entry = entry

        # Create parameters
        for i, (param_name, param_type) in enumerate(params):
            param = Vertex.parameter(
                name=param_name,
                type_info=self._ocl_type_to_aion(param_type),
                index=i,
                provenance=provenance,
            )
            self.graph.add_vertex(param)
            self.graph.add_edge(HyperEdge.data_flow(param, entry))
            self.variables[param_name] = param

        # Add parallel edge
        total_work_items = global_size[0] * global_size[1] * global_size[2]
        self.graph.add_edge(
            HyperEdge.parallel_edge(
                [entry],
                kind=ParallelismKind.SIMT,
                num_threads=total_work_items,
                affinity=HardwareAffinity.GPU,
            )
        )

        return self.graph

    def _ocl_type_to_aion(self, ocl_type: str) -> AIONType:
        """Convert OpenCL type to AION type."""
        type_map = {
            "int": AIONType.int(32),
            "int2": AIONType.array(AIONType.int(32), 2),
            "int4": AIONType.array(AIONType.int(32), 4),
            "int8": AIONType.array(AIONType.int(32), 8),
            "int16": AIONType.array(AIONType.int(32), 16),
            "float": AIONType.float(32),
            "float2": AIONType.array(AIONType.float(32), 2),
            "float4": AIONType.array(AIONType.float(32), 4),
            "float8": AIONType.array(AIONType.float(32), 8),
            "float16": AIONType.array(AIONType.float(32), 16),
            "double": AIONType.float(64),
            "half": AIONType.float(16),
            "size_t": AIONType.int(64),
        }

        # Handle memory space qualifiers
        space = "global"
        clean_type = ocl_type
        for qualifier in (
            "__global",
            "__local",
            "__constant",
            "__private",
            "global",
            "local",
            "constant",
            "private",
        ):
            if qualifier in ocl_type:
                space = qualifier.replace("__", "")
                clean_type = ocl_type.replace(qualifier, "").strip()
                break

        # Handle pointers
        if "*" in clean_type:
            base = clean_type.replace("*", "").strip()
            base_type = type_map.get(base, AIONType(kind="unit"))
            return AIONType.ptr(base_type, region=space)

        return type_map.get(clean_type, AIONType(kind="unit"))
