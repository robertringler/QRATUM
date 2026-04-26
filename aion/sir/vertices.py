"""AION-SIR Vertex Definitions.

Defines the vertex types for the AION semantic hypergraph:
- Const: Constant values
- Alloc: Memory allocation
- Load: Memory load operations
- Store: Memory store operations
- Apply: Function application
- Phi: SSA phi nodes
- KernelLaunch: GPU/FPGA kernel launch

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import uuid4


class VertexType(Enum):
    """Enumeration of vertex types in AION-SIR."""

    CONST = auto()
    ALLOC = auto()
    LOAD = auto()
    STORE = auto()
    APPLY = auto()
    PHI = auto()
    KERNEL_LAUNCH = auto()
    PARAMETER = auto()
    RETURN = auto()
    BRANCH = auto()
    MERGE = auto()


class HardwareAffinity(Enum):
    """Hardware target affinity for vertices."""

    ANY = auto()
    CPU = auto()
    GPU = auto()
    GPU_STREAM0 = auto()
    GPU_STREAM1 = auto()
    FPGA = auto()
    FPGA_LUT = auto()
    TPU = auto()
    WASM = auto()
    JVM = auto()


class EffectKind(Enum):
    """Effect kinds for effect tracking."""

    PURE = auto()
    READ = auto()
    WRITE = auto()
    ALLOC = auto()
    FREE = auto()
    IO = auto()
    NETWORK = auto()
    THREAD_SPAWN = auto()
    THREAD_JOIN = auto()
    CHANNEL_SEND = auto()
    CHANNEL_RECV = auto()
    ACTOR_SEND = auto()
    WARP_SYNC = auto()
    PIPELINE_STAGE = auto()
    ARBITRARY = auto()


@dataclass
class VertexMetadata:
    """Metadata attached to vertices.

    Attributes:
        type_info: Type information for the vertex output
        effects: Set of effects produced by this vertex
        lifetime: Lifetime identifier for memory safety
        parallelism: Parallelism hints (SIMD width, warp count)
        provenance: Source location and cross-language origin
        hardware_affinity: Target hardware preference
        region: Memory region this vertex belongs to
    """

    type_info: AIONType | None = None
    effects: set[EffectKind] = field(default_factory=set)
    lifetime: str = "static"
    parallelism: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None
    hardware_affinity: HardwareAffinity = HardwareAffinity.ANY
    region: str | None = None


@dataclass
class Provenance:
    """Source provenance information for cross-language debugging.

    Attributes:
        source_language: Original source language (C, Rust, Python, etc.)
        source_file: Original source file path
        source_line: Line number in source
        source_column: Column number in source
        original_name: Original symbol/variable name
        transformation_chain: List of transformations applied
    """

    source_language: str
    source_file: str = ""
    source_line: int = 0
    source_column: int = 0
    original_name: str = ""
    transformation_chain: list[str] = field(default_factory=list)

    def add_transformation(self, transformation: str) -> Provenance:
        """Add a transformation to the chain."""
        new_chain = self.transformation_chain.copy()
        new_chain.append(transformation)
        return Provenance(
            source_language=self.source_language,
            source_file=self.source_file,
            source_line=self.source_line,
            source_column=self.source_column,
            original_name=self.original_name,
            transformation_chain=new_chain,
        )


@dataclass
class AIONType:
    """AION type representation for dependent type system.

    Supports basic types, function types with effects, dependent types,
    and linear/affine types.
    """

    kind: str  # "int", "float", "ptr", "fn", "array", "struct", "region"
    params: list[AIONType] = field(default_factory=list)
    size: int | None = None
    effects: set[EffectKind] = field(default_factory=set)
    refinement: str | None = None  # SMT refinement predicate
    linear: bool = False  # Linear type (must be used exactly once)
    affine: bool = False  # Affine type (can be used at most once)

    @staticmethod
    def int(bits: int = 64, signed: bool = True) -> AIONType:
        """Create an integer type."""
        return AIONType(kind="int", size=bits, refinement=f"signed={signed}")

    @staticmethod
    def float(bits: int = 64) -> AIONType:
        """Create a floating point type."""
        return AIONType(kind="float", size=bits)

    @staticmethod
    def ptr(pointee: AIONType, region: str = "heap") -> AIONType:
        """Create a pointer type with region annotation."""
        return AIONType(kind="ptr", params=[pointee], refinement=f"region={region}")

    @staticmethod
    def fn(
        params: list[AIONType], ret: AIONType, effects: set[EffectKind] | None = None
    ) -> AIONType:
        """Create a function type with effects."""
        return AIONType(kind="fn", params=params + [ret], effects=effects or set())

    @staticmethod
    def array(element: AIONType, length: int | str) -> AIONType:
        """Create an array type with optional dependent length."""
        return AIONType(kind="array", params=[element], refinement=f"length={length}")

    @staticmethod
    def tensor(element: AIONType, shape: list[int | str]) -> AIONType:
        """Create a tensor type with shape."""
        shape_str = ",".join(str(s) for s in shape)
        return AIONType(kind="tensor", params=[element], refinement=f"shape=[{shape_str}]")


@dataclass
class Vertex:
    """A vertex in the AION-SIR hypergraph.

    Vertices represent computation nodes including:
    - Const: Compile-time constants
    - Alloc: Memory allocation
    - Load/Store: Memory operations
    - Apply: Function application
    - Phi: Control flow merge (SSA)
    - KernelLaunch: Hardware kernel dispatch

    Attributes:
        id: Unique vertex identifier
        vertex_type: Type of the vertex
        value: Associated value (for Const vertices)
        attributes: Additional typed attributes
        metadata: Vertex metadata (type, effects, provenance)
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    vertex_type: VertexType = VertexType.CONST
    value: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: VertexMetadata = field(default_factory=VertexMetadata)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vertex):
            return False
        return self.id == other.id

    @staticmethod
    def const(
        value: Any, type_info: AIONType | None = None, provenance: Provenance | None = None
    ) -> Vertex:
        """Create a constant vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.CONST,
            value=value,
            metadata=metadata,
        )

    @staticmethod
    def alloc(
        size: int | str,
        type_info: AIONType,
        region: str = "heap",
        affinity: HardwareAffinity = HardwareAffinity.ANY,
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create an allocation vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.ALLOC},
            region=region,
            hardware_affinity=affinity,
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.ALLOC,
            attributes={"size": size, "region": region},
            metadata=metadata,
        )

    @staticmethod
    def load(
        type_info: AIONType,
        region: str = "heap",
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create a load vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.READ},
            region=region,
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.LOAD,
            metadata=metadata,
        )

    @staticmethod
    def store(
        type_info: AIONType,
        region: str = "heap",
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create a store vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.WRITE},
            region=region,
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.STORE,
            metadata=metadata,
        )

    @staticmethod
    def apply(
        function_name: str,
        type_info: AIONType,
        effects: set[EffectKind] | None = None,
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create a function application vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects=effects or {EffectKind.PURE},
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.APPLY,
            attributes={"function": function_name},
            metadata=metadata,
        )

    @staticmethod
    def phi(
        type_info: AIONType,
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create a phi node for SSA."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.PHI,
            metadata=metadata,
        )

    @staticmethod
    def kernel_launch(
        kernel_name: str,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
        type_info: AIONType,
        affinity: HardwareAffinity = HardwareAffinity.GPU,
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create a kernel launch vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.IO, EffectKind.WARP_SYNC},
            hardware_affinity=affinity,
            parallelism={
                "grid_dim": grid_dim,
                "block_dim": block_dim,
            },
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.KERNEL_LAUNCH,
            attributes={"kernel": kernel_name},
            metadata=metadata,
        )

    @staticmethod
    def parameter(
        name: str,
        type_info: AIONType,
        index: int = 0,
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create a function parameter vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.PARAMETER,
            attributes={"name": name, "index": index},
            metadata=metadata,
        )

    @staticmethod
    def ret(
        type_info: AIONType,
        provenance: Provenance | None = None,
    ) -> Vertex:
        """Create a return vertex."""
        metadata = VertexMetadata(
            type_info=type_info,
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        return Vertex(
            vertex_type=VertexType.RETURN,
            metadata=metadata,
        )

    def with_affinity(self, affinity: HardwareAffinity) -> Vertex:
        """Create a copy of this vertex with updated hardware affinity."""
        new_metadata = VertexMetadata(
            type_info=self.metadata.type_info,
            effects=self.metadata.effects.copy(),
            lifetime=self.metadata.lifetime,
            parallelism=self.metadata.parallelism.copy(),
            provenance=self.metadata.provenance,
            hardware_affinity=affinity,
            region=self.metadata.region,
        )
        return Vertex(
            id=self.id,
            vertex_type=self.vertex_type,
            value=self.value,
            attributes=self.attributes.copy(),
            metadata=new_metadata,
        )

    def with_region(self, region: str) -> Vertex:
        """Create a copy of this vertex with updated region."""
        new_metadata = VertexMetadata(
            type_info=self.metadata.type_info,
            effects=self.metadata.effects.copy(),
            lifetime=self.metadata.lifetime,
            parallelism=self.metadata.parallelism.copy(),
            provenance=self.metadata.provenance,
            hardware_affinity=self.metadata.hardware_affinity,
            region=region,
        )
        return Vertex(
            id=self.id,
            vertex_type=self.vertex_type,
            value=self.value,
            attributes=self.attributes.copy(),
            metadata=new_metadata,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize vertex to dictionary for .aion_sir section."""
        return {
            "id": self.id,
            "type": self.vertex_type.name,
            "value": self.value,
            "attributes": self.attributes,
            "metadata": {
                "type_info": self._serialize_type(self.metadata.type_info),
                "effects": [e.name for e in self.metadata.effects],
                "lifetime": self.metadata.lifetime,
                "parallelism": self.metadata.parallelism,
                "hardware_affinity": self.metadata.hardware_affinity.name,
                "region": self.metadata.region,
                "provenance": self._serialize_provenance(self.metadata.provenance),
            },
        }

    def _serialize_type(self, t: AIONType | None) -> dict[str, Any] | None:
        """Serialize an AION type."""
        if t is None:
            return None
        return {
            "kind": t.kind,
            "params": [self._serialize_type(p) for p in t.params],
            "size": t.size,
            "effects": [e.name for e in t.effects],
            "refinement": t.refinement,
            "linear": t.linear,
            "affine": t.affine,
        }

    def _serialize_provenance(self, p: Provenance | None) -> dict[str, Any] | None:
        """Serialize provenance information."""
        if p is None:
            return None
        return {
            "source_language": p.source_language,
            "source_file": p.source_file,
            "source_line": p.source_line,
            "source_column": p.source_column,
            "original_name": p.original_name,
            "transformation_chain": p.transformation_chain,
        }
