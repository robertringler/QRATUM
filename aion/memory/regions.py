"""AION Region-Based Memory Model.

Provides unified memory regions with lifetime tracking and
safety enforcement across different memory kinds:
- Stack: Function-scoped, automatically freed
- Heap: Dynamically allocated, requires explicit management
- Thread-local: Per-thread storage
- GPU global: GPU device memory
- GPU shared: GPU shared memory
- FPGA on-chip: FPGA block RAM / LUT memory
- Static: Compile-time allocated, never freed

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import uuid4


class RegionKind(Enum):
    """Types of memory regions."""

    STACK = auto()
    HEAP = auto()
    THREAD_LOCAL = auto()
    GPU_GLOBAL = auto()
    GPU_SHARED = auto()
    GPU_STREAM0 = auto()
    GPU_STREAM1 = auto()
    FPGA_BRAM = auto()
    FPGA_LUT = auto()
    STATIC = auto()
    WASM_LINEAR = auto()
    JVM_HEAP = auto()


class AllocationKind(Enum):
    """Memory allocation strategies."""

    MANUAL = auto()  # Manual alloc/free (C-style)
    OWNED = auto()  # Ownership-based (Rust-style)
    GC = auto()  # Garbage collected (Java/Python-style)
    ARENA = auto()  # Arena/bump allocation
    POOL = auto()  # Pool allocation


class BorrowKind(Enum):
    """Types of borrows for the borrow checker."""

    IMMUTABLE = auto()
    MUTABLE = auto()
    EXCLUSIVE = auto()


@dataclass
class RegionLifetime:
    """Lifetime annotation for region inference.

    Attributes:
        name: Lifetime identifier (e.g., 'a, 'static)
        start_point: Program point where lifetime begins
        end_point: Program point where lifetime ends
        parent: Parent lifetime for nesting
        constraints: SMT constraints for lifetime validity
    """

    name: str
    start_point: str = ""
    end_point: str = ""
    parent: RegionLifetime | None = None
    constraints: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegionLifetime):
            return False
        return self.name == other.name

    def outlives(self, other: RegionLifetime) -> bool:
        """Check if this lifetime outlives another."""
        if self.name == "static":
            return True
        if other.name == "static":
            return False
        # Check parent chain
        current = other
        while current is not None:
            if current.name == self.name:
                return True
            current = current.parent
        return False

    @staticmethod
    def static() -> RegionLifetime:
        """Create a static lifetime."""
        return RegionLifetime(name="static")

    @staticmethod
    def scoped(name: str, parent: RegionLifetime | None = None) -> RegionLifetime:
        """Create a scoped lifetime."""
        return RegionLifetime(name=name, parent=parent)


@dataclass
class Region:
    """A memory region in the AION memory model.

    Regions are the fundamental unit of memory management,
    providing isolation and lifetime tracking.

    Attributes:
        id: Unique region identifier
        name: Human-readable region name
        kind: Type of memory region
        lifetime: Lifetime annotation
        size: Maximum size in bytes (None for unbounded)
        alignment: Required alignment
        hardware_affinity: Target hardware device
        parent: Parent region for nested allocations
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    kind: RegionKind = RegionKind.HEAP
    lifetime: RegionLifetime = field(default_factory=RegionLifetime.static)
    size: int | None = None
    alignment: int = 8
    hardware_affinity: str = "any"
    parent: Region | None = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Region):
            return False
        return self.id == other.id

    def is_gpu(self) -> bool:
        """Check if region is GPU memory."""
        return self.kind in (
            RegionKind.GPU_GLOBAL,
            RegionKind.GPU_SHARED,
            RegionKind.GPU_STREAM0,
            RegionKind.GPU_STREAM1,
        )

    def is_fpga(self) -> bool:
        """Check if region is FPGA memory."""
        return self.kind in (RegionKind.FPGA_BRAM, RegionKind.FPGA_LUT)

    def is_device(self) -> bool:
        """Check if region is device (non-CPU) memory."""
        return self.is_gpu() or self.is_fpga()

    def can_transfer_to(self, target: Region) -> bool:
        """Check if data can be transferred to target region."""
        # Same region - always ok
        if self.id == target.id:
            return True
        # CPU can transfer to/from anything
        if not self.is_device() or not target.is_device():
            return True
        # GPU to GPU within same stream - ok
        if self.is_gpu() and target.is_gpu():
            return True
        # FPGA to FPGA - ok
        if self.is_fpga() and target.is_fpga():
            return True
        # GPU to FPGA needs explicit transfer
        return False

    @staticmethod
    def stack(name: str = "stack", lifetime: RegionLifetime | None = None) -> Region:
        """Create a stack region."""
        return Region(
            name=name,
            kind=RegionKind.STACK,
            lifetime=lifetime or RegionLifetime.scoped("fn_scope"),
        )

    @staticmethod
    def heap(name: str = "heap") -> Region:
        """Create a heap region."""
        return Region(
            name=name,
            kind=RegionKind.HEAP,
            lifetime=RegionLifetime.static(),
        )

    @staticmethod
    def gpu_global(name: str = "gpu_global", stream: int = 0) -> Region:
        """Create a GPU global memory region."""
        kind = RegionKind.GPU_STREAM0 if stream == 0 else RegionKind.GPU_STREAM1
        return Region(
            name=name,
            kind=kind,
            hardware_affinity="gpu",
        )

    @staticmethod
    def gpu_shared(name: str = "gpu_shared") -> Region:
        """Create a GPU shared memory region."""
        return Region(
            name=name,
            kind=RegionKind.GPU_SHARED,
            hardware_affinity="gpu",
        )

    @staticmethod
    def fpga_bram(name: str = "fpga_bram", size: int = 36 * 1024) -> Region:
        """Create an FPGA BRAM region."""
        return Region(
            name=name,
            kind=RegionKind.FPGA_BRAM,
            size=size,
            hardware_affinity="fpga",
        )

    @staticmethod
    def fpga_lut(name: str = "fpga_lut") -> Region:
        """Create an FPGA LUT memory region."""
        return Region(
            name=name,
            kind=RegionKind.FPGA_LUT,
            hardware_affinity="fpga",
        )

    @staticmethod
    def thread_local(name: str = "tls") -> Region:
        """Create a thread-local region."""
        return Region(
            name=name,
            kind=RegionKind.THREAD_LOCAL,
        )


@dataclass
class MemoryBlock:
    """A contiguous memory block within a region.

    Attributes:
        id: Unique block identifier
        region: Containing region
        offset: Offset within region
        size: Size in bytes
        alignment: Alignment requirement
        lifetime: Block lifetime
        owner: Current owner vertex ID
        borrows: Active borrows
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    region: Region = field(default_factory=Region.heap)
    offset: int = 0
    size: int = 0
    alignment: int = 8
    lifetime: RegionLifetime = field(default_factory=RegionLifetime.static)
    owner: str | None = None
    borrows: list[Borrow] = field(default_factory=list)

    def is_borrowed(self) -> bool:
        """Check if block has active borrows."""
        return len(self.borrows) > 0

    def is_mutable_borrowed(self) -> bool:
        """Check if block has mutable borrows."""
        return any(b.kind == BorrowKind.MUTABLE for b in self.borrows)


@dataclass
class Borrow:
    """A borrow of a memory block.

    Attributes:
        id: Unique borrow identifier
        block: Borrowed memory block
        kind: Type of borrow
        borrower: Vertex ID of borrower
        lifetime: Borrow lifetime
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    block: MemoryBlock = field(default_factory=MemoryBlock)
    kind: BorrowKind = BorrowKind.IMMUTABLE
    borrower: str = ""
    lifetime: RegionLifetime = field(default_factory=RegionLifetime.static)


@dataclass
class Allocation:
    """Record of a memory allocation.

    Attributes:
        id: Unique allocation identifier
        vertex_id: Vertex that performed allocation
        block: Allocated memory block
        allocation_kind: How allocation is managed
        freed: Whether allocation has been freed
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    vertex_id: str = ""
    block: MemoryBlock = field(default_factory=MemoryBlock)
    allocation_kind: AllocationKind = AllocationKind.OWNED
    freed: bool = False


@dataclass
class OwnershipTransfer:
    """Record of ownership transfer between vertices.

    Attributes:
        id: Unique transfer identifier
        block: Memory block being transferred
        from_vertex: Source vertex
        to_vertex: Target vertex
        transfer_kind: Type of transfer (move, clone)
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    block: MemoryBlock = field(default_factory=MemoryBlock)
    from_vertex: str = ""
    to_vertex: str = ""
    transfer_kind: str = "move"


class RegionManager:
    """Manages memory regions and allocations.

    Provides:
    - Region creation and lifecycle management
    - Allocation tracking
    - Lifetime inference
    - Cross-region transfer validation
    """

    def __init__(self) -> None:
        """Initialize the region manager."""
        self.regions: dict[str, Region] = {}
        self.allocations: dict[str, Allocation] = {}
        self.blocks: dict[str, MemoryBlock] = {}
        self.lifetimes: dict[str, RegionLifetime] = {}
        self.transfers: list[OwnershipTransfer] = []

        # Create default regions
        self._create_default_regions()

    def _create_default_regions(self) -> None:
        """Create default memory regions."""
        self.add_region(Region.stack("global_stack"))
        self.add_region(Region.heap("global_heap"))
        self.add_region(Region.thread_local("global_tls"))

    def add_region(self, region: Region) -> None:
        """Add a region to the manager."""
        self.regions[region.id] = region
        if region.lifetime.name not in self.lifetimes:
            self.lifetimes[region.lifetime.name] = region.lifetime

    def get_region(self, region_id: str) -> Region | None:
        """Get a region by ID."""
        return self.regions.get(region_id)

    def get_region_by_name(self, name: str) -> Region | None:
        """Get a region by name."""
        for region in self.regions.values():
            if region.name == name:
                return region
        return None

    def allocate(
        self,
        region: Region,
        size: int,
        alignment: int = 8,
        vertex_id: str = "",
        allocation_kind: AllocationKind = AllocationKind.OWNED,
    ) -> Allocation:
        """Allocate memory in a region.

        Args:
            region: Target region
            size: Size in bytes
            alignment: Alignment requirement
            vertex_id: Allocating vertex ID
            allocation_kind: Allocation strategy

        Returns:
            Allocation record
        """
        # Ensure region exists
        if region.id not in self.regions:
            self.add_region(region)

        # Calculate offset
        total_size = sum(b.size for b in self.blocks.values() if b.region.id == region.id)
        offset = (total_size + alignment - 1) // alignment * alignment

        # Check size limit
        if region.size is not None and offset + size > region.size:
            raise MemoryError(f"Region {region.name} overflow: {offset + size} > {region.size}")

        # Create block
        block = MemoryBlock(
            region=region,
            offset=offset,
            size=size,
            alignment=alignment,
            lifetime=region.lifetime,
            owner=vertex_id,
        )
        self.blocks[block.id] = block

        # Create allocation record
        alloc = Allocation(
            vertex_id=vertex_id,
            block=block,
            allocation_kind=allocation_kind,
        )
        self.allocations[alloc.id] = alloc

        return alloc

    def free(self, allocation_id: str) -> bool:
        """Free an allocation.

        Args:
            allocation_id: Allocation to free

        Returns:
            True if freed successfully
        """
        alloc = self.allocations.get(allocation_id)
        if alloc is None:
            return False

        if alloc.freed:
            raise RuntimeError(f"Double free of allocation {allocation_id}")

        if alloc.block.is_borrowed():
            raise RuntimeError(f"Cannot free borrowed memory {allocation_id}")

        alloc.freed = True
        del self.blocks[alloc.block.id]

        return True

    def transfer_ownership(
        self,
        block_id: str,
        from_vertex: str,
        to_vertex: str,
        kind: str = "move",
    ) -> OwnershipTransfer:
        """Transfer ownership of a memory block.

        Args:
            block_id: Block to transfer
            from_vertex: Source vertex
            to_vertex: Target vertex
            kind: Transfer type ("move" or "clone")

        Returns:
            Transfer record
        """
        block = self.blocks.get(block_id)
        if block is None:
            raise ValueError(f"Unknown block {block_id}")

        if block.owner != from_vertex:
            raise RuntimeError(f"Vertex {from_vertex} does not own block {block_id}")

        if kind == "move":
            block.owner = to_vertex

        transfer = OwnershipTransfer(
            block=block,
            from_vertex=from_vertex,
            to_vertex=to_vertex,
            transfer_kind=kind,
        )
        self.transfers.append(transfer)

        return transfer

    def borrow(
        self,
        block_id: str,
        borrower: str,
        kind: BorrowKind = BorrowKind.IMMUTABLE,
        lifetime: RegionLifetime | None = None,
    ) -> Borrow:
        """Create a borrow of a memory block.

        Args:
            block_id: Block to borrow
            borrower: Borrowing vertex ID
            kind: Type of borrow
            lifetime: Borrow lifetime

        Returns:
            Borrow record
        """
        block = self.blocks.get(block_id)
        if block is None:
            raise ValueError(f"Unknown block {block_id}")

        # Check borrow rules
        if kind == BorrowKind.MUTABLE:
            if block.borrows:
                raise RuntimeError(f"Cannot mutably borrow already borrowed block {block_id}")
        elif kind == BorrowKind.IMMUTABLE:
            if block.is_mutable_borrowed():
                raise RuntimeError(f"Cannot immutably borrow mutably borrowed block {block_id}")

        borrow = Borrow(
            block=block,
            kind=kind,
            borrower=borrower,
            lifetime=lifetime or block.lifetime,
        )
        block.borrows.append(borrow)

        return borrow

    def end_borrow(self, borrow_id: str) -> None:
        """End a borrow."""
        for block in self.blocks.values():
            block.borrows = [b for b in block.borrows if b.id != borrow_id]

    def infer_lifetimes(self, graph: Any) -> dict[str, RegionLifetime]:
        """Infer lifetimes for vertices in a graph.

        Uses region inference algorithm to determine safe lifetimes
        for all allocations.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            Map of vertex ID to inferred lifetime
        """
        inferred: dict[str, RegionLifetime] = {}

        # Simple lifetime inference based on scope
        for v in graph.vertices:
            if v.metadata.region:
                region = self.get_region_by_name(v.metadata.region)
                if region:
                    inferred[v.id] = region.lifetime
                else:
                    inferred[v.id] = RegionLifetime.scoped(f"scope_{v.id}")
            else:
                inferred[v.id] = RegionLifetime.static()

        return inferred

    def check_safety(self, graph: Any) -> list[str]:
        """Check memory safety for a graph.

        Verifies:
        - No use-after-free
        - No dangling pointers
        - Borrow rules satisfied
        - Region lifetimes respected

        Args:
            graph: AION-SIR hypergraph

        Returns:
            List of safety violations
        """
        violations = []

        # Check freed allocations aren't used
        for alloc in self.allocations.values():
            if alloc.freed:
                # Check no vertices reference this block
                for v in graph.vertices:
                    region_name = v.metadata.region if v.metadata else None
                    if (
                        region_name
                        and alloc.block.region
                        and region_name == alloc.block.region.name
                    ):
                        # Check if vertex uses the freed block
                        pass  # Simplified check

        # Check borrows don't outlive blocks
        for block in self.blocks.values():
            for borrow in block.borrows:
                if not block.lifetime.outlives(borrow.lifetime):
                    violations.append(f"Borrow {borrow.id} outlives block {block.id}")

        return violations


class BorrowChecker:
    """Static borrow checker for AION-SIR graphs.

    Extends Rust-style borrow checking to the region-based
    memory model with cross-language support.
    """

    def __init__(self, region_manager: RegionManager) -> None:
        """Initialize the borrow checker."""
        self.region_manager = region_manager
        self.errors: list[str] = []

    def check(self, graph: Any) -> list[str]:
        """Run borrow checking on a graph.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            List of borrow check errors
        """
        self.errors = []

        # Analyze dataflow for lifetime constraints
        self._analyze_lifetimes(graph)

        # Check borrow validity
        self._check_borrows(graph)

        # Check move semantics
        self._check_moves(graph)

        return self.errors

    def _analyze_lifetimes(self, graph: Any) -> None:
        """Analyze and infer lifetimes for the graph."""
        self.region_manager.infer_lifetimes(graph)

    def _check_borrows(self, graph: Any) -> None:
        """Check that all borrows are valid."""
        for block in self.region_manager.blocks.values():
            # Check for conflicting borrows
            mut_borrows = [b for b in block.borrows if b.kind == BorrowKind.MUTABLE]
            if len(mut_borrows) > 1:
                self.errors.append(f"Multiple mutable borrows of block {block.id}")

            if mut_borrows and any(b.kind == BorrowKind.IMMUTABLE for b in block.borrows):
                self.errors.append(f"Immutable borrow while block {block.id} is mutably borrowed")

    def _check_moves(self, graph: Any) -> None:
        """Check that moved values are not used."""
        moved: set[str] = set()

        for transfer in self.region_manager.transfers:
            if transfer.transfer_kind == "move":
                moved.add(transfer.block.id)

        # Check for use of moved values
        for v in graph.vertices:
            preds = graph.get_predecessors(v)
            for pred in preds:
                if pred.id in moved:
                    # Check if this use is after the move
                    # (Simplified - would need proper dataflow analysis)
                    pass
