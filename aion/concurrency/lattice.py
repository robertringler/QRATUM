"""AION Formal Concurrency Lattice.

Implements effect tracking via a lattice of concurrency effects:
- Bottom (Pure): No side effects
- ThreadSpawn/Join: Thread creation and synchronization
- ChannelSend/Recv: Channel communication
- ActorSend: Actor message passing
- WarpSync: GPU warp synchronization
- PipelineStage: Pipeline parallelism
- Top (Arbitrary): Any effect possible

Function types track capabilities: Fn(..) -> t ! e

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet


class ConcurrencyEffect(Enum):
    """Concurrency effects ordered in the lattice.

    Lower values are more restrictive (fewer effects).
    """

    PURE = 0  # No effects (bottom)
    ALLOC = 1  # Memory allocation
    READ = 2  # Memory read
    WRITE = 3  # Memory write
    THREAD_SPAWN = 10  # Thread creation
    THREAD_JOIN = 11  # Thread join
    CHANNEL_SEND = 20  # Channel send
    CHANNEL_RECV = 21  # Channel receive
    ACTOR_SEND = 30  # Actor message send
    WARP_SYNC = 40  # GPU warp synchronization
    BARRIER = 41  # Thread barrier
    ATOMIC_READ = 50  # Atomic read
    ATOMIC_WRITE = 51  # Atomic write
    ATOMIC_RMW = 52  # Atomic read-modify-write
    PIPELINE_STAGE = 60  # Pipeline stage transition
    IO = 70  # I/O operations
    NETWORK = 71  # Network operations
    GPU_LAUNCH = 80  # GPU kernel launch
    FPGA_PROGRAM = 81  # FPGA programming
    ARBITRARY = 100  # Any effect (top)


# Effect lattice ordering relationships
EFFECT_ORDERING: dict[ConcurrencyEffect, set[ConcurrencyEffect]] = {
    ConcurrencyEffect.PURE: set(),
    ConcurrencyEffect.ALLOC: {ConcurrencyEffect.PURE},
    ConcurrencyEffect.READ: {ConcurrencyEffect.PURE},
    ConcurrencyEffect.WRITE: {ConcurrencyEffect.READ},
    ConcurrencyEffect.THREAD_SPAWN: {ConcurrencyEffect.WRITE},
    ConcurrencyEffect.THREAD_JOIN: {ConcurrencyEffect.THREAD_SPAWN},
    ConcurrencyEffect.CHANNEL_SEND: {ConcurrencyEffect.THREAD_JOIN},
    ConcurrencyEffect.CHANNEL_RECV: {ConcurrencyEffect.CHANNEL_SEND},
    ConcurrencyEffect.ACTOR_SEND: {ConcurrencyEffect.CHANNEL_RECV},
    ConcurrencyEffect.WARP_SYNC: {ConcurrencyEffect.ACTOR_SEND},
    ConcurrencyEffect.BARRIER: {ConcurrencyEffect.WARP_SYNC},
    ConcurrencyEffect.ATOMIC_READ: {ConcurrencyEffect.READ},
    ConcurrencyEffect.ATOMIC_WRITE: {ConcurrencyEffect.ATOMIC_READ, ConcurrencyEffect.WRITE},
    ConcurrencyEffect.ATOMIC_RMW: {ConcurrencyEffect.ATOMIC_WRITE},
    ConcurrencyEffect.PIPELINE_STAGE: {ConcurrencyEffect.BARRIER},
    ConcurrencyEffect.IO: {ConcurrencyEffect.PIPELINE_STAGE},
    ConcurrencyEffect.NETWORK: {ConcurrencyEffect.IO},
    ConcurrencyEffect.GPU_LAUNCH: {ConcurrencyEffect.WARP_SYNC},
    ConcurrencyEffect.FPGA_PROGRAM: {ConcurrencyEffect.IO},
    ConcurrencyEffect.ARBITRARY: {e for e in ConcurrencyEffect if e != ConcurrencyEffect.ARBITRARY},
}


class EffectLattice:
    """The effect lattice for concurrency effect analysis.

    Provides lattice operations:
    - Join (⊔): Least upper bound
    - Meet (⊓): Greatest lower bound
    - Order (⊑): Effect ordering
    """

    @staticmethod
    def bottom() -> ConcurrencyEffect:
        """Return the bottom element (Pure)."""
        return ConcurrencyEffect.PURE

    @staticmethod
    def top() -> ConcurrencyEffect:
        """Return the top element (Arbitrary)."""
        return ConcurrencyEffect.ARBITRARY

    @staticmethod
    def leq(a: ConcurrencyEffect, b: ConcurrencyEffect) -> bool:
        """Check if a ⊑ b (a is less than or equal to b in the lattice).

        Args:
            a: First effect
            b: Second effect

        Returns:
            True if a ⊑ b
        """
        if a == b:
            return True
        if b == ConcurrencyEffect.ARBITRARY:
            return True
        if a == ConcurrencyEffect.PURE:
            return True

        # Check transitive ordering via BFS - can we reach b from a going upward?
        visited = set()
        queue = [a]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for higher in EFFECT_ORDERING.get(current, set()):
                if higher == b:
                    return True  # Found path from a to b going upward, so a ⊑ b
                queue.append(higher)

        # Use value comparison as fallback
        return a.value <= b.value

    @staticmethod
    def join(a: ConcurrencyEffect, b: ConcurrencyEffect) -> ConcurrencyEffect:
        """Compute a ⊔ b (least upper bound).

        Args:
            a: First effect
            b: Second effect

        Returns:
            Least upper bound of a and b
        """
        if EffectLattice.leq(a, b):
            return b
        if EffectLattice.leq(b, a):
            return a

        # Find minimal common upper bound
        # For simplicity, use max by value
        return a if a.value > b.value else b

    @staticmethod
    def meet(a: ConcurrencyEffect, b: ConcurrencyEffect) -> ConcurrencyEffect:
        """Compute a ⊓ b (greatest lower bound).

        Args:
            a: First effect
            b: Second effect

        Returns:
            Greatest lower bound of a and b
        """
        if EffectLattice.leq(a, b):
            return a
        if EffectLattice.leq(b, a):
            return b

        # Find maximal common lower bound
        return a if a.value < b.value else b

    @staticmethod
    def join_all(effects: set[ConcurrencyEffect]) -> ConcurrencyEffect:
        """Compute join of all effects in a set."""
        if not effects:
            return EffectLattice.bottom()

        result = EffectLattice.bottom()
        for e in effects:
            result = EffectLattice.join(result, e)
        return result


@dataclass(frozen=True)
class EffectCapability:
    """A capability for performing certain effects.

    Capabilities are tracked in function types and can be
    passed, split, or combined.

    Attributes:
        effect: The effect this capability grants
        region: Optional region the capability applies to
        exclusive: Whether this is an exclusive capability
    """

    effect: ConcurrencyEffect
    region: str | None = None
    exclusive: bool = False

    def can_perform(self, effect: ConcurrencyEffect) -> bool:
        """Check if this capability allows performing an effect."""
        return EffectLattice.leq(effect, self.effect)

    def combine(self, other: EffectCapability) -> EffectCapability:
        """Combine two capabilities."""
        return EffectCapability(
            effect=EffectLattice.join(self.effect, other.effect),
            region=self.region if self.region == other.region else None,
            exclusive=self.exclusive and other.exclusive,
        )


@dataclass
class FunctionEffect:
    """Effect specification for a function type.

    Represents: Fn(params) -> return_type ! effects

    Attributes:
        effects: Set of effects the function may perform
        required_caps: Required capabilities to call
        granted_caps: Capabilities granted by the function
        pure: Whether the function is effect-free
    """

    effects: FrozenSet[ConcurrencyEffect] = field(default_factory=frozenset)
    required_caps: FrozenSet[EffectCapability] = field(default_factory=frozenset)
    granted_caps: FrozenSet[EffectCapability] = field(default_factory=frozenset)
    pure: bool = False

    def __post_init__(self) -> None:
        """Initialize pure flag based on effects."""
        object.__setattr__(
            self,
            "pure",
            len(self.effects) == 0 or self.effects == frozenset({ConcurrencyEffect.PURE}),
        )

    @staticmethod
    def pure_fn() -> FunctionEffect:
        """Create a pure function effect."""
        return FunctionEffect(
            effects=frozenset({ConcurrencyEffect.PURE}),
            pure=True,
        )

    @staticmethod
    def io_fn() -> FunctionEffect:
        """Create an I/O function effect."""
        return FunctionEffect(
            effects=frozenset({ConcurrencyEffect.IO}),
        )

    @staticmethod
    def concurrent_fn(effects: set[ConcurrencyEffect]) -> FunctionEffect:
        """Create a function with concurrent effects."""
        return FunctionEffect(effects=frozenset(effects))

    def join(self, other: FunctionEffect) -> FunctionEffect:
        """Join two function effects (for composition)."""
        return FunctionEffect(
            effects=self.effects | other.effects,
            required_caps=self.required_caps | other.required_caps,
            granted_caps=self.granted_caps & other.granted_caps,
        )

    def can_call_with(self, caps: set[EffectCapability]) -> bool:
        """Check if function can be called with given capabilities."""
        return all(any(c.can_perform(req.effect) for c in caps) for req in self.required_caps)

    def serialize(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "effects": [e.name for e in self.effects],
            "required_caps": [
                {"effect": c.effect.name, "region": c.region} for c in self.required_caps
            ],
            "granted_caps": [
                {"effect": c.effect.name, "region": c.region} for c in self.granted_caps
            ],
            "pure": self.pure,
        }


class EffectChecker:
    """Static effect checker for AION-SIR graphs.

    Verifies that:
    - All effects are properly annotated
    - Required capabilities are available
    - Effect ordering constraints are satisfied
    - Race conditions are prevented
    """

    def __init__(self) -> None:
        """Initialize the effect checker."""
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, graph: Any) -> tuple[list[str], list[str]]:
        """Run effect checking on a graph.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            Tuple of (errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Collect vertex effects
        self._check_effect_annotations(graph)

        # Check effect ordering
        self._check_effect_ordering(graph)

        # Check for races
        self._check_race_conditions(graph)

        # Check capability flow
        self._check_capability_flow(graph)

        return self.errors, self.warnings

    def _check_effect_annotations(self, graph: Any) -> None:
        """Verify all vertices have valid effect annotations."""
        for v in graph.vertices:
            effects = v.metadata.effects
            if not effects:
                self.warnings.append(f"Vertex {v.id} has no effect annotation")

    def _check_effect_ordering(self, graph: Any) -> None:
        """Check that effect edges enforce correct ordering."""
        effect_edges = graph.get_effect_edges()

        for edge in effect_edges:
            for src in edge.sources:
                for tgt in edge.targets:
                    src_effects = src.metadata.effects
                    tgt_effects = tgt.metadata.effects

                    # Effects with side effects must be ordered
                    if (
                        ConcurrencyEffect.WRITE in src_effects
                        and ConcurrencyEffect.WRITE in tgt_effects
                    ):
                        if edge.attributes.get("ordering") != "seq":
                            self.warnings.append(
                                f"Write-write conflict between {src.id} and {tgt.id} may need ordering"
                            )

    def _check_race_conditions(self, graph: Any) -> None:
        """Check for potential data races."""
        # Find parallel regions
        parallel_edges = graph.get_parallel_edges()

        for edge in parallel_edges:
            vertices = edge.targets

            # Check for conflicting memory accesses
            writers = [v for v in vertices if ConcurrencyEffect.WRITE in v.metadata.effects]

            if len(writers) > 1:
                # Check if they access the same region
                regions = {v.metadata.region for v in writers}
                if len(regions) == 1 and None not in regions:
                    self.errors.append(
                        f"Potential race: multiple writers to region {regions.pop()} "
                        f"in parallel region"
                    )

    def _check_capability_flow(self, graph: Any) -> None:
        """Check that capabilities flow correctly through the graph."""
        # Track available capabilities at each vertex
        available_caps: dict[str, set[EffectCapability]] = {}

        for v in graph.topological_order():
            # Get capabilities from predecessors
            preds = graph.get_predecessors(v)
            caps = set()
            for p in preds:
                caps.update(available_caps.get(p.id, set()))

            # Check if vertex effects are allowed
            for effect in v.metadata.effects:
                if effect not in (ConcurrencyEffect.PURE, ConcurrencyEffect.READ):
                    if not any(c.can_perform(effect) for c in caps):
                        # No capability - check if it's a root capability
                        # (kernels and entry points can introduce capabilities)
                        if v.vertex_type.name not in ("KERNEL_LAUNCH", "PARAMETER"):
                            self.warnings.append(
                                f"Vertex {v.id} performs {effect.name} without capability"
                            )

            # Grant capabilities from this vertex
            available_caps[v.id] = caps


@dataclass
class RaceAnalysis:
    """Result of race condition analysis."""

    has_races: bool = False
    race_pairs: list[tuple[str, str]] = field(default_factory=list)
    safe_parallel_regions: list[set[str]] = field(default_factory=list)


def analyze_races(graph: Any) -> RaceAnalysis:
    """Perform detailed race condition analysis.

    Args:
        graph: AION-SIR hypergraph

    Returns:
        RaceAnalysis with race information
    """
    analysis = RaceAnalysis()

    # Find all memory accesses
    reads: dict[str, list[Any]] = {}  # region -> vertices
    writes: dict[str, list[Any]] = {}

    from ..sir.vertices import EffectKind

    for v in graph.vertices:
        region = v.metadata.region or "heap"

        if EffectKind.READ in v.metadata.effects:
            reads.setdefault(region, []).append(v)
        if EffectKind.WRITE in v.metadata.effects:
            writes.setdefault(region, []).append(v)

    # Check for conflicts in parallel regions
    parallel_edges = graph.get_parallel_edges()

    for edge in parallel_edges:
        parallel_vertices = set(v.id for v in edge.targets)

        for region in writes:
            region_writes = [v for v in writes[region] if v.id in parallel_vertices]
            region_reads = [v for v in reads.get(region, []) if v.id in parallel_vertices]

            # Write-write conflicts
            if len(region_writes) > 1:
                analysis.has_races = True
                for i, w1 in enumerate(region_writes):
                    for w2 in region_writes[i + 1 :]:
                        analysis.race_pairs.append((w1.id, w2.id))

            # Write-read conflicts
            for w in region_writes:
                for r in region_reads:
                    if w.id != r.id:
                        # Check if there's an ordering edge
                        has_ordering = any(
                            (w in e.sources and r in e.targets)
                            or (r in e.sources and w in e.targets)
                            for e in graph.get_effect_edges()
                        )
                        if not has_ordering:
                            analysis.has_races = True
                            analysis.race_pairs.append((w.id, r.id))

    return analysis


@dataclass
class DeadlockAnalysis:
    """Result of deadlock analysis."""

    has_deadlock: bool = False
    cycles: list[list[str]] = field(default_factory=list)


def analyze_deadlocks(graph: Any) -> DeadlockAnalysis:
    """Analyze potential deadlocks in synchronization.

    Args:
        graph: AION-SIR hypergraph

    Returns:
        DeadlockAnalysis with deadlock information
    """
    analysis = DeadlockAnalysis()

    # Build lock ordering graph
    # Edges go from lock A to lock B if A is held while acquiring B
    lock_graph: dict[str, set[str]] = {}

    # Find synchronization vertices
    sync_vertices = [
        v
        for v in graph.vertices
        if any(
            e in v.metadata.effects
            for e in (
                ConcurrencyEffect.THREAD_JOIN,
                ConcurrencyEffect.CHANNEL_RECV,
                ConcurrencyEffect.BARRIER,
                ConcurrencyEffect.ATOMIC_RMW,
            )
        )
    ]

    # Build dependency graph based on control flow
    for v in sync_vertices:
        v_id = v.id
        lock_graph[v_id] = set()

        # Find other sync operations reachable from this one
        visited = set()
        queue = [s for s in graph.get_successors(v)]

        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)

            if current in sync_vertices and current.id != v_id:
                lock_graph[v_id].add(current.id)

            queue.extend(graph.get_successors(current))

    # Detect cycles using DFS
    def find_cycle(start: str, path: list[str], visited: set[str]) -> list[str] | None:
        if start in path:
            return path[path.index(start) :]
        if start in visited:
            return None

        visited.add(start)
        path.append(start)

        for neighbor in lock_graph.get(start, set()):
            cycle = find_cycle(neighbor, path.copy(), visited)
            if cycle:
                return cycle

        return None

    visited: set[str] = set()
    for v_id in lock_graph:
        cycle = find_cycle(v_id, [], visited)
        if cycle:
            analysis.has_deadlock = True
            analysis.cycles.append(cycle)

    return analysis
