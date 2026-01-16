# QRATUM Temporal Computing API Reference

Complete API documentation for the temporal computing framework.

## Core Classes

### TemporalEngine

Main engine for temporal computations.

```python
class TemporalEngine:
    def __init__(
        self,
        peak_flops: float = 2.5e18,
        deterministic: bool = True,
        merkle_verified: bool = True,
        paradox_resolution: str = "novikov",
        temporal_resolution: float = 1e-9,
        enable_pqc: bool = False,
    )
```

**Parameters:**
- `peak_flops`: Peak computational performance in FLOPS
- `deterministic`: Enable deterministic execution
- `merkle_verified`: Enable Merkle verification
- `paradox_resolution`: Strategy ("novikov", "many_worlds", "chronology_protection")
- `temporal_resolution`: Minimum time step in seconds
- `enable_pqc`: Enable post-quantum cryptography

#### forward()
```python
def forward(
    self,
    initial_state: Any,
    delta_t: float,
    evolution_fn: Callable[[Any, float], Any],
    timeline_id: Optional[str] = None,
    num_steps: Optional[int] = None,
) -> Tuple[TemporalState, TemporalProof]
```

Travel forward in computational time.

**Returns:** `(final_state, proof)`

#### backward()
```python
def backward(
    self,
    final_state: Any,
    delta_t: float,
    inverse_fn: Callable[[Any, float], Any],
    timeline_id: Optional[str] = None,
    num_steps: Optional[int] = None,
) -> Tuple[TemporalState, TemporalProof]
```

Travel backward in computational time.

**Returns:** `(initial_state, proof)`

#### branch()
```python
def branch(
    self,
    branch_point: Any,
    branches: List[Tuple[str, Callable[[Any], Any]]],
    evolution_fn: Callable[[Any, float], Any],
    delta_t: float,
    parent_timeline_id: Optional[str] = None,
) -> Dict[str, Tuple[TemporalState, TemporalProof]]
```

Create parallel timeline branches.

**Returns:** Dictionary mapping branch names to `(final_state, proof)`

#### converge()
```python
def converge(
    self,
    branches: Dict[str, TemporalState],
    convergence_fn: Callable[[List[Any]], Any],
    timeline_id: Optional[str] = None,
) -> Tuple[TemporalState, TemporalProof]
```

Merge parallel timelines.

**Returns:** `(converged_state, proof)`

---

### FTLComputation

Framework for faster-than-light computational paradigms.

```python
class FTLComputation:
    def __init__(self, peak_flops: float = 2.5e18)
```

#### compress_time()
```python
def compress_time(
    self,
    initial_state: Any,
    evolution_fn: Callable[[Any, float], Any],
    delta_t: float,
    compression_target: float = 1e6,
) -> Tuple[TemporalState, EffectiveVelocity]
```

Simulate time faster than real-time.

**Returns:** `(final_state, velocity_metrics)`

#### inverse_causality()
```python
def inverse_causality(
    self,
    desired_outcome: Any,
    inverse_fn: Callable[[Any, float], Any],
    delta_t: float,
) -> Tuple[TemporalState, EffectiveVelocity]
```

Compute backward from desired outcome.

**Returns:** `(initial_state, velocity_metrics)`

#### parallel_reality_search()
```python
def parallel_reality_search(
    self,
    branch_point: Any,
    branch_count: int,
    evolution_fn: Callable[[Any, float], Any],
    mutation_fn: Callable[[Any], Any],
    target_fn: Callable[[Any], float],
    duration: float,
) -> Tuple[TemporalState, str, EffectiveVelocity]
```

Search across parallel realities for optimal outcome.

**Returns:** `(best_state, best_branch_name, velocity_metrics)`

---

### PossibilityOracle

Pre-computed possibility spaces for instant lookup.

```python
class PossibilityOracle:
    def __init__(self, max_workers: int = 4)
```

#### precompute_all()
```python
def precompute_all(
    self,
    initial_states: List[Any],
    evolution_fn: Callable[[Any, float], Any],
    delta_t: float,
    space_id: Optional[str] = None,
) -> Tuple[PossibilitySpace, EffectiveVelocity]
```

Precompute all possibilities.

**Returns:** `(possibility_space, velocity_metrics)`

#### query()
```python
def query(
    self,
    state_key: str,
    space_id: str,
) -> Tuple[Optional[Any], EffectiveVelocity]
```

Instant lookup from precomputed space.

**Returns:** `(result, velocity_metrics)` (∞c velocity)

---

### TimelineManager

Manage multiverse of computed timelines.

```python
class TimelineManager:
    def __init__(self, enable_verification: bool = True)
```

#### create_timeline()
```python
def create_timeline(
    self,
    timeline_id: Optional[str] = None,
    parent_branch: Optional[TimelineBranch] = None,
) -> Timeline
```

Create new timeline.

#### branch_timeline()
```python
def branch_timeline(
    self,
    parent_id: str,
    branch_names: List[str],
    branch_point_depth: int,
    reason: str = "manual_branch",
) -> Dict[str, Timeline]
```

Branch timeline into children.

**Returns:** Dictionary mapping branch names to Timeline objects

#### merge_timelines()
```python
def merge_timelines(
    self,
    timeline_ids: List[str],
    merge_fn: Callable[[List[TemporalState]], Any],
    merged_timeline_id: Optional[str] = None,
) -> Timeline
```

Merge multiple timelines.

#### find_optimal_timeline()
```python
def find_optimal_timeline(
    self,
    objective_fn: Callable[[Timeline], float],
    timeline_ids: Optional[List[str]] = None,
) -> Tuple[Optional[Timeline], float]
```

Find optimal timeline by objective function.

**Returns:** `(best_timeline, best_score)`

---

### ParadoxResolver

Resolve temporal paradoxes.

```python
class ParadoxResolver:
    def __init__(self, strategy: str = "novikov")
```

**Strategies:**
- `"novikov"`: Self-consistency principle
- `"many_worlds"`: Branch on paradox
- `"chronology_protection"`: Reject paradox

#### check_and_resolve()
```python
def check_and_resolve(
    self,
    chain: StateChain,
    causality_fn: Optional[Callable[[Any, Any], bool]] = None,
) -> Tuple[bool, StateChain, List[ParadoxDetection]]
```

Check for paradoxes and resolve them.

**Returns:** `(is_valid, resolved_chain, detected_paradoxes)`

---

### CausalityValidator

Validate causal consistency.

```python
class CausalityValidator:
    def __init__(self)
```

#### validate_causal_chain()
```python
def validate_causal_chain(
    self,
    chain: StateChain,
    causality_fn: Optional[Callable[[Any, Any], bool]] = None,
) -> Tuple[bool, List[ParadoxDetection]]
```

Validate entire state chain.

**Returns:** `(is_valid, list_of_paradoxes)`

---

## Data Classes

### TemporalState

State at a point in computational time.

```python
@dataclass
class TemporalState:
    coordinate: TemporalCoordinate
    data: Any
    parent_hash: Optional[str] = None
    merkle_proof: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_hash() -> str
    def verify_hash() -> bool
    def serialize() -> bytes
    @classmethod
    def deserialize(data: bytes) -> TemporalState
```

### TemporalCoordinate

Position in computational spacetime.

```python
@dataclass
class TemporalCoordinate:
    state_hash: str
    timeline_id: str
    computational_t: float
    physical_t: float
    depth: int = 0
    
    def effective_velocity(other: TemporalCoordinate) -> float
```

### EffectiveVelocity

Metrics for computational FTL.

```python
@dataclass
class EffectiveVelocity:
    mechanism: str
    computational_delta_t: float
    physical_delta_t: float
    effective_c_multiple: float
    operations_executed: int = 0
    distance_equivalent: float = 0.0
    
    def is_ftl() -> bool
```

### TemporalProof

Cryptographic proof of temporal computation.

```python
@dataclass
class TemporalProof:
    initial_state_hash: str
    final_state_hash: str
    merkle_root: str
    operation: str
    timeline_id: str
    computational_delta_t: float
    physical_delta_t: float
    intermediate_hashes: List[str]
    signature: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None
```

---

## Constants

### Physical Constants
```python
SPEED_OF_LIGHT = 299_792_458  # m/s
PLANCK_TIME = 5.391247e-44     # seconds
PLANCK_LENGTH = 1.616255e-35   # meters
```

### Time Constants
```python
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 31_557_600
```

### QRATUM Constants
```python
QRATUM_PEAK_FLOPS = 2.5e18
OPERATIONS_PER_LIGHT_METER = 8.3e9
TARGET_COMPRESSION_RATIO = 1e9
```

### Strategies
```python
class ParadoxStrategy:
    NOVIKOV = "novikov"
    MANY_WORLDS = "many_worlds"
    CHRONOLOGY_PROTECTION = "chronology_protection"

class EntropyStrategy:
    RECONSTRUCT = "reconstruct"
    PROBABILISTIC = "probabilistic"
    MERKLE_RECOVER = "merkle_recover"

class FTLMechanism:
    PRECOMPUTATION = "precomputation"
    PREDICTIVE_MODELING = "predictive_modeling"
    PARALLEL_BRANCHING = "parallel_branching"
    TEMPORAL_COMPRESSION = "temporal_compression"
    INVERSE_CAUSALITY = "inverse_causality"
```

---

## Error Handling

### Common Exceptions

```python
ValueError: 
    - Invalid timeline ID
    - Paradoxical state transition
    - Invalid branch point

RuntimeError:
    - Evolution function failed
    - Merkle verification failed
    - State chain corruption

TimeoutError:
    - Computation exceeded time limit
```

### Best Practices

1. Always wrap temporal operations in try-except
2. Verify proofs after critical operations
3. Use checksums for large state objects
4. Enable deterministic mode for reproducibility
5. Periodically prune unused timelines
