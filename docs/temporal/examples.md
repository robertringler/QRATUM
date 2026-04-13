# QRATUM Temporal Computing Examples

Practical examples demonstrating the temporal computing framework.

## Example 1: Climate Prediction

Simulate 100 years of climate evolution in seconds.

```python
from qratum.temporal import TemporalEngine, SECONDS_PER_YEAR
import numpy as np

# Initialize engine
engine = TemporalEngine(
    peak_flops=2.5e18,
    deterministic=True,
    merkle_verified=True,
)

# Define climate state
class ClimateState:
    def __init__(self, temp, co2, ice_volume):
        self.temperature = temp
        self.co2_ppm = co2
        self.ice_volume = ice_volume

# Evolution function
def evolve_climate(state, delta_t):
    years = delta_t / SECONDS_PER_YEAR
    
    # Simple climate model
    new_temp = state.temperature + 0.02 * years
    new_co2 = state.co2_ppm + 2.5 * years
    new_ice = state.ice_volume - 0.1 * years
    
    return ClimateState(new_temp, new_co2, max(0, new_ice))

# Current climate
current = ClimateState(temp=15.0, co2=420, ice_volume=100)

# Predict 100 years forward
future, proof = engine.forward(
    initial_state=current,
    delta_t=100 * SECONDS_PER_YEAR,
    evolution_fn=evolve_climate,
)

print(f"Temperature in 2126: {future.data.temperature:.1f}°C")
print(f"CO2 in 2126: {future.data.co2_ppm:.0f} ppm")
print(f"Effective velocity: {proof.metadata['effective_velocity_c']:.2e}c")
```

## Example 2: Policy Analysis with Timeline Branching

Explore different policy choices and compare outcomes.

```python
from qratum.temporal import TemporalEngine, TimelineManager, SECONDS_PER_YEAR

engine = TemporalEngine(deterministic=True)
manager = TimelineManager()

# Economic state
class EconomyState:
    def __init__(self, gdp, unemployment, debt):
        self.gdp = gdp
        self.unemployment = unemployment
        self.debt = debt

# Evolution function
def evolve_economy(state, delta_t):
    years = delta_t / SECONDS_PER_YEAR
    growth_rate = 0.02
    
    new_gdp = state.gdp * (1 + growth_rate) ** years
    new_unemployment = state.unemployment * 0.98 ** years
    new_debt = state.debt * 1.03 ** years
    
    return EconomyState(new_gdp, new_unemployment, new_debt)

# Current state
current_economy = EconomyState(gdp=100, unemployment=5.0, debt=50)

# Define policy mutations
def apply_tax_cut(state):
    return EconomyState(state.gdp * 1.05, state.unemployment, state.debt * 1.1)

def apply_investment(state):
    return EconomyState(state.gdp * 1.02, state.unemployment * 0.9, state.debt * 1.05)

def apply_austerity(state):
    return EconomyState(state.gdp * 0.98, state.unemployment * 1.1, state.debt * 0.95)

# Explore all policy options
results = engine.branch(
    branch_point=current_economy,
    branches=[
        ("tax_cut", apply_tax_cut),
        ("investment", apply_investment),
        ("austerity", apply_austerity),
    ],
    evolution_fn=evolve_economy,
    delta_t=20 * SECONDS_PER_YEAR,
)

# Compare outcomes
for name, (final_state, proof) in results.items():
    print(f"\n{name.upper()}:")
    print(f"  GDP: {final_state.data.gdp:.2f}")
    print(f"  Unemployment: {final_state.data.unemployment:.2f}%")
    print(f"  Debt: {final_state.data.debt:.2f}")
```

## Example 3: Molecular Dynamics with Backward Reconstruction

Simulate protein folding forward, then reconstruct the unfolding pathway.

```python
from qratum.temporal import TemporalEngine
import numpy as np

engine = TemporalEngine(deterministic=True)

# Protein state (simplified)
class ProteinState:
    def __init__(self, energy, radius_of_gyration):
        self.energy = energy
        self.rg = radius_of_gyration

# Forward dynamics
def molecular_dynamics(state, delta_t):
    # Simulate folding (energy decreases)
    new_energy = state.energy - 0.5 * delta_t
    new_rg = state.rg * np.exp(-0.1 * delta_t)
    return ProteinState(new_energy, new_rg)

# Reverse dynamics
def reverse_dynamics(state, delta_t):
    # Reconstruct unfolding
    new_energy = state.energy + 0.5 * delta_t
    new_rg = state.rg * np.exp(0.1 * delta_t)
    return ProteinState(new_energy, new_rg)

# Start unfolded
unfolded = ProteinState(energy=100.0, rg=50.0)

# Fold for 1 microsecond
folded, proof_fwd = engine.forward(
    initial_state=unfolded,
    delta_t=1e-6,  # 1 microsecond
    evolution_fn=molecular_dynamics,
)

print(f"Folded energy: {folded.data.energy:.2f}")
print(f"Folded Rg: {folded.data.rg:.2f}")

# Reconstruct unfolding pathway
reconstructed, proof_bwd = engine.backward(
    final_state=folded.data,
    delta_t=-1e-6,
    inverse_fn=reverse_dynamics,
)

print(f"Reconstructed energy: {reconstructed.data.energy:.2f}")
print(f"Original energy: {unfolded.energy:.2f}")
```

## Example 4: Optimal Strategy Search

Find the best strategy among thousands of possibilities.

```python
from qratum.temporal import FTLComputation, SECONDS_PER_YEAR
import random

ftl = FTLComputation(peak_flops=2.5e18)

# Game state
class GameState:
    def __init__(self, score, level, resources):
        self.score = score
        self.level = level
        self.resources = resources

# Evolution
def play_game(state, delta_t):
    turns = int(delta_t / 60)  # 1 minute per turn
    new_score = state.score + turns * state.level
    new_level = state.level + turns // 10
    new_resources = max(0, state.resources + turns * 2 - turns * 1)
    return GameState(new_score, new_level, new_resources)

# Strategy mutation
def random_strategy(state):
    action = random.choice(['explore', 'defend', 'attack'])
    if action == 'explore':
        return GameState(state.score, state.level, state.resources + 5)
    elif action == 'defend':
        return GameState(state.score + 2, state.level, state.resources)
    else:  # attack
        return GameState(state.score + 5, state.level + 1, state.resources - 3)

# Target function: maximize score and resources
def score_strategy(state):
    return state.score + state.resources * 2

# Search 1000 parallel strategies
initial_state = GameState(score=0, level=1, resources=10)

best_state, best_strategy, velocity = ftl.parallel_reality_search(
    branch_point=initial_state,
    branch_count=1000,
    evolution_fn=play_game,
    mutation_fn=random_strategy,
    target_fn=score_strategy,
    duration=3600,  # 1 hour of gameplay
)

print(f"Best strategy: {best_strategy}")
print(f"Final score: {best_state.data.score}")
print(f"Final resources: {best_state.data.resources}")
print(f"Explored at {velocity.effective_c_multiple:.2e}c")
```

## Example 5: Paradox Handling

Demonstrate automatic paradox detection and resolution.

```python
from qratum.temporal import (
    TemporalEngine,
    ParadoxResolver,
    ParadoxStrategy,
    StateChain,
)

engine = TemporalEngine(
    paradox_resolution=ParadoxStrategy.MANY_WORLDS,
    deterministic=True,
)

resolver = ParadoxResolver(strategy=ParadoxStrategy.MANY_WORLDS)

# Time travel scenario
class TimelineState:
    def __init__(self, event_occurred):
        self.event_occurred = event_occurred

# This creates a grandfather paradox
def paradoxical_evolution(state, delta_t):
    # Event prevents itself from occurring
    return TimelineState(event_occurred=not state.event_occurred)

# Start with event occurred
initial = TimelineState(event_occurred=True)

# Attempt paradoxical evolution
try:
    final, proof = engine.forward(
        initial_state=initial,
        delta_t=100.0,
        evolution_fn=paradoxical_evolution,
    )
    
    timeline_id = final.coordinate.timeline_id
    chain = engine.get_timeline(timeline_id)
    
    # Check for paradoxes
    is_valid, resolved_chain, paradoxes = resolver.check_and_resolve(chain)
    
    if paradoxes:
        print(f"Detected {len(paradoxes)} paradoxes")
        for p in paradoxes:
            print(f"  - {p.paradox_type.value}: {p.description}")
            print(f"    Resolution: {p.resolution}")
    
    if is_valid:
        print("Timeline resolved successfully")
        print(f"Resolution strategy: {resolver.strategy}")
except Exception as e:
    print(f"Paradox prevented computation: {e}")
```

## Example 6: Using the Possibility Oracle

Precompute all possibilities for instant lookup.

```python
from qratum.temporal import PossibilityOracle

oracle = PossibilityOracle()

# Define discrete states
states = list(range(100))

# Evolution function
def evolve(state, delta_t):
    return (state * 2) % 100

# Precompute all possibilities
space, velocity = oracle.precompute_all(
    initial_states=states,
    evolution_fn=evolve,
    delta_t=10.0,
    space_id="modular_evolution",
)

print(f"Precomputed {space.size()} states")
print(f"Precomputation velocity: {velocity.effective_c_multiple:.2e}c")

# Now instant lookups
for initial in [5, 17, 42, 99]:
    state_key = oracle._hash_state(initial)
    result, query_velocity = oracle.query(
        state_key=state_key,
        space_id="modular_evolution",
    )
    print(f"State {initial} evolves to {result} (∞c lookup)")
```

## Example 7: Distributed Exascale Computation

Run temporal computation across multiple nodes.

```python
from qratum.temporal import (
    TemporalEngineDistributed,
    ExascaleConfig,
)

# Configure for 100 nodes
config = ExascaleConfig(
    enable_qdr=True,
    enable_aetherfabric=True,
    enable_pqc=True,
    num_nodes=100,
    peak_flops_per_node=2.5e18,
)

# Total compute: 250 ExaFLOPS
engine = TemporalEngineDistributed(config=config)

# Define large-scale simulation
class GlobalState:
    def __init__(self, grid):
        self.grid = grid  # Distributed grid

def distributed_evolution(state, delta_t):
    # Evolution distributed across nodes
    # QDR ensures deterministic AllReduce
    # AetherFabric handles routing
    return updated_state

# Run distributed computation
result = engine.forward_distributed(
    initial_state=global_initial_state,
    delta_t=SECONDS_PER_YEAR,
    evolution_fn=distributed_evolution,
)

# Get statistics
stats = engine.get_statistics()
print(f"Total nodes: {stats['distributed']['num_nodes']}")
print(f"Total FLOPS: {stats['distributed']['total_peak_flops']:.2e}")
print(f"QDR syncs: {stats['qdr']['sync_count']}")
print(f"AetherFabric messages: {stats['aetherfabric']['message_count']}")
```

## Performance Tips

1. **Use coarse temporal resolution** for faster computation
2. **Enable checkpointing** for long simulations
3. **Prune unlikely branches** early
4. **Cache evolution functions** when possible
5. **Use precomputation** for repeated queries
6. **Batch state transitions** for better throughput
7. **Profile your evolution function** - it's the bottleneck
8. **Consider approximate methods** for non-critical regions

## Common Patterns

### Pattern: Sensitivity Analysis

```python
# Vary one parameter across branches
results = engine.branch(
    branch_point=base_state,
    branches=[(f"param_{x}", lambda s: mutate_param(s, x)) 
              for x in param_values],
    evolution_fn=evolve,
    delta_t=duration,
)
```

### Pattern: Monte Carlo Temporal Sampling

```python
# Sample many random futures
ftl = FTLComputation()
outcomes = []
for _ in range(1000):
    state, velocity = ftl.compress_time(
        initial_state=perturb(base_state),
        evolution_fn=stochastic_evolve,
        delta_t=duration,
    )
    outcomes.append(state)

# Analyze distribution
mean_outcome = analyze(outcomes)
```

### Pattern: Iterative Refinement

```python
# Coarse simulation first
coarse_result, _ = engine.forward(
    initial_state=state,
    delta_t=duration,
    evolution_fn=coarse_evolution,
    num_steps=100,
)

# Refine interesting regions
fine_result, _ = engine.forward(
    initial_state=coarse_result.data,
    delta_t=duration / 100,
    evolution_fn=fine_evolution,
    num_steps=10000,
)
```
