# XENON Architecture

## Overview

XENON (eXtensible bio-mEchanism simulatioN and visualizatiOn) is a modular system for simulating and visualizing biochemical reaction networks. It integrates with the Qubic visualization pipeline to provide real-time visualization of stochastic simulation algorithms (SSA).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         XENON System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │ Core Module  │         │   Adapters   │                     │
│  │              │         │              │                     │
│  │ BioMechanism │────────▶│ Bio-Mechanism│                     │
│  │ MolecularState│────────▶│   Adapter    │                     │
│  │  Transition  │────────▶│   Adapters   │                     │
│  └──────────────┘         └──────┬───────┘                     │
│                                   │                              │
│                                   │ VisualizationData            │
│                                   ▼                              │
│                          ┌────────────────┐                     │
│                          │ Xenon Pipeline │                     │
│                          │    Adapter     │                     │
│                          └────────┬───────┘                     │
│                                   │                              │
└───────────────────────────────────┼──────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Qubic Visualization Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
│  │ Streaming  │    │  Timeseries│    │   Static   │           │
│  │  Pipeline  │    │  Pipeline  │    │  Pipeline  │           │
│  └─────┬──────┘    └─────┬──────┘    └─────┬──────┘           │
│        │                  │                  │                  │
│        └──────────────────┼──────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │  Backend Renderers     │                         │
│              │  - GPU (CUDA/OpenGL)   │                         │
│              │  - Headless (Server)   │                         │
│              │  - Matplotlib (CPU)    │                         │
│              └────────────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Core Module (`xenon/core/`)

Defines fundamental data structures for bio-mechanism simulation.

#### BioMechanism

- **Purpose**: Represents biochemical reaction network as DAG
- **Components**: List of states, list of transitions, evidence score
- **Methods**: 
  - `get_state(state_id)`: Retrieve state by ID
  - `get_transitions_from(state_id)`: Get outgoing transitions
  - `get_transitions_to(state_id)`: Get incoming transitions

#### MolecularState

- **Purpose**: Represents discrete molecular configuration
- **Properties**: State ID, protein name, free energy, concentration
- **Physics**: Gibbs free energy (ΔG) in kJ/mol

#### Transition

- **Purpose**: Represents state-to-state reaction
- **Kinetics**: Rate constant (k) in s⁻¹
- **Thermodynamics**: ΔG, activation energy
- **Algorithm**: Used in Gillespie SSA for stochastic transitions

### Adapter Layer (`xenon/adapters/`)

Transforms XENON data structures into visualization-ready formats.

#### BioMechanismAdapter

**Input**: `BioMechanism`  
**Output**: `VisualizationData` (3D network)

**Transformation Pipeline**:
1. Extract states → nodes
2. Extract transitions → edges
3. Apply spatial layout algorithm
4. Generate 3D positions
5. Create mesh geometry
6. Map properties to scalar fields

**Layout Algorithms**:
- **Spring**: Force-directed (O(n²))
- **Circular**: Uniform distribution (O(n))
- **Hierarchical**: Level-based (O(n + e))

#### MolecularStateAdapter

**Input**: `MolecularState`  
**Output**: `VisualizationData` (surface or point cloud)

**Representations**:
1. **Energy Surface**: 2D grid with energy as height
2. **Point Cloud**: Stochastic particle representation

**Scalar Fields**:
- `energy`: Per-vertex energy values
- `concentration`: Molecular density

#### TransitionAdapter

**Input**: `Transition`  
**Output**: `VisualizationData` (arrow or curve)

**Visualizations**:
1. **Arrow**: Directional edge between states
2. **Energy Barrier**: Reaction coordinate energy profile

**Styling**:
- Color mapping: Rate constant → RGB (blue=slow, red=fast)
- Thickness: Log-scale rate constant
- Metadata: Source/target, kinetic parameters

### Pipeline Integration (`qubic/visualization/adapters/xenon_adapter.py`)

#### XenonSimulationAdapter

**Purpose**: Bridge between XENON and Qubic visualization

**Interface**: Implements `SimulationAdapter` base class

**Methods**:
- `load_data(source)`: Auto-detect XENON type and convert
- `validate_source(source)`: Check if source is XENON object
- `load_mechanism_timeseries(mechanisms)`: Batch conversion

**Type Detection**:
```python
if isinstance(source, BioMechanism):
    → BioMechanismAdapter
elif isinstance(source, MolecularState):
    → MolecularStateAdapter
elif isinstance(source, Transition):
    → TransitionAdapter
```

## Data Flow

### Static Visualization

```
BioMechanism
    ↓
BioMechanismAdapter.to_3d_network()
    ↓
VisualizationData
    ↓
XenonSimulationAdapter.load_data()
    ↓
StaticPipeline.render()
    ↓
Output (PNG/PDF/SVG)
```

### Streaming Visualization

```
SSA Simulation Loop
    ↓
Generate BioMechanism snapshot
    ↓
BioMechanismAdapter.to_3d_network()
    ↓
XenonSimulationAdapter.load_data()
    ↓
StreamingPipeline.render_frame()
    ↓
WebSocket broadcast
    ↓
Client display
```

### Timeseries Visualization

```
List[BioMechanism] (trajectory)
    ↓
XenonSimulationAdapter.load_mechanism_timeseries()
    ↓
List[VisualizationData]
    ↓
TimeseriesPipeline.render()
    ↓
Animation (MP4/GIF)
```

## Stochastic Simulation Algorithm (SSA)

### Gillespie Algorithm

1. **State**: Track current molecular state `s`
2. **Propensity**: Compute total transition rate `a₀ = Σ kᵢ`
3. **Sample Time**: Draw `τ ~ Exp(a₀)`
4. **Select Reaction**: Choose transition `i` with probability `kᵢ/a₀`
5. **Update**: Move to target state, advance time by `τ`
6. **Repeat**: Until termination condition

### Integration Points

- **Per-Step**: Visualize current state distribution
- **Per-Transition**: Highlight active reaction pathway
- **Trajectory**: Animate full time evolution

## Performance Considerations

### Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| BioMechanism | O(1) | O(n + e) |
| Spring Layout | O(n²) | O(n) |
| Circular Layout | O(n) | O(n) |
| Energy Surface | O(r²) | O(r²) |
| Arrow Generation | O(s) | O(s) |

Where:
- `n` = number of states
- `e` = number of transitions
- `r` = surface resolution
- `s` = number of segments

### Optimization Strategies

1. **Lazy Evaluation**: Compute normals only when needed
2. **Caching**: Store layout positions across frames
3. **LOD**: Reduce resolution for distant/small meshes
4. **Batching**: Group multiple transitions for rendering

## Determinism and Reproducibility

### Seed Management

- SSA uses `np.random.seed()` for reproducibility
- Each adapter uses deterministic layouts when seeded
- CLI provides `--seed` option (future enhancement)

### Validation

- Unit tests verify deterministic output
- Integration tests check reproducibility
- Coverage target: >90%

## Safety and Compliance

### Read-Only Operations

- Adapters **never modify** source data
- Pure transformation: `XENON → VisualizationData`
- No side effects beyond visualization

### Error Handling

- Type validation at adapter entry points
- Dimension checks in `VisualizationData`
- Graceful degradation for missing fields

### Security

- No external network calls
- No file system access (except explicit exports)
- No code execution from data

## Extension Points

### New Adapter Types

1. Create adapter class in `xenon/adapters/`
2. Implement transformation to `VisualizationData`
3. Add to `XenonSimulationAdapter` type dispatch
4. Write unit tests

### Custom Layouts

1. Add method to `BioMechanismAdapter`
2. Implement position calculation algorithm
3. Register layout name in `to_3d_network()`

### New Visualizations

1. Subclass `TransitionAdapter` or `MolecularStateAdapter`
2. Override `to_*()` methods
3. Maintain `VisualizationData` output contract

## Testing Strategy

### Unit Tests

- Each adapter tested in isolation
- Synthetic data generation
- Property-based testing for edge cases

### Integration Tests

- End-to-end: XENON → Adapter → Pipeline
- Multiple data types
- Timeseries handling

### Performance Tests

- Large mechanisms (>1000 states)
- High-frequency streaming (>30 FPS)
- Memory profiling

## CLI Architecture

### Command Structure

```
xenon [OPTIONS]
    --num-states N          # Simulation size
    --visualize             # Enable viz
    --viz-backend BACKEND   # Renderer choice
    --export-json FILE      # Output format
    --min-evidence SCORE    # Filter threshold
    --verbose               # Logging level
```

### Execution Flow

1. Parse arguments
2. Create sample mechanism (or load from file)
3. Run SSA simulation
4. Convert to visualization data
5. Render/export based on options

## Future Architecture

### Planned Enhancements

1. **File I/O**: Load mechanisms from SBML, JSON, YAML
2. **Parallel SSA**: Multi-trajectory simulation
3. **GPU Acceleration**: CUDA-accelerated network layout
4. **Interactive Exploration**: Mouse picking, zooming
5. **Statistical Analysis**: Ensemble visualization

### API Stability

- Core data structures: **Stable** (v1.0)
- Adapter interfaces: **Stable** (v1.0)
- CLI options: **Evolving** (v0.x)
- Internal algorithms: **Subject to change**

## References

### Academic

- Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical reactions"
- Fruchterman, T. M. J., & Reingold, E. M. (1991). "Graph drawing by force-directed placement"

### Technical

- Qubic Visualization Pipeline: `/qubic/visualization/README.md`
- NumPy Documentation: https://numpy.org/doc/
- NetworkX (reference): https://networkx.org/

## Maintenance

### Code Owners

- Core Module: QuASIM Team
- Adapters: Visualization Team
- CLI: Integration Team

### Versioning

- Semantic versioning (MAJOR.MINOR.PATCH)
- Breaking changes require MAJOR bump
- New features: MINOR bump
- Bug fixes: PATCH bump

### Dependencies

- **Required**: `numpy`, `matplotlib` (for data structures)
- **Optional**: `websockets` (for streaming)
- **Development**: `pytest`, `pytest-cov`, `ruff`
