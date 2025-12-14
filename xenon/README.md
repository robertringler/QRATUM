# XENON: Bio-Mechanism Simulation and Visualization

XENON is a bio-mechanism simulation system integrated with the Qubic visualization pipeline. It models biochemical reaction networks using stochastic simulation algorithms (SSA) and provides adapters for real-time visualization.

## Overview

XENON transforms bio-mechanism simulation outputs (BioMechanism, MolecularState, Transition events) into the standardized Qubic visualization data model, enabling:

- **Network Visualization**: Visualize biochemical reaction networks as directed graphs
- **Energy Landscapes**: Render molecular state energy surfaces
- **Transition Dynamics**: Display reaction pathways and energy barriers
- **Real-Time Streaming**: Stream SSA simulation events to visualization pipeline

## Architecture

```
xenon/
├── core/               # Core data structures
│   ├── mechanism.py    # BioMechanism, MolecularState, Transition
│   └── __init__.py
├── adapters/           # Visualization adapters
│   ├── bio_mechanism_adapter.py    # DAG to network visualization
│   ├── molecular_state_adapter.py  # Energy surfaces and point clouds
│   ├── transition_adapter.py       # Arrows and energy barriers
│   └── __init__.py
├── tests/              # Unit and integration tests
│   ├── test_adapters.py
│   ├── test_integration.py
│   └── test_cli.py
├── cli.py              # Command-line interface
├── streaming_demo.py   # Streaming visualization demo
└── README.md
```

## Core Data Structures

### BioMechanism

Represents a biological mechanism as a directed acyclic graph (DAG) of molecular states and transitions.

```python
from xenon.core.mechanism import BioMechanism, MolecularState, Transition

states = [
    MolecularState("S1", "ProteinA", free_energy=-10.0, concentration=1.0),
    MolecularState("S2", "ProteinB", free_energy=-15.0, concentration=0.0),
]

transitions = [
    Transition("S1", "S2", rate_constant=1.5, delta_g=-5.0, activation_energy=20.0),
]

mechanism = BioMechanism(
    mechanism_id="MECH_001",
    states=states,
    transitions=transitions,
    evidence_score=0.9
)
```

### MolecularState

Represents a molecular configuration with thermodynamic properties.

**Attributes:**
- `state_id`: Unique identifier
- `protein_name`: Name of the protein
- `free_energy`: Gibbs free energy (ΔG) in kJ/mol
- `concentration`: Molecular concentration (0-1)
- `metadata`: Additional state-specific data

### Transition

Represents a state transition (reaction) with kinetic parameters.

**Attributes:**
- `source_state`: Source state ID
- `target_state`: Target state ID
- `rate_constant`: Transition rate (k) in s⁻¹
- `delta_g`: Free energy change (ΔG) in kJ/mol
- `activation_energy`: Activation barrier in kJ/mol
- `metadata`: Additional transition data

## Visualization Adapters

### BioMechanismAdapter

Converts bio-mechanism DAGs into 3D network visualizations.

```python
from xenon.adapters import BioMechanismAdapter

adapter = BioMechanismAdapter(mechanism)

# Get graph data (nodes and edges)
viz_model = adapter.to_viz_model()

# Convert to 3D network with layout
viz_data = adapter.to_3d_network(layout="spring", scale=10.0)
```

**Supported Layouts:**
- `spring`: Force-directed spring layout
- `circular`: Circular arrangement
- `hierarchical`: Level-based hierarchy

### MolecularStateAdapter

Converts molecular states into energy surfaces or point clouds.

```python
from xenon.adapters import MolecularStateAdapter

adapter = MolecularStateAdapter(state)

# Create energy surface
viz_data = adapter.to_energy_surface(resolution=50)

# Create point cloud
viz_data = adapter.to_point_cloud(num_points=1000)
```

### TransitionAdapter

Converts transitions into arrows or energy barrier diagrams.

```python
from xenon.adapters import TransitionAdapter
import numpy as np

adapter = TransitionAdapter(transition)

# Create arrow between positions
source_pos = np.array([0.0, 0.0, 0.0])
target_pos = np.array([10.0, 0.0, 0.0])
viz_data = adapter.to_arrow(source_pos, target_pos)

# Create energy barrier diagram
viz_data = adapter.to_energy_barrier(num_points=50)
```

## Integration with Qubic Visualization Pipeline

### XenonSimulationAdapter

The `XenonSimulationAdapter` provides seamless integration with the Qubic visualization pipeline.

```python
from qubic.visualization.adapters.xenon_adapter import XenonSimulationAdapter
from qubic.visualization.pipelines.streaming import StreamingPipeline

# Create adapter
adapter = XenonSimulationAdapter(layout="spring", scale=10.0)

# Load XENON data
viz_data = adapter.load_data(mechanism)

# Stream to visualization pipeline (placeholder for async implementation)
# pipeline = StreamingPipeline(backend="gpu")
# pipeline.render_frame(viz_data)
```

### Streaming Workflow

```python
from xenon.adapters import BioMechanismAdapter

# For each simulation step
for mechanism in runtime.get_mechanisms(min_evidence=0.8):
    # Convert to visualization data
    adapter = BioMechanismAdapter(mechanism)
    viz_data = adapter.to_3d_network()
    
    # Stream to pipeline
    pipeline.render_frame(viz_data)
```

## Command-Line Interface

The XENON CLI provides simulation and visualization capabilities.

### Basic Usage

```bash
# Run simulation with 5 states
python xenon/cli.py --num-states 5

# Run with verbose output
python xenon/cli.py --num-states 5 --verbose

# Enable visualization
python xenon/cli.py --num-states 5 --visualize --viz-backend matplotlib

# Export visualization data
python xenon/cli.py --num-states 5 --visualize --export-json output.json
```

### CLI Options

- `--num-states N`: Number of molecular states (default: 5)
- `--visualize`: Enable visualization of simulation output
- `--viz-backend BACKEND`: Visualization backend (gpu, headless, matplotlib)
- `--export-json FILE`: Export visualization data to JSON file
- `--min-evidence SCORE`: Minimum evidence score for mechanisms (default: 0.8)
- `-v, --verbose`: Enable verbose output

## Streaming Demo

The streaming demo showcases real-time SSA simulation with visualization:

```bash
# Run streaming demo
PYTHONPATH=/home/runner/work/Qubic/Qubic:$PYTHONPATH \
python xenon/streaming_demo.py --steps 10

# With custom evidence threshold
PYTHONPATH=/home/runner/work/Qubic/Qubic:$PYTHONPATH \
python xenon/streaming_demo.py --steps 10 --min-evidence 0.9
```

The demo:
1. Creates a protein folding mechanism
2. Runs stochastic simulation (Gillespie algorithm)
3. Generates visualization snapshots at each step
4. Converts to Qubic visualization data format

## Examples

### Example 1: Simple Protein Folding

```python
from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.adapters import BioMechanismAdapter

# Define protein folding pathway
states = [
    MolecularState("Unfolded", "ProteinX", 0.0, 1.0),
    MolecularState("Intermediate", "ProteinX", -15.0, 0.0),
    MolecularState("Native", "ProteinX", -35.0, 0.0),
]

transitions = [
    Transition("Unfolded", "Intermediate", 2.0, -15.0, 25.0),
    Transition("Intermediate", "Native", 3.0, -20.0, 18.0),
]

mechanism = BioMechanism("Folding", states, transitions, 0.95)

# Visualize
adapter = BioMechanismAdapter(mechanism)
viz_data = adapter.to_3d_network(layout="hierarchical", scale=5.0)
```

### Example 2: Energy Landscape

```python
from xenon.core.mechanism import MolecularState
from xenon.adapters import MolecularStateAdapter

# Create molecular state
state = MolecularState("Native", "ProteinX", -35.0, 1.0)

# Generate energy surface
adapter = MolecularStateAdapter(state)
viz_data = adapter.to_energy_surface(resolution=100, energy_scale=1.5)

# Access data
print(f"Vertices: {viz_data.vertices.shape}")
print(f"Energy range: {viz_data.get_field_range('energy')}")
```

### Example 3: Transition Visualization

```python
from xenon.core.mechanism import Transition
from xenon.adapters import TransitionAdapter
import numpy as np

# Create transition
transition = Transition("S1", "S2", 5.0, -10.0, 25.0)

# Visualize as energy barrier
adapter = TransitionAdapter(transition)
viz_data = adapter.to_energy_barrier(num_points=100)

# Get color/thickness for visualization
color = adapter.get_color_by_rate(min_rate=0.0, max_rate=10.0)
thickness = adapter.get_thickness_by_rate()
```

## Testing

Run the test suite:

```bash
# All tests
pytest xenon/tests/ -v

# Specific test file
pytest xenon/tests/test_adapters.py -v

# With coverage
pytest xenon/tests/ --cov=xenon --cov-report=html
```

## Design Principles

1. **Non-Destructive**: Adapters are read-only on simulation data
2. **Separation of Concerns**: Core simulation logic separate from visualization
3. **Standardization**: All adapters output `VisualizationData` format
4. **Extensibility**: Easy to add new adapter types
5. **Performance**: Efficient conversion for real-time streaming

## Future Enhancements

- [ ] Full async streaming pipeline integration
- [ ] WebGL/GPU-accelerated rendering
- [ ] Interactive network exploration
- [ ] Animation export (MP4, GIF)
- [ ] Multiple mechanism comparison
- [ ] Parameter sensitivity visualization
- [ ] Pathway probability analysis

## References

- **Gillespie Algorithm**: Stochastic simulation of chemical kinetics
- **Biochemical Networks**: Reaction network modeling
- **Free Energy**: Thermodynamic driving forces
- **Qubic Visualization**: Standardized visualization pipeline

## License

Apache 2.0 License - See main repository LICENSE file.
