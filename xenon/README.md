# XENON: Xenobiotic Execution Network for Organismal Neurosymbolic Reasoning

**A post-GPU biological intelligence platform that replaces tensor-based approaches with mechanism-based continuous learning.**

## Overview

XENON represents a paradigm shift in biological intelligence:

- **Computational Primitive**: Biological mechanism DAGs (not tensors)
- **Learning**: Sequential Bayesian updating (not gradient descent)
- **Value**: Mechanistic explanations with provenance (not static weights)
- **Moat**: Non-exportable experimental history (not replicable datasets)

## Installation

### Dependencies

```bash
pip install numpy>=1.21 scipy>=1.7 click>=8.0
```

Optional (recommended):
```bash
pip install networkx>=2.6
```

### Install XENON

```bash
# From source
git clone https://github.com/robertringler/Qubic.git
cd Qubic
pip install -e .
```

## Quick Start

### Python API

```python
from xenon import XENONRuntime, BioMechanism, MolecularState, Transition

# Create a simple mechanism
mech = BioMechanism("test_mechanism")

# Define molecular states
state1 = MolecularState(
    name="Protein_inactive",
    molecule="TestProtein",
    free_energy=-10.0,
    concentration=100.0
)
state2 = MolecularState(
    name="Protein_active",
    molecule="TestProtein",
    free_energy=-12.0,
    concentration=0.0
)

mech.add_state(state1)
mech.add_state(state2)

# Define transition
transition = Transition(
    source="Protein_inactive",
    target="Protein_active",
    rate_constant=1.5e-3,
)

mech.add_transition(transition)

# Validate thermodynamics
is_feasible = mech.is_thermodynamically_feasible()
print(f"Thermodynamically feasible: {is_feasible}")

# Run XENON runtime
runtime = XENONRuntime()
runtime.add_target(
    name="test_target",
    protein="TestProtein",
    objective="characterize"
)

summary = runtime.run(max_iterations=10)
print(f"Converged: {summary['converged']}")
print(f"Mechanisms discovered: {summary['mechanisms_discovered']}")

# Get high-confidence mechanisms
mechanisms = runtime.get_mechanisms(min_evidence=0.5)
for mech in mechanisms:
    print(f"{mech.name}: posterior={mech.posterior:.4f}")
```

### Command-Line Interface

```bash
# Run XENON learning loop
xenon run --target EGFR --max-iter 100 --output results.json

# Query learned mechanisms
xenon query --target EGFR --min-evidence 0.7 --input results.json

# Validate mechanism file
xenon validate --mechanism-file mechanism.json
```

## Architecture

XENON consists of seven core components:

1. **Mechanism Representation** (`xenon.core`)
   - Biological mechanisms as directed acyclic graphs
   - Molecular states with thermodynamic properties
   - Chemical transitions with kinetic parameters

2. **Bayesian Learning** (`xenon.learning`)
   - Sequential Bayesian updating from experiments
   - Prior computation from literature/conservation
   - Evidence-based mechanism ranking

3. **Stochastic Simulation** (`xenon.simulation`)
   - Gillespie SSA (exact stochastic simulation)
   - Langevin dynamics (continuous approximation)
   - Performance target: 10^6 reactions/second

4. **XENON Runtime** (`xenon.runtime`)
   - Continuous learning loop (no epochs)
   - Hypothesis generation and mutation
   - Experiment selection and execution
   - Convergence detection

5. **Mechanism Repository** (Phase 2)
   - Versioned storage with provenance
   - Deduplication via mechanism hashing
   - Experimental lineage tracking

6. **Hypothesis Generation** (Phase 2+)
   - Literature mining (PubMed/bioRxiv)
   - Ontology reasoning (GO/ChEBI)
   - Mechanism synthesis and ranking

7. **Cloud Lab Integration** (Phase 2+)
   - Automated experiment execution
   - Real-time feedback loop
   - Multi-target parallelization

## API Reference

### Core Classes

#### `BioMechanism`
Biological mechanism represented as a DAG.

**Methods:**
- `add_state(state)` - Add molecular state
- `add_transition(transition)` - Add transition
- `is_thermodynamically_feasible(temperature)` - Validate thermodynamics
- `validate_conservation_laws()` - Check conservation
- `get_causal_paths(source, target)` - Find reaction pathways
- `compute_mechanism_hash()` - SHA256 hash for deduplication
- `to_dict() / from_dict()` - Serialization

#### `MolecularState`
A molecular state (e.g., phosphorylated protein, ligand-bound receptor).

**Attributes:**
- `name` - Unique identifier
- `molecule` - Molecule name
- `properties` - State properties (dict)
- `concentration` - Concentration in nM
- `free_energy` - Gibbs free energy (kcal/mol)

#### `Transition`
A chemical reaction or conformational change.

**Attributes:**
- `source` - Source state
- `target` - Target state
- `rate_constant` - Rate constant (1/s)
- `activation_energy` - Activation energy (kcal/mol)
- `reversible` - Whether reversible
- `reverse_rate` - Reverse rate constant

#### `XENONRuntime`
Main runtime orchestrating the learning loop.

**Methods:**
- `add_target(name, protein, objective)` - Define learning target
- `run(max_iterations)` - Execute learning loop
- `get_mechanisms(min_evidence)` - Retrieve high-confidence mechanisms
- `get_summary()` - Get runtime statistics

### Simulation

#### `GillespieSimulator`
Exact stochastic simulation (SSA).

**Methods:**
- `run(t_max, initial_state, seed)` - Run simulation
- Returns: `(times, trajectories)` tuple

#### `LangevinSimulator`
Brownian dynamics with thermal noise.

**Methods:**
- `run(t_max, dt, initial_state, seed)` - Run simulation
- Returns: `(times, trajectories)` tuple

### Learning

#### `BayesianUpdater`
Bayesian updating of mechanism posteriors.

**Methods:**
- `update_mechanisms(mechanisms, experiment_result)` - Bayesian update
- `compute_likelihood(mechanism, experiment)` - P(data | mechanism)
- `prune_low_evidence(mechanisms, threshold)` - Remove low-posterior mechanisms
- `get_evidence_summary(mechanisms)` - Summary statistics

#### `MechanismPrior`
Prior probability computation.

**Methods:**
- `compute_prior(mechanism)` - Prior probability
- `rate_constant_prior(transition)` - Kinetics plausibility
- `initialize_mechanism_priors(mechanisms)` - Initialize and normalize priors

## Examples

### Example 1: Simple Two-State System

```python
from xenon import create_mechanism, simulate_mechanism, validate_mechanism

# Create mechanism from dictionaries
states = [
    {"name": "State_A", "molecule": "Protein", "free_energy": -10.0},
    {"name": "State_B", "molecule": "Protein", "free_energy": -12.0},
]

transitions = [
    {"source": "State_A", "target": "State_B", "rate_constant": 1.0},
]

mech = create_mechanism("two_state", states, transitions)

# Validate
validation = validate_mechanism(mech)
print(validation)

# Simulate
times, traj = simulate_mechanism(
    mech,
    t_max=5.0,
    initial_state={"State_A": 100.0, "State_B": 0.0},
    method="gillespie",
)

print(f"Final concentrations: A={traj['State_A'][-1]:.2f}, B={traj['State_B'][-1]:.2f}")
```

### Example 2: Reversible Enzyme Catalysis

```python
from xenon import BioMechanism, MolecularState, Transition
from xenon.simulation import GillespieSimulator

# E + S <-> ES -> E + P
mech = BioMechanism("enzyme_catalysis")

# States
enzyme = MolecularState(name="E", molecule="Enzyme", concentration=10.0)
substrate = MolecularState(name="S", molecule="Substrate", concentration=100.0)
complex = MolecularState(name="ES", molecule="Complex", concentration=0.0)
product = MolecularState(name="P", molecule="Product", concentration=0.0)

for state in [enzyme, substrate, complex, product]:
    mech.add_state(state)

# Reactions (simplified)
# E + S -> ES (assume pseudo-first-order for simulation simplicity)
mech.add_transition(Transition(source="S", target="ES", rate_constant=0.1))
# ES -> E + S
mech.add_transition(Transition(source="ES", target="S", rate_constant=0.05))
# ES -> E + P
mech.add_transition(Transition(source="ES", target="P", rate_constant=0.02))

# Simulate
simulator = GillespieSimulator(mech)
times, traj = simulator.run(
    t_max=100.0,
    initial_state={"E": 10.0, "S": 100.0, "ES": 0.0, "P": 0.0},
    seed=42,
)

print(f"Product formed: {traj['P'][-1]:.2f} nM")
```

### Example 3: Multi-Target Learning

```python
from xenon import run_xenon

targets = [
    {"name": "EGFR_target", "protein": "EGFR", "objective": "characterize"},
    {"name": "KRAS_target", "protein": "KRAS", "objective": "find_inhibitor"},
]

results = run_xenon(targets, max_iterations=50)

print(f"Total mechanisms: {results['mechanisms_discovered']}")
print(f"Converged: {results['converged']}")
```

## Performance Benchmarks

Phase 1 targets:

- **Mechanism storage**: 10^6 mechanisms in <10 GB
- **Gillespie SSA**: 10^6 reactions/second (single CPU core)
- **Bayesian update**: <100 ms per experiment
- **Convergence**: <100 iterations for simple systems

## Scientific Validation

XENON enforces rigorous scientific constraints:

1. **Thermodynamic Consistency**
   - ΔG = -RT ln(K_eq)
   - Detailed balance for reversible reactions
   - No perpetual motion (cycle ΔG sum = 0)

2. **Conservation Laws**
   - Mass conservation
   - Charge conservation
   - Energy conservation

3. **Kinetic Plausibility**
   - Rate constants within physical bounds
   - Diffusion limits
   - Activation energies

## Comparison to AlphaFold/Tensor Approaches

| Aspect | XENON (Mechanism-Based) | AlphaFold/Deep Learning |
|--------|-------------------------|-------------------------|
| **Primitive** | Biological mechanism DAG | Tensor |
| **Learning** | Sequential Bayesian | Gradient descent |
| **Output** | Mechanistic explanation | Static prediction |
| **Provenance** | Full experimental lineage | Training dataset |
| **Interpretability** | Causal pathways | Black box |
| **Update** | Continuous (new experiments) | Retrain from scratch |
| **Hardware** | CPU-optimized | GPU-dependent |
| **Moat** | Non-exportable history | Replicable weights |

## Why This Displaces NVIDIA GPUs

1. **Computational Primitive**: Mechanism DAGs are sparse graphs, not dense tensors
2. **Learning Algorithm**: Bayesian updates are CPU-efficient, no backpropagation
3. **Data Efficiency**: Learn from single experiments, not millions of examples
4. **Continuous Learning**: No retraining, only incremental updates
5. **Interpretability**: Output is mechanistic explanation, not learned weights
6. **Moat**: Accumulated experimental history is non-exportable

## Phase 2+ Roadmap

- **Cloud Lab Integration**: Automated experiment execution (Emerald Cloud Lab, Strateos)
- **Literature Mining**: PubMed/bioRxiv automated extraction
- **Ontology Reasoning**: GO, ChEBI, UniProt integration
- **Multi-Omics**: Proteomics, metabolomics, genomics data fusion
- **Drug Discovery**: Automated inhibitor/activator screening
- **Scale**: 10^9 mechanisms, 10^6 experiments

## Citation

```
@software{xenon2024,
  title={XENON: Xenobiotic Execution Network for Organismal Neurosymbolic Reasoning},
  author={XENON Project},
  year={2024},
  url={https://github.com/robertringler/Qubic}
}
```

## License

Apache 2.0

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Support

- Documentation: [ARCHITECTURE.md](ARCHITECTURE.md)
- Issues: https://github.com/robertringler/Qubic/issues
- Email: [Contact via GitHub]
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
