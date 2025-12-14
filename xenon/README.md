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
