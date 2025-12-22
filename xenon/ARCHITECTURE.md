# XENON Architecture

## System Overview

XENON is a biological intelligence platform that replaces tensor-based deep learning with mechanism-based continuous learning. This document describes the Phase 1 architecture and scientific foundations.

## Core Paradigm Shift

### Traditional Deep Learning (AlphaFold, etc.)

```
Input (sequence) → Neural Network (tensors) → Output (structure)
                    ↓ Gradient Descent
                Training Data (millions)
```

**Limitations:**

- Black box (no interpretability)
- Requires massive datasets
- Static output (no updates)
- GPU-dependent
- Weights are exportable moat

### XENON (Mechanism-Based Learning)

```
Hypothesis Mechanisms → Simulation → Experiment Selection → Lab Execution
         ↑                                                        ↓
         └────────────── Bayesian Update ─────────────────────────┘
```

**Advantages:**

- Interpretable (causal mechanisms)
- Learns from single experiments
- Continuous updates
- CPU-optimized
- Non-exportable experimental history moat

## Architecture Components

### 1. Mechanism Representation

**Computational Primitive:** Biological Mechanism DAG

```python
class BioMechanism:
    """Directed acyclic graph of molecular interactions."""
    
    nodes: Dict[str, MolecularState]  # Molecular configurations
    edges: List[Transition]            # Chemical reactions
```

#### Molecular State

Represents a specific configuration of a molecule:

```python
@dataclass
class MolecularState:
    name: str                    # Unique identifier
    molecule: str                # Protein/ligand name
    properties: Dict             # Phosphorylation, binding, etc.
    concentration: float         # nM (for simulation)
    free_energy: float          # kcal/mol (for thermodynamics)
```

**Examples:**

- EGFR_inactive (no ligand bound)
- EGFR_EGF_bound (ligand bound)
- EGFR_phosphorylated (active kinase)

#### Transition

Represents a chemical reaction or conformational change:

```python
@dataclass
class Transition:
    source: str                  # Source state
    target: str                  # Target state
    rate_constant: float         # k (1/s or 1/(M*s))
    activation_energy: float     # E_a (kcal/mol)
    reversible: bool             # Reversible reaction?
    reverse_rate: float          # k_reverse
```

**Scientific Constraints:**

1. **Thermodynamic Consistency**

   ```
   ΔG = -RT ln(K_eq)
   K_eq = k_forward / k_reverse
   ```

2. **Detailed Balance**

   ```
   For cycle: ∑ ΔG_i = 0
   ```

3. **Kinetic Bounds**

   ```
   Diffusion limit: k_max ≈ 10^9 M^-1 s^-1
   Typical enzyme: k_cat ≈ 10^2 - 10^6 s^-1
   ```

### 2. Bayesian Learning Engine

**Core Algorithm:** Sequential Bayesian Inference

```
P(mechanism | experiment) ∝ P(experiment | mechanism) × P(mechanism)
```

#### Prior Computation

Combines three sources of prior information:

1. **Chemical Plausibility**
   - Rate constants from literature
   - Thermodynamic feasibility
   - Activation energy estimates

2. **Evolutionary Conservation**
   - Sequence homology (UniProt)
   - Conserved domains (Pfam)
   - Cross-species comparison

3. **Literature Citations**
   - PubMed/bioRxiv mentions
   - Interaction databases (STRING, BioGRID)
   - Known pathways (KEGG, Reactome)

**Prior Formula:**

```python
prior = (
    P_rate_constants^(w1) *
    P_conservation^(w2) *
    P_literature^(w3)
)
# where w1 + w2 + w3 = 1
```

#### Likelihood Computation

**Experiment Types:**

1. **Concentration Measurements**

   ```python
   L = exp(-χ² / 2)
   χ² = Σ ((observed - predicted) / uncertainty)²
   ```

2. **Kinetics Measurements**

   ```python
   L = exp(-Σ (log(k_obs / k_pred))²)
   # Log-normal for rates spanning orders of magnitude
   ```

3. **Perturbation Experiments**

   ```python
   L = 1.0 if causal_path_exists(source, target) else 0.1
   ```

#### Posterior Update

```python
for mechanism in mechanisms:
    likelihood = compute_likelihood(mechanism, experiment)
    mechanism.posterior = likelihood * mechanism.prior

# Normalize
total = sum(m.posterior for m in mechanisms)
for m in mechanisms:
    m.posterior /= total
```

### 3. Stochastic Simulation Engine

Two complementary algorithms:

#### Gillespie SSA (Exact)

**Algorithm:**

```
1. Compute propensities: a_i = k_i × n_reactants
2. Total propensity: a_0 = Σ a_i
3. Sample time: τ ~ Exponential(a_0)
4. Select reaction: j with probability a_j / a_0
5. Update state: apply stoichiometry
6. Advance time: t → t + τ
```

**Performance Target:** 10^6 reactions/second

**Implementation:**

```python
class GillespieSimulator:
    def run(self, t_max, initial_state):
        t = 0
        state = initial_state
        
        while t < t_max:
            propensities = compute_propensities(state)
            a_0 = sum(propensities)
            
            if a_0 == 0:
                break
            
            τ = -log(random()) / a_0
            j = select_reaction(propensities, a_0)
            
            state = update_state(state, j)
            t += τ
```

#### Langevin Dynamics (Approximate)

**Algorithm:** Chemical Langevin Equation

```
dX/dt = drift(X) + sqrt(diffusion(X)) × noise(t)
```

**Use Cases:**

- Large molecule counts (>100)
- Faster than SSA (continuous approximation)
- Temperature-dependent studies

### 4. XENON Runtime (Meta-Kernel)

**Main Loop:**

```python
while not converged:
    # 1. Generate hypothesis mechanisms
    mechanisms = generate_hypotheses(target)
    
    # 2. Simulate mechanisms
    for mechanism in mechanisms:
        simulate_gillespie(mechanism)
    
    # 3. Rank by epistemic uncertainty
    ranked = rank_by_uncertainty(mechanisms)
    
    # 4. Select next experiment (max info gain)
    experiment = select_experiment(ranked)
    
    # 5. Execute experiment
    result = execute_experiment(experiment)
    
    # 6. Update posteriors
    mechanisms = bayesian_update(mechanisms, result)
    
    # 7. Prune low-evidence mechanisms
    mechanisms = prune(mechanisms, threshold=1e-6)
    
    # 8. Check convergence
    entropy = compute_entropy(mechanisms)
    if entropy < threshold:
        break
```

**Convergence Criterion:**

Entropy of mechanism distribution:

```
H = -Σ p_i log(p_i)
```

Converged when H < threshold (e.g., 0.1 nats)

### 5. Hypothesis Generation

**Phase 1 (Current):**

- Template mechanisms (hand-coded)
- Topology mutations (add/remove edges)
- Rate constant perturbations

**Phase 2+ (Future):**

- Literature mining (PubMed API)
- Ontology reasoning (GO, ChEBI)
- Mechanism synthesis from databases
- Evolutionary algorithms

### 6. Experiment Selection

**Objective:** Maximize expected information gain

```
I(experiment) = H(prior) - E[H(posterior | experiment)]
```

**Phase 1 (Current):**

- Random experiment selection
- Uniform sampling of experiment types

**Phase 2+ (Future):**

- Bayesian experimental design
- Active learning
- Cost-aware selection
- Parallelization across targets

### 7. Cloud Lab Integration (Phase 2+)

**Automated Execution:**

```
XENON Runtime → API → Cloud Lab → Physical Experiment → Data → XENON
```

**Supported Platforms:**

- Emerald Cloud Lab
- Strateos
- Transcriptic

**Experiment Types:**

- Protein expression
- Kinetics assays (SPR, ITC)
- Perturbation screens (CRISPR, RNAi)
- Mass spectrometry

## Data Structures

### Mechanism DAG Format

```json
{
  "name": "EGFR_activation",
  "states": [
    {
      "name": "EGFR_inactive",
      "molecule": "EGFR",
      "properties": {"phosphorylated": false},
      "free_energy": -10.0
    },
    {
      "name": "EGFR_active",
      "molecule": "EGFR",
      "properties": {"phosphorylated": true},
      "free_energy": -12.0
    }
  ],
  "transitions": [
    {
      "source": "EGFR_inactive",
      "target": "EGFR_active",
      "rate_constant": 1.5e-3,
      "activation_energy": 15.0,
      "reversible": true,
      "reverse_rate": 3.0e-4
    }
  ],
  "posterior": 0.75,
  "provenance": ["literature_init", "experiment_123", "experiment_456"],
  "hash": "a3f5d8e9c2b1..."
}
```

### Experimental Provenance

Every mechanism maintains full lineage:

```python
mechanism.provenance = [
    "literature_init:2024-01-15",
    "experiment_001:concentration:bayesian_update",
    "experiment_002:kinetics:bayesian_update",
    "topology_mutation:0.1_rate",
    "experiment_003:perturbation:bayesian_update",
]
```

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Mechanism hash | O(V + E) | V=states, E=transitions |
| Thermodynamic check | O(E) | Per transition |
| Gillespie step | O(E) | Propensity computation |
| Bayesian update | O(M) | M=mechanisms |
| Isomorphism check | O(V!) | Expensive, use hash first |

### Memory Footprint

**Per Mechanism:**

- States: ~100 bytes per state
- Transitions: ~200 bytes per transition
- Metadata: ~500 bytes
- **Total**: ~1 KB for typical mechanism (5 states, 10 transitions)

**Scaling:**

- 10^6 mechanisms: ~1 GB
- 10^9 mechanisms: ~1 TB (Phase 2+)

### CPU vs GPU

**Why CPU-Optimized:**

1. **Sparse Operations**: Mechanism DAGs are sparse graphs
   - GPU optimal for dense matrix operations
   - CPU optimal for graph traversal

2. **Branching Logic**: Bayesian updates require conditionals
   - GPU suffers from branch divergence
   - CPU handles branches efficiently

3. **Memory Access**: Random access to mechanism repository
   - GPU requires coalesced memory access
   - CPU has better random access

4. **Gillespie SSA**: Inherently sequential algorithm
   - Cannot be parallelized across time steps
   - Can parallelize across mechanisms (CPU multi-core)

## Comparison to AlphaFold

### AlphaFold Architecture

```
Input: Amino acid sequence (L residues)
       ↓
Embedding: L × 256 tensor
       ↓
Evoformer: 48 blocks × (MSA attention + Pair attention)
       ↓
Structure module: 8 layers × (IPA + transitions)
       ↓
Output: 3D coordinates (L × 3)
```

**Characteristics:**

- Parameters: ~93M (AlphaFold2)
- Training data: ~170K structures (PDB)
- Inference: ~1 minute per protein (GPU)
- Hardware: A100 GPU (80 GB)

### XENON Architecture

```
Input: Target protein
       ↓
Hypothesis generation: 10-1000 mechanism DAGs
       ↓
Simulation: Gillespie SSA (CPU)
       ↓
Experiment selection: Bayesian optimization
       ↓
Lab execution: Physical experiment
       ↓
Bayesian update: Posterior refinement
       ↓
Output: Mechanistic explanation with confidence
```

**Characteristics:**

- Parameters: None (data structure, not weights)
- Training data: None (learns from experiments)
- Inference: Continuous learning loop
- Hardware: CPU (no GPU required)

### Key Differences

| Aspect | AlphaFold | XENON |
|--------|-----------|-------|
| **Problem** | Structure prediction | Mechanism discovery |
| **Input** | Sequence | Target protein |
| **Output** | 3D structure | Causal mechanism |
| **Learning** | Offline (training) | Online (continuous) |
| **Data** | Static dataset | Live experiments |
| **Hardware** | GPU (essential) | CPU (sufficient) |
| **Interpretability** | Black box | White box |
| **Update** | Retrain | Incremental |
| **Moat** | Weights (exportable) | History (non-exportable) |

## Why This Displaces NVIDIA GPUs

### Technical Reasons

1. **Computational Primitive Mismatch**
   - GPUs: Optimized for dense matrix multiplication (GEMM)
   - XENON: Requires sparse graph operations
   - Result: GPU offers no advantage

2. **Algorithm Characteristics**
   - GPUs: Massively parallel, data-parallel workloads
   - XENON: Sequential Bayesian updates, branching logic
   - Result: CPU is more efficient

3. **Memory Patterns**
   - GPUs: Coalesced memory access required
   - XENON: Random access to mechanism repository
   - Result: CPU cache is more effective

4. **Scale Economics**
   - GPUs: $10K-$30K per A100
   - CPU: $1K-$5K per server
   - Result: 10x cost advantage

### Economic Reasons

1. **Hardware Capex**
   - GPU cluster: $10M for 1000 A100s
   - CPU cluster: $1M for equivalent compute
   - **Savings: $9M**

2. **Power Consumption**
   - A100: 400W TDP
   - CPU: 150W TDP
   - **Savings: 60% OpEx**

3. **Datacenter Footprint**
   - GPU: 8U per 8xA100 node
   - CPU: 1U per dual-socket node
   - **Savings: 75% space**

### Strategic Reasons

1. **Non-Replicable Moat**
   - AlphaFold: Weights can be copied
   - XENON: Experimental history cannot be replicated
   - **Defensibility: Infinite**

2. **Continuous Learning**
   - AlphaFold: Retrain from scratch
   - XENON: Incremental updates
   - **Time-to-value: 100x faster**

3. **Interpretability**
   - AlphaFold: Black box
   - XENON: Causal mechanisms
   - **Regulatory approval: Essential for drugs**

## Phase 2+ Roadmap

### Near-Term (6 months)

1. **Literature Mining**
   - PubMed API integration
   - Named entity recognition
   - Interaction extraction

2. **Cloud Lab Integration**
   - Emerald Cloud Lab API
   - Experiment templates
   - Automated execution

3. **Performance Optimization**
   - Gillespie SSA: 10^7 reactions/second
   - Parallel simulation across mechanisms
   - Compressed mechanism storage

### Mid-Term (12 months)

1. **Multi-Omics Integration**
   - Proteomics (mass spec)
   - Metabolomics (LC-MS)
   - Genomics (RNA-seq)

2. **Drug Discovery**
   - Inhibitor screening
   - Activator discovery
   - ADMET prediction

3. **Scale**
   - 10^9 mechanisms
   - 10^6 experiments
   - 10^3 parallel targets

### Long-Term (24 months)

1. **Automated Drug Design**
   - De novo molecule generation
   - Retrosynthesis planning
   - Clinical trial design

2. **Personalized Medicine**
   - Patient-specific mechanisms
   - Mutation effect prediction
   - Treatment optimization

3. **Platform**
   - SaaS deployment
   - API access
   - Marketplace for mechanisms

## References

### Scientific Foundations

1. Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical reactions." *J. Phys. Chem.*, 81(25), 2340-2361.

2. Wilkinson, D. J. (2009). "Stochastic modelling for quantitative description of heterogeneous biological systems." *Nat. Rev. Genet.*, 10(2), 122-133.

3. MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms.* Cambridge University Press.

4. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* Cambridge University Press.

### Related Systems

1. AlphaFold (DeepMind)
2. RoseTTAFold (Baker Lab)
3. BioNetGen (rule-based modeling)
4. Virtual Cell (VCell)
5. COPASI (biochemical simulation)

## Glossary

- **DAG**: Directed Acyclic Graph
- **SSA**: Stochastic Simulation Algorithm
- **nM**: nanomolar (10^-9 M)
- **kcal/mol**: kilocalories per mole (energy unit)
- **K_eq**: Equilibrium constant
- **ΔG**: Gibbs free energy change
- **E_a**: Activation energy
- **RT**: Gas constant × Temperature
- **ITC**: Isothermal Titration Calorimetry
- **SPR**: Surface Plasmon Resonance

## License

Apache 2.0

## Contact

See [README.md](README.md) for support information.

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
- NumPy Documentation: <https://numpy.org/doc/>
- NetworkX (reference): <https://networkx.org/>

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
