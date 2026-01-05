# QCMG: Quantacosmorphysigenetic Field Simulation

## Overview

The Quantacosmorphysigenetic (QCMG) module provides a mathematically rigorous symbolic field evolution system that models the coevolution of physical and informational fields in quantum systems.

## Mathematical Framework

### Field Equations

The QCMG module implements coupled field equations describing the tensor product evolution:

```
Φ = ∂_t (Φ_m ⊗ Φ_i)
```

Where:

- **Φ_m**: Physical/material field representing energy density and spatial structure
- **Φ_i**: Informational field representing entropy, complexity, and coherence
- **⊗**: Tensor coupling operator

### Hamiltonian Dynamics

Evolution follows Hamiltonian dynamics with controlled entropy production:

```
∂Φ_m/∂t = -δH/δΦ_i + η_m(t)
∂Φ_i/∂t = -δH/δΦ_m + η_i(t)
```

Where:

- **H**: Total Hamiltonian (kinetic + coupling energy)
- **η_m(t), η_i(t)**: Dissipative terms modeling energy dissipation

### Physical Observables

The module computes three key observables:

1. **Coherence C(t) ∈ [0, 1]**: Quantum coherence measure

   ```
   C = |⟨Φ_m|Φ_i⟩| / (||Φ_m|| ||Φ_i||)
   ```

2. **Entropy S(t) ≥ 0**: Von Neumann entropy

   ```
   S = -Σ p_i log(p_i)
   ```

   where p_i are derived from field amplitude distributions

3. **Energy E(t)**: Total field energy

   ```
   E = ∫ (|∇Φ_m|² + |∇Φ_i|² + λ|Φ_m·Φ_i*|²) dx
   ```

   where λ is the coupling strength

## Numerical Implementation

### RK4 Integration

The module uses 4th-order Runge-Kutta (RK4) integration for time evolution:

```python
k1 = f(t, y)
k2 = f(t + dt/2, y + dt*k1/2)
k3 = f(t + dt/2, y + dt*k2/2)
k4 = f(t + dt, y + dt*k3)

y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

### Boundary Conditions

Periodic boundary conditions are enforced via finite difference schemes:

- Central differences for gradients
- Second-order differences for Laplacians

### Stability and Normalization

To prevent numerical instabilities:

- Fields are periodically renormalized (every 10 steps)
- Validation checks at each step ensure bounded observables
- Adaptive dissipation prevents runaway growth

## Python API

### Basic Usage

```python
from quasim.sim import QuantacosmomorphysigeneticField, QCMGParameters

# Configure simulation
params = QCMGParameters(
    grid_size=64,        # Number of spatial grid points
    dt=0.01,            # Time step
    coupling_strength=1.0,  # Field coupling
    dissipation_rate=0.01,  # Energy dissipation
    random_seed=42      # For reproducibility
)

# Initialize field
field = QuantacosmomorphysigeneticField(params)
field.initialize(mode="gaussian")  # or "soliton", "random"

# Evolve field
for i in range(100):
    state = field.evolve(steps=1)
    print(f"Step {i}: C={state.coherence:.3f}, S={state.entropy:.3f}, E={state.energy:.3f}")

# Export results
field.save_to_json("results.json", include_history=True)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_size` | int | 64 | Number of spatial discretization points |
| `dt` | float | 0.01 | Time step for integration (smaller = more accurate) |
| `coupling_strength` | float | 1.0 | Strength of Φ_m-Φ_i coupling |
| `dissipation_rate` | float | 0.01 | Energy dissipation rate (≥0) |
| `random_seed` | int or None | None | Random seed for reproducibility |

### Initialization Modes

**Gaussian Mode** (`mode="gaussian"`):

- Creates a Gaussian wave packet centered at π
- Symmetric profile with controlled width
- Good for studying diffusion and spreading

**Soliton Mode** (`mode="soliton"`):

- Creates a soliton-like localized state using sech profile
- Maintains coherence over longer times
- Good for studying stability

**Random Mode** (`mode="random"`):

- Random complex field configuration
- Tests robustness and general behavior
- Requires fixed seed for reproducibility

### State Export

```python
# Export full state with history
data = field.export_state(include_history=True)

# Export compact state (current only)
data = field.export_state(include_history=False)

# Save to JSON
field.save_to_json("output.json")
```

## Command-Line Interface

### Basic Usage

```bash
# Run simulation with defaults
python -m quasim.sim.qcmg_cli --iterations 100

# Custom parameters
python -m quasim.sim.qcmg_cli \
    --iterations 200 \
    --grid-size 128 \
    --dt 0.005 \
    --coupling 0.5 \
    --init-mode soliton \
    --seed 42 \
    --verbose

# Export results
python -m quasim.sim.qcmg_cli \
    --iterations 100 \
    --export results.json \
    --seed 42
```

### CLI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--iterations` | int | 100 | Number of evolution steps |
| `--grid-size` | int | 64 | Spatial grid size |
| `--dt` | float | 0.01 | Time step |
| `--coupling` | float | 1.0 | Coupling strength |
| `--dissipation` | float | 0.01 | Dissipation rate |
| `--init-mode` | str | gaussian | Initialization mode (gaussian/soliton/random) |
| `--seed` | int | None | Random seed |
| `--export` | str | None | Output JSON file path |
| `--verbose` | flag | False | Enable detailed output |
| `--no-history` | flag | False | Disable history in export |

## Validation and Testing

### Automated Checks

Every simulation step validates:

- ✓ Coherence bounded: 0 ≤ C ≤ 1
- ✓ Entropy non-negative: S ≥ 0
- ✓ Energy finite: |E| < ∞
- ✓ No NaN or Inf values
- ✓ Field normalization maintained

### Running Tests

```bash
# Run full test suite
pytest tests/test_qcmg_sim.py -v

# With coverage
pytest tests/test_qcmg_sim.py --cov=quasim.sim.qcmg_field --cov-report=html

# Specific test class
pytest tests/test_qcmg_sim.py::TestFieldEvolution -v
```

### Test Coverage

The test suite covers:

- Parameter validation (6 tests)
- Field initialization (5 tests)
- Evolution dynamics (3 tests)
- Observable validation (4 tests)
- Reproducibility (2 tests)
- State validation (4 tests)
- Export functionality (3 tests)
- Long-time stability (1 test)
- CLI integration (4 tests)

**Total: 32 tests, >95% code coverage**

## Physical Interpretation

### Coherence Dynamics

Coherence C(t) measures the alignment between material and informational fields:

- **C ≈ 1**: Fields are highly aligned (quantum coherent state)
- **C ≈ 0**: Fields are orthogonal (decoherent state)
- Typically decreases over time due to dissipation

### Entropy Production

Entropy S(t) tracks information content and disorder:

- Generally increases over time (second law)
- Rate depends on dissipation and coupling
- Bounded by system size (S_max ~ log(N))

### Energy Conservation

With dissipation rate γ, energy approximately follows:

```
E(t) ≈ E(0) exp(-2γt)
```

## Performance Characteristics

### Computational Complexity

| Grid Size N | Time per Step | Memory Usage |
|-------------|---------------|--------------|
| 32 | ~0.5 ms | ~1 KB |
| 64 | ~2 ms | ~4 KB |
| 128 | ~8 ms | ~16 KB |
| 256 | ~32 ms | ~64 KB |

Scaling: O(N) per step (dominated by field operations)

### Optimization Tips

1. **Reduce grid size** for exploratory runs
2. **Increase dt** if high accuracy not needed (within stability limit)
3. **Disable history** for long runs to save memory
4. **Use compiled NumPy** (OpenBLAS, MKL) for better performance

## Example Use Cases

### 1. Decoherence Study

```python
# Study coherence loss over time
params = QCMGParameters(grid_size=64, dissipation_rate=0.02)
field = QuantacosmomorphysigeneticField(params)
field.initialize(mode="gaussian")

coherences = []
for _ in range(500):
    state = field.evolve(steps=1)
    coherences.append(state.coherence)

# Fit exponential decay
# C(t) ~ exp(-t/τ)
```

### 2. Soliton Stability

```python
# Test soliton stability
params = QCMGParameters(grid_size=128, dissipation_rate=0.001)
field = QuantacosmomorphysigeneticField(params)
field.initialize(mode="soliton")

energies = []
for _ in range(1000):
    state = field.evolve(steps=1)
    energies.append(state.energy)

# Check energy conservation
```

### 3. Coupling Effects

```python
# Compare different coupling strengths
for coupling in [0.5, 1.0, 2.0]:
    params = QCMGParameters(coupling_strength=coupling, random_seed=42)
    field = QuantacosmomorphysigeneticField(params)
    field.initialize(mode="random")
    
    state = field.evolve(steps=100)
    print(f"Coupling {coupling}: Final coherence = {state.coherence:.3f}")
```

## Troubleshooting

### Numerical Instabilities

**Problem**: Fields grow unbounded or NaN values appear

**Solutions**:

- Reduce time step `dt`
- Increase dissipation rate
- Check initial conditions aren't too extreme
- Verify grid size is adequate

### Poor Convergence

**Problem**: Results depend strongly on `dt`

**Solutions**:

- Use smaller time step (RK4 error ~ dt⁴)
- Increase grid resolution
- Check for stiff dynamics (high coupling)

### Memory Issues

**Problem**: Out of memory for long runs

**Solutions**:

- Disable history tracking (`include_history=False`)
- Reduce grid size
- Process in batches, saving periodically

## References

### Theoretical Background

The QCMG formalism synthesizes concepts from:

- Quantum field theory (field quantization)
- Information theory (entropy measures)
- Hamiltonian mechanics (variational principles)
- Numerical PDE methods (finite differences, RK integration)

### Related Work

- Von Neumann entropy in quantum systems
- Quantum decoherence theory
- Soliton dynamics in nonlinear systems
- Coupled oscillator networks

## License

Part of QuASIM. See main repository LICENSE.

## Contributing

Contributions welcome! Areas for enhancement:

- Multi-dimensional grids (2D, 3D)
- Advanced boundary conditions
- GPU acceleration
- Adaptive time stepping
- Additional initialization modes
- Quantum noise models

Please see CONTRIBUTING.md in the main repository.
