# Quantacosmomorphysigenetic (QCMG) Field Evolution Module

The QCMG module implements the coevolution of physical (Φ_m) and informational (Φ_i) fields through coupled differential equations, modeling the emergence of structure from quantum-classical interactions.

## Mathematical Framework

The evolution follows coupled differential equations:

```
∂Φ_m/∂t = -δH/δΦ_i + η_m(t)
∂Φ_i/∂t = -δH/δΦ_m + η_i(t)
```

Where the Hamiltonian is:

```
H[Φ_m, Φ_i] = ∫ [½|∇Φ_m|² + ½|∇Φ_i|² + V(Φ_m, Φ_i)] dx
```

With physical constraints:

- **Coherence**: C(t) = |⟨Φ_m|Φ_i⟩| / (||Φ_m|| ||Φ_i||) ∈ [0, 1]
- **Entropy**: S(t) = -Tr(ρ log ρ) where ρ ∝ Φ_m ⊗ Φ_i ≥ 0
- **Energy**: Approximately conserved with controlled dissipation

## Features

- **Multiple initialization modes**: Gaussian, soliton, and random field configurations
- **RK4 integration**: Fourth-order Runge-Kutta for accurate numerical evolution
- **Physical observables**: Automatic computation of coherence, entropy, and energy
- **History tracking**: Full evolution history recording
- **State export**: JSON-serializable state snapshots

## Quick Start

```python
from quasim.qcmg import QCMGParameters, QuantacosmorphysigeneticField

# Configure parameters
params = QCMGParameters(
    grid_size=64,
    spatial_extent=10.0,
    dt=0.01,
    coupling_strength=0.1,
    random_seed=42
)

# Initialize field
field = QuantacosmorphysigeneticField(params)
field.initialize(mode="gaussian")

# Evolve the system
for i in range(100):
    state = field.evolve()
    print(f"t={state.time:.2f}, C={state.coherence:.3f}, S={state.entropy:.3f}")

# Export state
export = field.export_state()
```

## Parameters

### QCMGParameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_size` | int | 64 | Number of spatial grid points |
| `spatial_extent` | float | 10.0 | Physical size of simulation domain |
| `dt` | float | 0.01 | Time step for evolution |
| `coupling_strength` | float | 0.1 | Field interaction strength |
| `mass_scale` | float | 1.0 | Physical field mass parameter |
| `info_scale` | float | 1.0 | Information field scale |
| `interaction_strength` | float | 0.05 | Nonlinear coupling strength |
| `thermal_noise` | float | 0.001 | Stochastic noise amplitude |
| `damping_coeff` | float | 0.01 | Energy dissipation rate |
| `random_seed` | int | 42 | Random seed for reproducibility |

## Initialization Modes

### Gaussian

Gaussian wave packet with momentum:

```python
field.initialize(mode="gaussian")
```

- Produces localized, coherent field configurations
- Good for studying wave packet dynamics

### Soliton

Solitonic (localized, stable) solution:

```python
field.initialize(mode="soliton")
```

- High coherence between fields
- Demonstrates stability properties

### Random

Random field configuration:

```python
field.initialize(mode="random")
```

- Low initial coherence
- Tests thermalization and decoherence

## API Reference

### QuantacosmorphysigeneticField

**Methods:**

- `__init__(parameters)`: Initialize simulator with parameters
- `initialize(mode)`: Set initial field configuration
- `evolve()`: Advance fields by one time step
- `get_state()`: Get current field state
- `get_history()`: Get full evolution history
- `export_state(state)`: Export state to JSON-serializable dict

### FieldState

**Attributes:**

- `phi_m`: Physical field amplitude (complex array)
- `phi_i`: Informational field amplitude (complex array)
- `coherence`: Quantum coherence measure [0, 1]
- `entropy`: Von Neumann entropy (non-negative)
- `time`: Evolution time
- `energy`: Total field energy

**Methods:**

- `to_dict()`: Convert state to serializable dictionary

## Example

See `examples/qcmg_demo.py` for a complete demonstration:

```bash
cd /path/to/QuASIM
python3 examples/qcmg_demo.py
```

## Physical Interpretation

The QCMG module simulates a coupled system where:

1. **Physical field (Φ_m)**: Represents material/energy density
2. **Informational field (Φ_i)**: Represents entropy and complexity
3. **Coupling**: Models quantum-classical information transfer
4. **Evolution**: Shows emergence of structure from initial conditions

### Observable Dynamics

- **Coherence decay**: Measures quantum-to-classical transition
- **Entropy production**: Demonstrates second law of thermodynamics
- **Energy flow**: Shows energy transfer between subsystems

## Theoretical Background

The implementation is based on:

- Quantum field theory (QFT) formalism
- Information geometry principles
- Non-equilibrium thermodynamics
- Hamiltonian dynamics with dissipation

## Testing

Run the test suite:

```bash
pytest tests/software/test_qcmg_field.py -v
```

All 21 tests should pass, covering:

- Parameter configuration
- Field initialization
- Evolution dynamics
- Physical constraints (coherence, entropy, energy)
- State serialization
- Reproducibility

## Performance

- Grid size scales as O(N) for spatial operations
- RK4 integration provides high accuracy with moderate cost
- Typical run: 64-point grid, 1000 steps completes in ~1 second
- Memory usage: Dominated by history storage if enabled

## License

Part of the QuASIM framework. See repository root for license information.
