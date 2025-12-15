# QuASIM → QRATUM Migration Guide

This guide helps you migrate from QuASIM to QRATUM (Quantum Resource Allocation, Tensor Analysis, and Unified Modeling).

## Why the Rebrand?

QuASIM has evolved into a comprehensive quantum simulation platform and has been rebranded to **QRATUM** to better reflect its expanded capabilities:

- **Quantum Resource Allocation**: Intelligent backend selection and resource management
- **Tensor Analysis**: Advanced tensor network simulation for large qubit counts
- **Unified Modeling**: Consistent API across CPU, GPU, and multi-GPU backends

## What's Changed?

### Package Name

- **Old**: `quasim`
- **New**: `qratum`

### Version

- **QuASIM**: `0.1.0` (deprecated)
- **QRATUM**: `2.0.0` (current)

### Timeline

- **Now**: Both `quasim` and `qratum` work (with deprecation warning)
- **Version 2.x**: Backward compatibility maintained
- **Version 3.0.0**: `quasim` compatibility shim removed

## Migration Steps

### 1. Install QRATUM

```bash
# If you have quasim installed
pip uninstall quasim

# Install qratum
pip install qratum
```

### 2. Update Imports

#### Basic Import Changes

**Before (QuASIM):**
```python
from quasim import Simulator, Circuit
from quasim.core import gates
```

**After (QRATUM):**
```python
from qratum import Simulator, Circuit
from qratum.core import gates
```

#### Detailed Import Mapping

| QuASIM Import | QRATUM Import |
|--------------|---------------|
| `from quasim import Config` | `from qratum import QRATUMConfig` |
| `from quasim import Runtime` | Use `Simulator` instead |
| `from quasim.qc import *` | `from qratum.core import *` |
| `import quasim` | `import qratum` |

### 3. Update Code

#### Creating a Simulator

**Before (QuASIM):**
```python
from quasim import Config, runtime

config = Config(
    simulation_precision="fp32",
    backend="cpu",
    seed=42
)

with runtime(config) as rt:
    result = rt.simulate(circuit)
```

**After (QRATUM):**
```python
from qratum import Simulator

simulator = Simulator(
    backend="cpu",
    precision="fp32",
    seed=42
)

result = simulator.run(circuit, shots=1024)
```

#### Building Circuits

**Before (QuASIM distributed):**
```python
from quasim.qc.circuit import Circuit
from quasim.qc.gates import H, CNOT

circuit = Circuit(2)
# Apply gates manually
```

**After (QRATUM):**
```python
from qratum import Circuit

circuit = Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)
# Fluent API for easy circuit building
```

### 4. API Changes

#### State Vector Access

**Before:**
```python
# State vector was internal
result = runtime.simulate(circuit)
```

**After:**
```python
# Explicit state vector access
state = simulator.run_statevector(circuit)
print(state)  # Human-readable output
probs = state.probabilities()
```

#### Measurement Results

**Before:**
```python
# Simple list output
result = runtime.simulate(circuit)
```

**After:**
```python
# Rich Result object
result = simulator.run(circuit, shots=1024)
counts = result.get_counts()
probs = result.get_probabilities()
most_frequent = result.most_frequent(5)
```

### 5. Backend Selection

QRATUM introduces intelligent automatic backend selection:

```python
from qratum import Simulator

# Automatic backend selection based on circuit size
sim = Simulator(backend="auto")

# Or explicitly choose backend
sim_cpu = Simulator(backend="cpu")
sim_gpu = Simulator(backend="gpu")
sim_multi = Simulator(backend="multi-gpu")
sim_tn = Simulator(backend="tensor-network")  # For >40 qubits
```

**Auto-selection logic:**
- 1-10 qubits: `cpu`
- 11-32 qubits: `gpu` (if available, else `cpu`)
- 33-40 qubits: `multi-gpu` (if available, else `tensor-network`)
- 40+ qubits: `tensor-network`

## Example: Complete Migration

### Before (QuASIM)

```python
from quasim import Config, runtime
from quasim.qc.circuit import Circuit

config = Config(backend="cpu", seed=42)

# Build circuit
circuit = [[1, 0], [0, 1]]  # Identity gate

with runtime(config) as rt:
    result = rt.simulate(circuit)
    print(f"Latency: {rt.average_latency}")
```

### After (QRATUM)

```python
from qratum import Simulator, Circuit

# Build circuit with fluent API
circuit = Circuit(2)
circuit.h(0)       # Hadamard on qubit 0
circuit.cnot(0, 1)  # CNOT with control=0, target=1

# Run simulation
simulator = Simulator(backend="cpu", seed=42)
result = simulator.run(circuit, shots=1024)

# Analyze results
print(result)  # Pretty-printed results
print(f"Most frequent state: {result.most_frequent(1)}")

# Get state vector without measurement
state = simulator.run_statevector(circuit)
print(f"State amplitudes: {state}")
```

## Backward Compatibility Mode

During the transition period (QRATUM 2.x), you can use the compatibility layer:

```python
# This works but shows deprecation warning
import quasim

# Access QRATUM features through quasim
circuit = quasim.QuantumCircuit(2)
circuit.h(0)
circuit.cnot(0, 1)

sim = quasim.QuantumSimulator(backend="cpu")
result = sim.run(circuit)
```

## Configuration Changes

### Environment Variables

**Before (QuASIM):**
- No environment variable support

**After (QRATUM):**
```bash
export QRATUM_BACKEND=gpu
export QRATUM_PRECISION=fp32
export QRATUM_SEED=42
export QRATUM_LOG_LEVEL=INFO
```

### Configuration Files

QRATUM supports `.qratum/` directory in home:

```bash
~/.qratum/
├── cache/           # Compiled circuit cache
└── config.yaml      # Global configuration (future)
```

## Feature Additions in QRATUM

### New in QRATUM 2.0

1. **Fluent Circuit API**: Chain gate operations
2. **Rich Measurement Results**: Comprehensive result analysis
3. **Density Matrix Support**: For mixed states and noise
4. **Auto Backend Selection**: Intelligent resource allocation
5. **Algorithm Library**: VQE, QAOA, Grover, Shor, QFT, etc.
6. **Chemistry Module**: Molecular simulation with PySCF
7. **Machine Learning**: Quantum neural networks and kernels
8. **Noise Models**: Realistic noise simulation
9. **Error Mitigation**: ZNE, PEC, readout correction

### New Modules

- `qratum.algorithms`: Pre-built quantum algorithms
- `qratum.chemistry`: Quantum chemistry simulations
- `qratum.ml`: Quantum machine learning
- `qratum.noise`: Noise models and error mitigation
- `qratum.backends`: Multiple backend implementations

## Testing Your Migration

Create a test script to verify your migration:

```python
# test_migration.py
import qratum

def test_basic_circuit():
    """Test basic circuit works."""
    circuit = qratum.Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    simulator = qratum.Simulator(backend="cpu", seed=42)
    result = simulator.run(circuit, shots=1000)
    
    counts = result.get_counts()
    assert len(counts) == 2  # Should have |00⟩ and |11⟩
    assert "00" in counts
    assert "11" in counts
    print("✓ Basic circuit test passed")

def test_statevector():
    """Test state vector access."""
    circuit = qratum.Circuit(1)
    circuit.h(0)
    
    simulator = qratum.Simulator(backend="cpu")
    state = simulator.run_statevector(circuit)
    
    probs = state.probabilities()
    assert abs(probs[0] - 0.5) < 0.01
    assert abs(probs[1] - 0.5) < 0.01
    print("✓ State vector test passed")

if __name__ == "__main__":
    test_basic_circuit()
    test_statevector()
    print("\n✓ All migration tests passed!")
```

Run with:
```bash
python test_migration.py
```

## Getting Help

- **Documentation**: https://qratum.io/docs
- **GitHub Issues**: https://github.com/robertringler/QRATUM/issues
- **Migration Questions**: Tag with `migration` label

## Deprecation Timeline

| Version | Status | Notes |
|---------|--------|-------|
| 2.0.0 | Current | Full backward compatibility |
| 2.1.0 - 2.x | Stable | Deprecation warnings only |
| 3.0.0 | Breaking | `quasim` shim removed |

## Checklist

- [ ] Updated all `import quasim` to `import qratum`
- [ ] Updated all `from quasim` to `from qratum`
- [ ] Migrated `Config` to `QRATUMConfig` or `Simulator` args
- [ ] Migrated `runtime()` context to `Simulator` class
- [ ] Updated circuit building to use fluent API
- [ ] Updated measurement handling to use `Result` objects
- [ ] Tested all quantum functionality
- [ ] Removed `quasim` from requirements.txt
- [ ] Updated documentation and comments
- [ ] Verified no deprecation warnings

## Summary

The migration from QuASIM to QRATUM is straightforward:

1. **Change package name**: `quasim` → `qratum`
2. **Update imports**: Adjust import statements
3. **Use new API**: Simulator, Circuit, Result classes
4. **Test thoroughly**: Run your test suite
5. **Enjoy new features**: Algorithms, chemistry, ML, noise models

The QRATUM team is committed to making this transition smooth. The backward compatibility layer ensures your code continues working while you migrate at your own pace.

---

**Questions?** Open an issue on GitHub or check the documentation.

**Ready to migrate?** Start with updating imports and testing basic circuits.

**Want to explore new features?** Check out `examples/` directory for comprehensive demos.
