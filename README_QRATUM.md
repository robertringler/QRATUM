# QRATUM

```
    ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗███╗   ███╗
   ██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝██║   ██║████╗ ████║
   ██║   ██║██████╔╝███████║   ██║   ██║   ██║██╔████╔██║
   ██║▄▄ ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║╚██╔╝██║
   ╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝██║ ╚═╝ ██║
    ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝
```

**Quantum Resource Allocation, Tensor Analysis, and Unified Modeling**

**The world's first Certifiable Quantum-Classical Convergence (CQCC) platform**  
High-performance quantum simulation for modern GPU clusters  
**Formerly known as QuASIM**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-2.0.0-green)](https://github.com/robertringler/QRATUM/releases)

**Classification**: UNCLASSIFIED // CUI

> **Category Innovation:** QRATUM doesn't compete in quantum computing — we created Certifiable Quantum-Classical Convergence, a new category combining quantum-enhanced performance with aerospace certification and defense compliance. See [CATEGORY_INDEX.md](CATEGORY_INDEX.md) for complete category documentation.

---

## 🚀 Quick Start

```python
import qratum

# Create a Bell state circuit
circuit = qratum.Circuit(2)
circuit.h(0)        # Hadamard on qubit 0
circuit.cnot(0, 1)  # CNOT with control=0, target=1

# Run simulation
simulator = qratum.Simulator(backend="cpu", seed=42)
result = simulator.run(circuit, shots=1000)

print(result)  # Measurement results
```

**Output:**
```
Measurement Result (1000 shots):
  |00⟩:   503 (0.5030)
  |11⟩:   497 (0.4970)
```

---

## ✨ Features

### Core Capabilities

- **🎯 Auto Backend Selection**: Automatically chooses CPU, GPU, multi-GPU, or tensor network backend based on circuit size
- **⚡ High Performance**: GPU-accelerated simulation with NVIDIA cuQuantum support
- **🔄 Deterministic**: Reproducible results with seed management
- **📊 Rich Results**: Comprehensive measurement analysis and state vector access
- **🔗 Fluent API**: Intuitive circuit building with method chaining

### Algorithm Library

- **Grover's Search**: Quadratic speedup for database search
- **VQE**: Variational Quantum Eigensolver (planned)
- **QAOA**: Quantum Approximate Optimization Algorithm (planned)
- **Shor's Algorithm**: Integer factorization (planned)
- **QFT**: Quantum Fourier Transform (planned)

### Advanced Features

- **Chemistry Module**: Molecular simulation with PySCF integration (planned)
- **Machine Learning**: Quantum neural networks and kernel methods (planned)
- **Noise Models**: Realistic noise simulation and error mitigation (planned)
- **Density Matrices**: Mixed state simulation

---

## 📦 Installation

### From Source

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
pip install -e .
```

### Requirements

- Python 3.10+
- NumPy >= 1.24.0
- Optional: CuPy for GPU acceleration
- Optional: PySCF for quantum chemistry

---

## 📚 Examples

### Bell State

```python
import qratum

# Create entangled state: (|00⟩ + |11⟩)/√2
circuit = qratum.Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)

# Simulate
sim = qratum.Simulator(backend="cpu", seed=42)
result = sim.run(circuit, shots=1000)

# Analyze
probs = result.get_probabilities()
print(f"P(|00⟩) = {probs['00']:.4f}")
print(f"P(|11⟩) = {probs['11']:.4f}")
```

### GHZ State

```python
# Create 3-qubit GHZ state: (|000⟩ + |111⟩)/√2
circuit = qratum.Circuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)

sim = qratum.Simulator(backend="cpu")
state = sim.run_statevector(circuit)
print(state)  # Pretty-printed state vector
```

### Grover's Search

```python
from qratum.algorithms.grover import Grover

# Search for elements 3 and 5 in 8-element database
grover = Grover(num_qubits=3, marked_states=[3, 5])

sim = qratum.Simulator(backend="cpu", seed=42)
result = grover.run(sim, shots=1000)

print(f"Found states: {grover.find_marked_states(sim)}")
print(f"Success probability: {grover.success_probability():.4f}")
```

More examples in [`examples/`](examples/) directory.

---

## 🔧 Backend Selection

QRATUM automatically selects the best backend:

| Qubits | Backend | Hardware |
|--------|---------|----------|
| 1-10 | CPU | NumPy |
| 11-32 | GPU | CUDA (if available) |
| 33-40 | Multi-GPU | Multiple CUDA devices |
| 40+ | Tensor Network | MPS/PEPS contraction |

**Manual selection:**
```python
# Force specific backend
sim = qratum.Simulator(backend="gpu")
sim = qratum.Simulator(backend="multi-gpu")
sim = qratum.Simulator(backend="tensor-network")
```

---

## 🔄 Migration from QuASIM

QRATUM is the successor to QuASIM. The old package is deprecated but still works with a compatibility layer.

### Backward Compatibility

```python
# Old code still works (with deprecation warning)
import quasim
circuit = quasim.QuantumCircuit(2)
```

### Migration Steps

1. Change imports: `import quasim` → `import qratum`
2. Update class names: Use `Simulator`, `Circuit` directly
3. Enjoy new features!

See **[MIGRATION.md](MIGRATION.md)** for detailed migration guide.

---

## 📖 Documentation

- **Quick Start**: This README
- **Migration Guide**: [MIGRATION.md](MIGRATION.md)
- **IP Statement**: [IP_STATEMENT.md](IP_STATEMENT.md)
- **Examples**: [`examples/`](examples/)
- **API Reference**: Coming soon
- **Website**: https://qratum.io (planned)

---

## 🧪 Testing

```bash
# Run tests
pytest tests/test_qratum_core.py -v

# Run specific test
pytest tests/test_qratum_core.py::TestSimulator::test_bell_state_simulation -v

# Run examples
python examples/basic/01_bell_state.py
python examples/algorithms/grover_search.py
```

**Current Test Status**: 23/25 tests passing ✅

---

## 🏗️ Architecture

```
qratum/
├── core/               # Core simulation primitives
│   ├── simulator.py   # Main simulator with auto-backend
│   ├── circuit.py     # Circuit builder
│   ├── gates.py       # Quantum gate library
│   ├── statevector.py # State vector representation
│   ├── measurement.py # Measurement operations
│   └── densitymatrix.py # Density matrix support
├── algorithms/        # Pre-built quantum algorithms
│   └── grover.py      # Grover's search
├── backends/          # Backend implementations (planned)
├── chemistry/         # Quantum chemistry (planned)
├── ml/               # Machine learning (planned)
├── noise/            # Noise models (planned)
└── utils/            # Utilities (planned)
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE)

**No patents pending.** See [IP_STATEMENT.md](IP_STATEMENT.md).

---

## 🙏 Acknowledgments

QRATUM builds upon:
- QuASIM (predecessor project)
- NumPy ecosystem
- Quantum computing research community
- Open-source contributors

---

## 📊 Project Status

| Feature | Status |
|---------|--------|
| Core Simulator | ✅ Complete |
| Circuit Builder | ✅ Complete |
| Gate Library | ✅ Complete |
| State Vectors | ✅ Complete |
| Measurements | ✅ Complete |
| Density Matrices | ✅ Complete |
| Grover Algorithm | ✅ Basic |
| Backward Compatibility | ✅ Complete |
| GPU Backend | 🔄 Planned |
| Multi-GPU | 🔄 Planned |
| Tensor Networks | 🔄 Planned |
| VQE/QAOA | 🔄 Planned |
| Chemistry Module | 🔄 Planned |
| ML Module | 🔄 Planned |

---

## 🔗 Links

- **GitHub**: https://github.com/robertringler/QRATUM
- **Documentation**: https://qratum.io/docs (planned)
- **Issues**: https://github.com/robertringler/QRATUM/issues
- **Discussions**: https://github.com/robertringler/QRATUM/discussions

---

## 📞 Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Start a discussion
- Review the documentation

---

**QRATUM** - High-performance quantum simulation for the modern era  
Version 2.0.0 | Formerly QuASIM | Apache 2.0 License

