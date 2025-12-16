# QRATUM 

### Classical Simulation Framework with Planned Quantum Extensions
High-Assurance • Deterministic • Modular • Multi-Domain Scientific Computing

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-beta-yellow.svg)](QUANTUM_INTEGRATION_ROADMAP.md)


QRATUM is a deterministic, classical simulation framework designed for reproducible multi-domain modeling in research environments. 
It provides a solid foundation for numerical simulation with plans to integrate genuine quantum computing capabilities in future versions.


**Current Capabilities** ✅:
- **Classical Numerical Simulation**: NumPy-based computational framework
- **Deterministic Execution**: Reproducible results via seed management
- **Modular Architecture**: Well-organized codebase for scientific computing
- **Configuration Management**: Runtime contexts and parameter handling
- **Development Tooling**: pytest, ruff, CI/CD infrastructure

**NOT Currently Implemented** ❌:
- Quantum computing libraries (no Qiskit, PennyLane, Cirq)
- Actual quantum circuit simulation
- Real QAOA, VQE, or quantum algorithms
- Quantum hardware or simulator backends
- cuQuantum or GPU quantum acceleration


---

## Current Features (Classical Computing)

QRATUM v2.0 provides:

- **Deterministic Execution**: Seeded randomness for reproducible simulations
- **Classical Numerical Methods**: NumPy-based scientific computing
- **Modular Design**: Clean separation of concerns for extensibility
- **Configuration Management**: Runtime contexts and parameter validation
- **Optimization Placeholders**: Framework for future quantum algorithm integration
- **Development Infrastructure**: Comprehensive testing and CI/CD

**Use Cases**:
- Numerical simulation development and prototyping
- Classical optimization algorithm research
- Deterministic computation workflows
- Educational projects in scientific computing  
 

---

## Current Capabilities (Detailed)

### Classical Simulation
- Deterministic execution with seed management
- Basic numerical computation primitives
- Configuration and runtime management
- Modular architecture for extension

### Development Infrastructure
- Python 3.10+ support
- pytest-based testing framework
- ruff for code quality
- CI/CD via GitHub Actions
- Type hints and documentation

---

## Architecture

QRATUM follows a modular design:

```
qratum/
├── quasim/              # Legacy simulation modules
│   ├── opt/             # Optimization framework (classical + placeholders)
│   ├── api/             # API interfaces
│   ├── sim/             # Simulation primitives
│   └── hcal/            # Hardware abstraction
├── qstack/              # Stack management utilities
├── qubic/               # Visualization components
├── tests/               # Comprehensive test suite
└── docs/                # Documentation
```

### Design Principles
- **Modularity**: Clean interfaces between components
- **Testability**: Comprehensive test coverage
- **Extensibility**: Ready for quantum algorithm integration
- **Reproducibility**: Deterministic execution via seed management

---

## Installation

### Prerequisites
- Python 3.10 or later
- pip or conda package manager

### Basic Installation

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=quasim tests/

# Run specific test module
pytest tests/test_specific.py
```  

---

## Usage Examples

### Basic Simulation (Classical)

```python
from quasim import Config, runtime

# Create configuration
config = Config(
    simulation_precision="fp32",
    backend="cpu",
    seed=42  # For reproducibility
)

# Run simulation
with runtime(config) as rt:
    # Simplified example - actual usage depends on specific modules
    result = rt.simulate(circuit_data)
    print(f"Simulation completed with latency: {rt.average_latency}s")
```

### Using Optimization Framework

```python
from quasim.opt.optimizer import HybridOptimizer  # Classical implementation
from quasim.opt.problems import OptimizationProblem

# Note: Despite the name, this currently uses classical random search
# See QUANTUM_CAPABILITY_AUDIT.md for details
optimizer = HybridOptimizer(
    algorithm="random_search",  # Honest naming
    backend="cpu",
    max_iterations=100,
    random_seed=42
)

# Define your problem
problem = OptimizationProblem(...)

# Optimize (classically)
result = optimizer.optimize(problem)
print(f"Best solution: {result['solution']}")
print(f"Objective value: {result['objective_value']}")
```  
---

## License

Apache 2.0 License - See [LICENSE](LICENSE) file for details.

