# Installation

This guide covers all installation methods for QRATUM.

## Installation Methods

| Method | Best For | Time |
|--------|----------|------|
| [pip install](#pip-installation) | Development | 2 min |
| [Docker Compose](#docker-installation) | Quick start | 3 min |
| [From Source](#from-source) | Contributors | 5 min |

## pip Installation

### Basic Installation (Classical + Quantum)

```bash
pip install qratum
```

Or install from the repository:

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
pip install -r requirements.txt
```

### Optional Dependencies

=== "Quantum Computing"

    ```bash
    pip install qratum[quantum]
    ```
    
    Includes: Qiskit, PennyLane, PySCF

=== "Development"

    ```bash
    pip install qratum[dev]
    ```
    
    Includes: pytest, ruff, mypy, coverage tools

=== "Visualization"

    ```bash
    pip install qratum[viz]
    ```
    
    Includes: imageio, plotly, torch, websockets

=== "All Dependencies"

    ```bash
    pip install qratum[quantum,dev,viz]
    ```

## Docker Installation

### Using Docker Compose (Recommended)

```bash
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM
docker-compose up --build
```

### Using Docker Directly

```bash
docker build -t qratum:latest .
docker run -p 8000:8000 qratum:latest
```

## From Source

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation

```python
from quasim.quantum import check_quantum_dependencies, get_quantum_status

print(get_quantum_status())
# Output: "Quantum computing enabled with: qiskit, pennylane"
```

## IBM Quantum Hardware Access (Optional)

To run on real quantum hardware:

1. Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API token from the account dashboard
3. Configure in code:

```python
from quasim.quantum.core import QuantumConfig

config = QuantumConfig(
    backend_type="ibmq",
    ibmq_token="YOUR_API_TOKEN_HERE",
    shots=1024
)
```

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| RAM | 4 GB |
| Disk Space | 2 GB |
| OS | Linux, macOS, Windows |

### Recommended for Production

| Component | Requirement |
|-----------|-------------|
| Python | 3.11+ |
| RAM | 16 GB+ |
| Disk Space | 10 GB+ |
| GPU | NVIDIA CUDA-capable |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QRATUM_LOG_LEVEL` | Logging verbosity | `INFO` |
| `QRATUM_SEED` | Default random seed | `None` |
| `JAX_PLATFORM_NAME` | JAX backend | `cpu` |

## Next Steps

- [Local Development Setup](local-development.md)
- [Run Your First Simulation](first-simulation.md)
- [Configuration Reference](../reference/configuration.md)
