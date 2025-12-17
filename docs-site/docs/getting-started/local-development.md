# Local Development Setup

This guide helps you set up a complete local development environment for QRATUM.

## Prerequisites

- Python 3.10+
- Git
- Make (optional, for convenience commands)
- Docker (for integration testing)

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/robertringler/QRATUM.git
cd QRATUM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install all development dependencies
pip install -e ".[dev,quantum,viz]"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Or using pytest directly
pytest tests/

# Run with coverage
pytest --cov=quasim tests/

# Run only quantum tests
pytest tests/quantum/

# Skip slow tests
pytest -m "not slow"
```

### Code Formatting

```bash
# Format code
make fmt

# Or manually
ruff format .
```

### Linting

```bash
# Run linters
make lint

# Or manually
ruff check .
```

### Building

```bash
# Validate Python modules
make build

# Build documentation
cd docs-site && mkdocs build
```

## Directory Structure

```
QRATUM/
├── quasim/              # Core simulation engine
│   ├── quantum/         # Quantum algorithms (VQE, QAOA)
│   ├── opt/             # Classical optimization
│   ├── sim/             # Simulation primitives
│   ├── api/             # API interfaces
│   └── hcal/            # Hardware calibration
├── examples/            # Usage examples
├── tests/               # Test suite
├── docs-site/           # Documentation source
├── integrations/        # External tool adapters
├── compliance/          # Compliance automation
├── infra/               # Infrastructure configs
└── autonomous_systems_platform/  # Phase III RL optimization
```

## IDE Configuration

### VS Code

Recommended extensions (`.vscode/extensions.json`):

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "ms-toolsai.jupyter"
  ]
}
```

### PyCharm

1. Open the QRATUM directory as a project
2. Configure Python interpreter to use `.venv`
3. Enable ruff as the external linter

## Running Services Locally

### Backend Only

```bash
cd autonomous_systems_platform/services/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export JAX_PLATFORM_NAME=cpu
python app.py
```

Backend available at `http://localhost:8000`

### Frontend Only

```bash
cd autonomous_systems_platform/services/frontend
python -m http.server 8080
```

Frontend available at `http://localhost:8080`

### Full Stack (Docker Compose)

```bash
docker-compose up --build
```

## Common Development Tasks

### Adding a New Feature

1. Create a feature branch
2. Write tests first (TDD approach)
3. Implement the feature
4. Run `make lint` and `make test`
5. Update documentation
6. Create a pull request

### Running Benchmarks

```bash
make bench
```

### Generating Video Artifacts

```bash
make video
```

## Troubleshooting

### Import Errors

Ensure you've installed in development mode:

```bash
pip install -e .
```

### Test Failures

Check that all dependencies are installed:

```bash
pip install -e ".[dev,quantum]"
```

### GPU Not Detected

Set the JAX platform explicitly:

```bash
export JAX_PLATFORM_NAME=cpu  # For CPU-only
export JAX_PLATFORM_NAME=cuda  # For NVIDIA GPU
```

## Next Steps

- [Run Your First Simulation](first-simulation.md)
- [Contributing Guide](../contributing.md)
- [CI/CD Integration](../tutorials/cicd-integration.md)
