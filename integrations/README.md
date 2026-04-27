# QuASIM Integrations

Enterprise-grade adapters, SDKs, and services for integrating QuASIM across aerospace simulation workflows.

## Overview

This directory contains all integration components for QuASIM:

- **Adapters**: Drop-in connectors for commercial simulation tools
- **Services**: REST/gRPC API for job management
- **SDKs**: Client libraries (Python, C++)
- **Kernels**: CUDA/JAX compute primitives
- **Benchmarks**: Performance validation suite
- **Infrastructure**: IaC, containers, Kubernetes manifests
- **Compliance**: SBOM, security, export control documentation

## Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install numpy pyyaml

# Optional: FastAPI for API service
pip install fastapi uvicorn

# Optional: Testing
pip install pytest pytest-cov
```

### 2. Run Fluent Adapter

```bash
# Create test data
mkdir -p /tmp/quasim_demo
echo '{"solver": "pressure_poisson", "max_iterations": 100, "convergence_tolerance": 1e-6}' > /tmp/quasim_demo/job.json
echo "# Test mesh" > /tmp/quasim_demo/mesh.msh

# Run adapter
python3 integrations/adapters/fluent/quasim_fluent_driver.py \
  --mesh /tmp/quasim_demo/mesh.msh \
  --job /tmp/quasim_demo/job.json \
  --output /tmp/quasim_demo/results.csv
```

### 3. Run Benchmarks

```bash
# Quick benchmarks (reduced problem sizes)
python3 integrations/benchmarks/aero/run_benchmarks.py --quick

# Full benchmarks
python3 integrations/benchmarks/aero/run_benchmarks.py
```

### 4. Start API Service (Optional)

```bash
# Requires FastAPI
python3 integrations/services/quasim-api/server.py
# Navigate to http://localhost:8000/docs for API documentation
```

## Directory Structure

```
integrations/
├── adapters/           # Simulation tool adapters
│   ├── fluent/        # Ansys Fluent adapter
│   ├── starccm/       # Siemens STAR-CCM+ adapter
│   ├── abaqus/        # Dassault Abaqus adapter
│   ├── fun3d/         # NASA FUN3D adapter
│   └── omniverse/     # NVIDIA Omniverse adapter
├── services/          # API services
│   └── quasim-api/    # REST/gRPC job service
├── sdk/               # Client libraries
│   ├── python/        # Python SDK
│   └── cpp/           # C++ SDK with CMake
├── kernels/           # Compute kernels
│   ├── cfd/           # CFD kernels (pressure Poisson, etc.)
│   ├── materials/     # Material micro-solvers
│   └── orbital/       # Orbital mechanics kernels
├── benchmarks/        # Performance benchmarks
│   └── aero/          # Aerospace scenarios
├── infra/             # Infrastructure as Code
│   ├── terraform/     # AWS/EKS provisioning
│   ├── helm/          # Helm charts
│   ├── k8s/           # Kubernetes manifests
│   └── docker/        # Dockerfiles
├── compliance/        # Security & compliance
│   └── EXPORT.md      # ITAR export control guide
└── README.md          # This file
```

## Adapters

### Fluent Adapter

Read-only file shim for Ansys Fluent:

```bash
quasim_fluent_driver --mesh wing.msh --job config.json --output results.csv
```

Features:

- Reads Fluent mesh and boundary conditions
- Converts to QuASIM tensor format
- Runs CFD kernels
- Writes results as CSV/HDF5/VTK

### STAR-CCM+ Adapter

Java macro for Siemens STAR-CCM+:

```java
// Run from STAR-CCM+ macro menu
quasim.adapters.starccm.QuASIMStarCCMMacro
```

### Abaqus Adapter

UMAT subroutine for Dassault Abaqus:

```fortran
*Material, name=QuASIM_Material
*User Material, constants=3
```

### FUN3D Adapter

Python wrapper for NASA FUN3D:

```bash
quasim_fun3d_wrapper --flow flow.dat --mesh mesh.ugrid --output fields.dat
```

### Omniverse Adapter

OmniGraph node for NVIDIA Omniverse:

```python
from quasim.adapters.omniverse import QuASIMOmniGraphNode
node = QuASIMOmniGraphNode("physics_node")
```

## SDKs

### Python SDK

```python
from quasim_client import QuASIMClient

client = QuASIMClient(api_url="http://localhost:8000")

# Submit CFD job
job = client.submit_cfd(
    mesh_file="wing.msh",
    config={"solver": "pressure_poisson", "max_iterations": 1000}
)

# Wait for completion
result = client.wait_for_completion(job.job_id)
```

### C++ SDK

```cpp
#include <quasim_client.hpp>

quasim::Client client("localhost:50051");

// Submit CFD job
auto job = client.submit_cfd("wing.msh", config);

// Wait for completion
auto result = client.wait_for_completion(job.id);
```

## Benchmarks

Run aerospace benchmarks to validate performance:

```bash
# Quick benchmarks
make bench

# Or directly
python3 integrations/benchmarks/aero/run_benchmarks.py --quick
```

Benchmark output:

- `benchmarks/aero/report/perf.csv` - Performance data
- `benchmarks/aero/report/perf_substitution_table.md` - Markdown report

Expected results:

- ≥10× throughput improvement over legacy solvers
- Energy and cost metrics (kWh, $/sim)
- RMSE vs. reference solutions

## API Service

FastAPI + gRPC service for job management:

**REST Endpoints:**

- `POST /jobs/submit` - Submit new job
- `GET /jobs/{id}/status` - Get job status
- `POST /jobs/{id}/cancel` - Cancel job
- `GET /artifacts/{id}` - Download artifact
- `GET /metrics` - System metrics
- `GET /validate` - Validate job config

**gRPC Service:** (planned)

- High-performance job submission
- Streaming telemetry
- Bidirectional communication

## Infrastructure

### Docker

Build containers:

```bash
# API service
docker build -f integrations/infra/docker/Dockerfile.api -t quasim-api .

# Benchmark runner
docker build -f integrations/infra/docker/Dockerfile.bench -t quasim-bench .
```

### Kubernetes

Deploy to Kubernetes:

```bash
# Using Helm
helm install quasim integrations/infra/helm/quasim-api/

# Using kubectl
kubectl apply -f integrations/infra/k8s/
```

### Terraform

Provision infrastructure:

```bash
cd integrations/infra/terraform/
terraform init
terraform apply
```

## Compliance & Security

### ITAR Export Control

See [compliance/EXPORT.md](compliance/EXPORT.md) for ITAR compliance procedures.

### Security

- SBOM generation with Syft
- Container signing with cosign
- SAST scanning with CodeQL
- License scanning with FOSSA/OSS Review Toolkit

### DO-178C Compliance

For safety-critical aerospace code:

- MISRA-like coding standards
- Static analysis (clang-tidy, cppcheck)
- >90% unit test coverage
- Requirements traceability

## Testing

Run tests:

```bash
# Full test suite
make test

# Python tests only
pytest tests/

# Integration tests
pytest tests/integration/
```

## Documentation

Additional documentation:

- Integration cookbooks (coming soon)
- Performance tuning guide (coming soon)
- API reference (OpenAPI at `/docs`)
- Examples (coming soon)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## Support

For questions or issues:

1. Check documentation
2. Search existing issues
3. Open a new issue with the `integrations` label

## License

See [LICENSE](../LICENSE) file for licensing information.
