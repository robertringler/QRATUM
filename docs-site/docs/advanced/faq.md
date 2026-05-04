# Frequently Asked Questions

Common questions about QRATUM.

## General

### What is QRATUM?

QRATUM (Quantum Resource Allocation, Tensor Analysis, and Unified Modeling) is a hybrid quantum-classical simulation platform. It implements genuine quantum algorithms (VQE, QAOA) while maintaining aerospace certification compliance (DO-178C Level A).

### What does QRATUM stand for?

**Q**uantum **R**esource **A**llocation, **T**ensor Analysis, and **U**nified **M**odeling

### Is QRATUM open source?

Yes, QRATUM is released under the Apache 2.0 License.

### What's the relationship to QuASIM?

QuASIM was the former name of the project. All references to QuASIM now refer to QRATUM.

---

## Quantum Computing

### Does QRATUM provide quantum speedup?

**No, not currently.** QRATUM runs on classical simulators that simulate quantum behavior. For the problem sizes currently supported (2-20 qubits), classical methods are faster.

### When will QRATUM provide quantum advantage?

Quantum advantage for molecular simulation requires:

- Error-corrected quantum computers (not available in 2025)
- Hundreds to thousands of logical qubits
- Problem sizes beyond classical simulation capability

Current estimates suggest 2030+ for practical quantum advantage in chemistry.

### What quantum algorithms does QRATUM implement?

- **VQE** (Variational Quantum Eigensolver) - Molecular ground states
- **QAOA** (Quantum Approximate Optimization Algorithm) - Combinatorial optimization

### Can I run QRATUM on real quantum hardware?

Yes, with an IBM Quantum account:

```python
config = QuantumConfig(
    backend_type="ibmq",
    ibmq_token="your_token"
)
```

Note: Real hardware has significant limitations (noise, queue times).

### What's the maximum qubit count?

- Simulator: ~25-30 qubits (memory limited)
- IBM Quantum: 127+ qubits (device dependent)
- Practical limit for useful computation: ~10-20 qubits (NISQ noise)

---

## Installation

### What Python version do I need?

Python 3.10 or later. We recommend Python 3.11.

### Do I need a GPU?

No, GPUs are optional. They provide acceleration for larger simulations but are not required.

### How do I install cuQuantum?

```bash
# Requires NVIDIA GPU and CUDA
pip install cuquantum-python

# Or install from NVIDIA directly
# See: https://docs.nvidia.com/cuda/cuquantum/
```

### Why is installation slow?

Some dependencies (Qiskit, PySCF) require compilation. This is normal and may take several minutes.

---

## Usage

### How do I get reproducible results?

Use the `seed` parameter:

```python
config = QuantumConfig(seed=42)
# Same seed = same results
```

### How many shots should I use?

| Use Case | Shots | Accuracy |
|----------|-------|----------|
| Development | 100-500 | Low |
| Testing | 1024 | Medium |
| Production | 4096+ | High |

### How do I speed up computations?

See [Performance Tuning](performance-tuning.md). Quick tips:

1. Reduce shots during development
2. Use GPU acceleration if available
3. Cache compiled circuits

### Can I parallelize computations?

Yes, using Python multiprocessing:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(compute_energy, parameters))
```

---

## Compliance

### Is QRATUM DO-178C certified?

QRATUM implements DO-178C Level A requirements and is designed for certification. Actual certification requires DER review and FAA approval for specific applications.

### What compliance frameworks does QRATUM support?

- DO-178C Level A (Aerospace)
- NIST 800-53 Rev 5 (Federal)
- CMMC 2.0 Level 2 (Defense)
- DFARS (Defense Contracts)
- FIPS 140-3 (Cryptography)
- ISO 27001 (Information Security)

### How do I generate compliance reports?

```bash
# SBOM generation
python compliance/scripts/sbom_generator.py

# MC/DC coverage
python compliance/scripts/mcdc_analyzer.py

# Full compliance report
python compliance/scripts/generate_compliance_report.py
```

---

## Deployment

### Can I deploy QRATUM to production?

QRATUM includes production-ready Kubernetes configurations. However, the quantum simulation components are research-grade and should not be used for mission-critical decisions.

### What cloud providers are supported?

- Amazon Web Services (EKS)
- Google Cloud Platform (GKE)
- Microsoft Azure (AKS)
- Any Kubernetes cluster

### How do I monitor QRATUM in production?

QRATUM integrates with:

- Prometheus (metrics)
- Grafana (dashboards)
- Loki (logs)

See [Kubernetes Deployment](../tutorials/kubernetes-deployment.md).

---

## Development

### How do I contribute?

See [Contributing Guide](../contributing.md). Key steps:

1. Fork the repository
2. Create a feature branch
3. Write tests
4. Submit a pull request

### How do I run tests?

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=quasim tests/

# Skip slow tests
pytest -m "not slow"
```

### How do I format code?

```bash
# Format
make fmt

# Or manually
ruff format .
ruff check . --fix
```

---

## Troubleshooting

### VQE isn't converging

Try:

1. Increase `max_iterations`
2. Use a different optimizer (`COBYLA`, `L-BFGS-B`, `SLSQP`)
3. Increase `ansatz_layers`
4. Check your input parameters

### I'm getting memory errors

Try:

1. Reduce qubit count
2. Reduce shot count
3. Set memory limits: `JAX_PLATFORM_NAME=cpu`
4. Use batching for large computations

### Real hardware results are wrong

Real quantum hardware has significant noise. Try:

1. Increasing shot count
2. Using error mitigation (when available)
3. Running multiple times and averaging
4. Using simulator for validation

---

## Licensing

### Can I use QRATUM commercially?

Yes, under the Apache 2.0 License. You can:

- Use in commercial products
- Modify and distribute
- Patent derivative works

You must:

- Include the license and copyright notice
- State significant changes

### Are there export controls?

QRATUM source code is not export-controlled. However, applications using QRATUM for defense or aerospace may be subject to ITAR/EAR. Consult legal counsel for specific applications.

---

## Getting Help

### Where can I get help?

1. **Documentation:** You're here!
2. **GitHub Issues:** [Open an issue](https://github.com/robertringler/QRATUM/issues)
3. **Discussions:** GitHub Discussions (if enabled)

### How do I report bugs?

Open a GitHub issue with:

- QRATUM version
- Python version
- Operating system
- Full error traceback
- Steps to reproduce

### How do I report security issues?

**Do not open public issues for security vulnerabilities.**

Email security concerns privately to the maintainers. See [SECURITY.md](https://github.com/robertringler/QRATUM/blob/main/SECURITY.md).
