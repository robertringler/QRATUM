# MERA Compression Benchmark Suite

Comprehensive benchmark suite for empirical validation of QRATUM's MERA (Multi-scale Entanglement Renormalization Ansatz) tensor compression algorithm.

## Overview

This benchmark suite generates compression artifacts across diverse quantum state types to validate:

- **Compression Ratios**: Measure actual compression vs. documented claims (10-50×, typ. 30×)
- **Fidelity Preservation**: Verify F(ρ, ρ') ≥ 0.995 constraint
- **Performance Characteristics**: Runtime and scalability analysis
- **Entanglement Preservation**: Test across different entanglement structures

## Quick Start

### Basic Usage

```bash
# Run full benchmark suite
python benchmarks/compression/run_suite.py

# Specify custom output directory
python benchmarks/compression/run_suite.py --output /path/to/artifacts

# Test specific qubit sizes
python benchmarks/compression/run_suite.py --qubits 10,15,20

# Use custom configuration
python benchmarks/compression/run_suite.py --config my_tests.yaml

# Enable verbose logging
python benchmarks/compression/run_suite.py --verbose
```

### Requirements

**Required:**
- Python 3.10+
- NumPy ≥ 1.24
- PyYAML ≥ 6.0
- QRATUM with `quasim.holo.anti_tensor` module

**Optional (for quantum circuit generation):**
- Qiskit ≥ 1.0.0
- Qiskit Aer ≥ 0.13.0

Install dependencies:
```bash
pip install numpy pyyaml qiskit qiskit-aer
```

## Test Cases

The suite includes the following test categories:

### 1. Random States
Baseline test with normalized random complex vectors.
- **Purpose**: Unstructured quantum states
- **Expected Compression**: 10-30× (moderate)

### 2. Product States
Separable states with no entanglement (|0⟩^⊗n, |+⟩^⊗n).
- **Purpose**: Best-case compression scenario
- **Expected Compression**: 40-100× (high)

### 3. GHZ States
Maximally entangled states: (|000...0⟩ + |111...1⟩) / √2
- **Purpose**: Structured entanglement
- **Expected Compression**: 30-50× (good)

### 4. W States
Symmetric entangled states with single excitations.
- **Purpose**: Different entanglement topology
- **Expected Compression**: 20-40× (moderate-good)

### 5. Random Quantum Circuits
States generated from random gate sequences.
- **Purpose**: Realistic quantum computation states
- **Expected Compression**: 15-35× (variable)

## Configuration

Edit `test_cases.yaml` to customize benchmarks:

```yaml
test_cases:
  - name: "custom_test"
    generator: "random"  # random, ghz, w, product, circuit
    qubits: [10, 15, 20]
    seeds: [42]
    fidelity_target: 0.995
    epsilon: 1e-3

compression_settings:
  fidelity_targets: [0.99, 0.995, 0.999]
  epsilon_sweep: [1e-2, 1e-3, 1e-4]

output:
  artifact_dir: "artifacts/compression"
  summary_json: "compression_summary.json"
  report_md: "compression_report.md"
```

## Output Artifacts

The benchmark suite generates:

### 1. Individual Test Artifacts (.npz files)
```
artifacts/compression/
├── random_states_10q_seed42.npz
├── ghz_states_15q.npz
├── w_states_20q.npz
└── ...
```

Each `.npz` file contains:
- `original`: Original quantum state vector
- `compressed`: Compressed representation
- `metadata`: Compression statistics (ratio, fidelity, runtime, etc.)

**Loading artifacts:**
```python
import numpy as np

data = np.load("artifacts/compression/random_states_10q_seed42.npz", allow_pickle=True)
original = data["original"]
compressed = data["compressed"]
metadata = data["metadata"][0]

print(f"Compression ratio: {metadata['compression_ratio']:.2f}×")
print(f"Fidelity: {metadata['fidelity_achieved']:.4f}")
```

### 2. Summary Report (JSON)
`compression_summary.json` contains aggregate statistics:
```json
{
  "total_tests": 45,
  "passed_tests": 43,
  "failed_tests": 2,
  "compression_ratio_stats": {
    "min": 8.3,
    "max": 51.2,
    "mean": 26.4,
    "median": 24.7,
    "std": 12.1
  },
  "fidelity_stats": {...},
  "by_state_type": {...}
}
```

### 3. Markdown Report
`compression_report.md` provides human-readable summary with:
- Summary statistics table
- Results by state type
- Comparison vs. documented claims
- Validation status

## Interpreting Results

### Compression Ratio
- **Excellent**: > 40× (product states, highly structured)
- **Good**: 25-40× (GHZ, W states)
- **Acceptable**: 10-25× (random states, circuits)
- **Poor**: < 10× (may indicate algorithm issues)

### Fidelity
- **Target**: F ≥ 0.995 (99.5% overlap)
- **High Quality**: F ≥ 0.999
- **Acceptable**: F ≥ 0.99
- **Low Quality**: F < 0.99 (investigate)

### Status Indicators
- ✅ **VALIDATED**: Meets or exceeds claimed typical performance
- ⚠️ **WITHIN_RANGE**: In claimed range but below typical
- ❌ **BELOW_MIN**: Below minimum claimed performance
- 🔸 **EXCEEDS_MAX**: Better than expected (verify)

## Adding Custom Test Cases

### 1. Add to Configuration
Edit `test_cases.yaml`:
```yaml
test_cases:
  - name: "my_custom_test"
    generator: "random"
    qubits: [12, 16, 20]
    seeds: [1, 2, 3]
    fidelity_target: 0.995
    epsilon: 1e-3
```

### 2. Add Custom Generator (Advanced)
Create a new generator in `generators/quantum_states.py`:
```python
def generate_custom_state(n_qubits: int, param: float) -> Array:
    """Generate custom quantum state."""
    dim = 2**n_qubits
    state = # ... your generation logic
    return state / np.linalg.norm(state)
```

Update `run_suite.py` to use new generator.

## Troubleshooting

### Issue: Import Error for AHTC
**Symptom:** `ImportError: cannot import name 'compress'`

**Solution:**
```bash
# Ensure QRATUM is properly installed
cd /path/to/QRATUM
pip install -e .

# Verify import works
python -c "from quasim.holo.anti_tensor import compress; print('OK')"
```

### Issue: Qiskit Not Available
**Symptom:** Warning about Qiskit unavailability

**Solution:**
```bash
pip install qiskit qiskit-aer

# Or run without circuit tests
# Edit test_cases.yaml to remove circuit test cases
```

### Issue: Out of Memory
**Symptom:** Tests crash for large qubit counts

**Solution:**
- Reduce maximum qubit count in configuration
- Typical limits: 20-25 qubits on standard hardware
- For > 25 qubits, use HPC resources with > 64GB RAM

### Issue: Tests Timeout
**Symptom:** Tests hang or take very long

**Solution:**
- Increase timeout in `test_cases.yaml`:
  ```yaml
  performance:
    timeout_per_test: 600  # seconds
  ```
- Reduce circuit depth for random circuit tests
- Use fewer epsilon values in sweeps

## Performance Expectations

Approximate runtimes on standard hardware (Intel i7, 16GB RAM):

| Qubits | State Dimension | Runtime per Test | Memory Usage |
|--------|----------------|------------------|--------------|
| 10 | 1,024 | 0.1 - 0.5s | < 100 MB |
| 15 | 32,768 | 0.5 - 2s | < 500 MB |
| 20 | 1,048,576 | 2 - 10s | 2-4 GB |
| 25 | 33,554,432 | 10 - 60s | 8-16 GB |

**Full suite runtime:** Approximately 5-15 minutes (depending on configuration)

## Integration with CI/CD

The benchmark suite can be integrated into GitHub Actions:

```yaml
# .github/workflows/compression-benchmarks.yml
name: Compression Benchmarks
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2am
  workflow_dispatch:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install qiskit qiskit-aer
      - name: Run benchmarks
        run: python benchmarks/compression/run_suite.py
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: compression-benchmarks
          path: artifacts/compression/
```

## Validation Framework Integration

Update `tools/verify/quasim_verify/checks/tech_compression.py` to use benchmark artifacts:

```python
# Process directory of .npz files
artifact_dir = Path("artifacts/compression")
results = []

for npz_file in artifact_dir.glob("*.npz"):
    data = np.load(npz_file, allow_pickle=True)
    metadata = data["metadata"][0]
    results.append(metadata["compression_ratio"])

# Compute statistics
median_ratio = np.median(results)
check_passed = median_ratio >= min_threshold
```

## References

- **AHTC Implementation**: `quasim/holo/anti_tensor.py`
- **Technical Dossier**: `legal/appendices/ahtc_technical_dossier.md`
- **Algorithm Documentation**: `docs/holo/ahtc.md`
- **Patent Outline**: `legal/ahtc_patent_outline.md`

## License

Copyright (c) 2025 QRATUM Team. Licensed under Apache 2.0.

## Support

For issues or questions:
1. Check this README and troubleshooting section
2. Review existing test artifacts and reports
3. Open an issue on GitHub with:
   - Configuration used
   - Error messages
   - System specifications
   - `compression_summary.json` if available
