# QuASIM Verification Tool

Deterministic verification tool that validates technical, compliance, and economic claims made in the QuASIM × QuNimbus Executive Brief and related documentation.

## Overview

This tool:

- **Extracts and normalizes claims** (technical, compliance, economic) from your brief and repo
- **Runs deterministic checks** (benchmarks, RMSE, compression ratio, RL convergence, Φ_QEVF math, Monte Carlo valuation)
- **Verifies evidence presence** (DER/PSAC placeholders, CMMC mappings, SOC2/ISO cert IDs)
- **Lints risky language** (e.g., "53B now", "3.1M logical qubits") unless matching evidence flags exist
- **Emits audit-grade artifacts**: JSON report, SARIF lint, signed hash chain

## Installation

From the repository root:

```bash
pip install -e tools/verify
```

Or install with dependencies:

```bash
cd tools/verify
pip install -e .
```

## Quick Start

### 1. Create Configuration File

Copy the example configuration:

```bash
cp tools/verify/examples/verify.config.yaml my-verify.yaml
```

Edit `my-verify.yaml` to point to your actual data files.

### 2. Run Verification

```bash
quasim-verify --config my-verify.yaml --verbose
```

Or run as a module:

```bash
python -m quasim_verify.cli --config my-verify.yaml --verbose
```

### 3. Check Results

The tool generates three output files:

- **JSON Report**: `artifacts/verify/report.json` - Complete verification results
- **SARIF Report**: `artifacts/verify/report.sarif` - Static analysis format for CI tools
- **Audit Chain**: `artifacts/verify/audit.sha256chain` - SHA256 hash chain for evidence files

## Configuration

### Minimal Configuration

```yaml
version: 1

inputs:
  brief_paths:
    - QuASIM_Executive_Brief.md
  telemetry:
    spacex_csv: data/telemetry/f9_public.csv
    nasa_csv: data/telemetry/orion_public.csv
  artifacts:
    benchmarks_npz_dir: artifacts/benchmarks/
    rl_convergence_json: artifacts/metrics/rl_convergence.json
    compression_npz: artifacts/compression/sample_mps_mera.npz
  compliance:
    psac_id: docs/compliance/DO178C/PSAC.pdf
    sas_id: docs/compliance/DO178C/SAS.pdf
    der_letter: docs/compliance/DO178C/DER_letter.pdf
    cmmc_map: docs/compliance/CMMC_L2_110.yaml
  economics:
    phi_inputs_yaml: artifacts/econ/phi_inputs.yaml
    montecarlo_params_yaml: artifacts/econ/mc_params.yaml

checks:
  - id: TECH-001  # Benchmark validation
  - id: TECH-002  # Telemetry RMSE
  - id: TECH-003  # RL convergence
  - id: TECH-004  # Compression ratio
  - id: ECON-201  # Φ_QEVF validation
  - id: ECON-202  # Monte Carlo valuation
  - id: COMP-101  # Compliance artifacts
  - id: COMP-102  # CMMC mapping
  - id: GOV-301   # Language lint
  - id: DOC-401   # Audit chain

policy:
  tolerances:
    rmse_max: 0.02
    rl_convergence_min: 0.993
    compression_min_ratio: 18.0
    benchmark_speedup_min: 10.0
    valuation_p50_min: 5.0e8
    valuation_p50_max: 1.5e9
    cmmc_practices_min: 110
  ban_phrases_unless_evidence:
    - "53B"
    - "3.1 million logical qubits"
  require_der_for_level_a: true

outputs:
  report_json: artifacts/verify/report.json
  report_sarif: artifacts/verify/report.sarif
  audit_chain_file: artifacts/verify/audit.sha256chain
  exit_on_fail: true
```

## Available Checks

### Technical Checks

- **TECH-001**: Benchmark validation - Validates QuASIM benchmark speedups against baseline
- **TECH-002**: Telemetry RMSE - Verifies <2% RMSE vs SpaceX/NASA public telemetry
- **TECH-003**: RL convergence - Validates RL convergence ≥0.993
- **TECH-004**: Compression ratio - Validates 18.7× MERA compression ratio

### Economic Checks

- **ECON-201**: Φ_QEVF validation - Validates quantum economic value function parameters
- **ECON-202**: Monte Carlo valuation - Runs Monte Carlo to verify P50 enterprise value bounds

### Compliance Checks

- **COMP-101**: Compliance artifacts - Verifies presence of DO-178C PSAC/SAS/DER documents
- **COMP-102**: CMMC mapping - Validates CMMC Level 2 practice mapping completeness (≥110 practices)

### Governance Checks

- **GOV-301**: Language lint - Flags banned phrases without evidence support
- **DOC-401**: Audit chain - Generates SHA256 hash chain for all evidence files

## Data File Formats

### Telemetry CSV

Expected columns:

- `time_s`: Time in seconds
- `altitude_km`: Predicted altitude
- `altitude_km_ref`: Reference altitude

### Benchmark NPZ

Required fields:

- `speedup`: Computed speedup ratio, OR
- `baseline_time` and `quasim_time`: Timing data for speedup calculation

### RL Convergence JSON

Required field:

- `final_convergence`: Float value of final convergence metric

### Compression NPZ

Required fields:

- `raw_flops`: Raw FLOP count
- `compressed_flops`: Compressed FLOP count

### Φ_QEVF Inputs YAML

Required fields:

```yaml
base_value_per_eph: 0.0004
eta_ent: 0.93
eta_baseline: 1.0
coherence_penalty: 0.95
runtime_factor: 1.0
market_multiplier: 1.0
```

### Monte Carlo Parameters YAML

Required structure:

```yaml
trials: 5000
seed: 42
scenarios:
  - prob: 0.2
    value: 500000000
  - prob: 0.5
    value: 750000000
  - prob: 0.3
    value: 1000000000
```

### CMMC Mapping YAML

Required structure:

```yaml
practices:
  - id: AC.L2-3.1.1
    description: "Limit system access..."
    implementation: "..."
  # ... at least 110 practices for CMMC L2
```

## Interpreting Results

### JSON Report

The JSON report contains:

```json
{
  "version": "0.1.0",
  "started_at": "2025-01-15T10:00:00Z",
  "finished_at": "2025-01-15T10:00:05Z",
  "results": [
    {
      "id": "TECH-001",
      "passed": true,
      "severity": "error",
      "details": {...},
      "evidence_paths": [...]
    }
  ],
  "summary": {
    "total": 10,
    "passed": 9,
    "failed": 1,
    "errors": 1,
    "warnings": 0,
    "duration_s": 5.23
  }
}
```

### Exit Codes

- **0**: All checks passed or only warnings
- **1**: One or more error-level checks failed (if `exit_on_fail: true`)

## Adding New Checks

### 1. Create Check Module

Create a new file in `quasim_verify/checks/`:

```python
"""My custom check (CUSTOM-001)."""

from typing import Any, Dict
from ..models import CheckResult

def run(cfg: Dict[str, Any]) -> CheckResult:
    """Run custom check.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        CheckResult with pass/fail status
    """
    try:
        # Your check logic here
        passed = True
        details = {"result": "OK"}
        
        return CheckResult(
            id="CUSTOM-001",
            passed=passed,
            details=details,
            evidence_paths=[]
        )
    except Exception as e:
        return CheckResult(
            id="CUSTOM-001",
            passed=False,
            details={"error": str(e)}
        )
```

### 2. Register Check

Add to `quasim_verify/registry.py`:

```python
from .checks import my_custom_check

REG: Dict[str, Callable] = {
    # ... existing checks ...
    "CUSTOM-001": my_custom_check.run,
}
```

### 3. Add to Configuration

Add to your `verify.config.yaml`:

```yaml
checks:
  - id: CUSTOM-001
```

## CI Integration

### GitHub Actions

Create `.github/workflows/verify.yml`:

```yaml
name: Verify Claims & Compliance
on:
  pull_request:
  push:
    branches: [ main ]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e tools/verify
      - run: quasim-verify --config tools/verify/examples/verify.config.yaml
```

### Upload SARIF Results

To integrate with GitHub Code Scanning:

```yaml
- name: Upload SARIF results
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: artifacts/verify/report.sarif
```

## Testing

Run the test suite:

```bash
pytest tests/verify/ -v
```

Run with coverage:

```bash
pytest tests/verify/ --cov=quasim_verify --cov-report=html
```

## Troubleshooting

### Missing Dependencies

```bash
pip install numpy pandas scipy pydantic pyyaml tabulate jsonschema
```

### File Not Found Errors

Check that:

1. All paths in `verify.config.yaml` are correct
2. Paths are relative to repository root or absolute
3. Required data files exist

### Failed Checks

Use verbose mode to see detailed output:

```bash
quasim-verify --config my-verify.yaml --verbose
```

Check the JSON report for detailed error messages:

```bash
cat artifacts/verify/report.json | python -m json.tool
```

## License

Apache 2.0 - See repository LICENSE file.

## Support

For issues or questions:

1. Check this README
2. Review example configurations in `examples/`
3. Open an issue on GitHub
