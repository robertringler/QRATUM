# QuASIM Verification Tool Examples

This directory contains example configuration files for the QuASIM verification tool.

## Files

- **verify.config.yaml**: Main configuration file with input/output paths and policy settings
- **claims.yaml**: Registry of claims with evidence mapping

## Quick Start

1. Install the verification tool:
```bash
pip install -e tools/verify
```

2. Create test data (see main README for data file formats)

3. Run verification:
```bash
quasim-verify --config tools/verify/examples/verify.config.yaml --verbose
```

## Customizing Configuration

### Adjusting Tolerances

Edit `verify.config.yaml` to adjust tolerance thresholds:

```yaml
policy:
  tolerances:
    rmse_max: 0.02                  # Maximum RMSE for telemetry
    rl_convergence_min: 0.993       # Minimum RL convergence
    compression_min_ratio: 18.0     # Minimum compression ratio
    benchmark_speedup_min: 10.0     # Minimum benchmark speedup
    valuation_p50_min: 5.0e8        # Minimum P50 valuation
    valuation_p50_max: 1.5e9        # Maximum P50 valuation
    cmmc_practices_min: 110         # Minimum CMMC practices
```

### Adding Banned Phrases

Add phrases that should be flagged if found in documentation:

```yaml
policy:
  ban_phrases_unless_evidence:
    - "53B"
    - "3.1 million logical qubits"
    - "your custom phrase"
```

### Disabling Checks

Remove checks from the list to skip them:

```yaml
checks:
  - id: TECH-001  # Keep this check
  # - id: TECH-002  # Skip this check (commented out)
  - id: TECH-003  # Keep this check
```

## Example Output

After running verification, you'll find:

- `artifacts/verify/report.json` - Complete results in JSON format
- `artifacts/verify/report.sarif` - SARIF format for CI/CD integration
- `artifacts/verify/audit.sha256chain` - Cryptographic audit trail

### Example Report Structure

```json
{
  "version": "1",
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
    "passed": 10,
    "failed": 0,
    "errors": 0,
    "warnings": 0,
    "duration_s": 5.23
  }
}
```
