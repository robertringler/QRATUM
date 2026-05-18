# QuASIM Validation Reference Data

This directory contains reference datasets for validation testing.

## Structure

Each module/kernel can have a corresponding reference file:

- `<module_path_with_underscores>.json` - Reference data for validation

## Reference Data Format

```json
{
  "module": "quasim/module_name.py",
  "fidelity": 0.999,
  "deterministic": true,
  "test_vectors": [
    {
      "input": [...],
      "expected_output": [...],
      "tolerance": 1e-10
    }
  ],
  "compliance": {
    "DO-178C": true,
    "CMMC": true,
    "ISO-26262": true
  }
}
```

## Adding Reference Data

To add reference data for a module:

1. Run the module with known inputs
2. Capture the outputs
3. Create a JSON file with the format above
4. Place it in this directory with the correct naming convention

## Validation Process

The validation script compares module outputs against these reference datasets to ensure:

- Deterministic reproducibility
- Fidelity thresholds (≥ 0.995)
- Compliance with standards
