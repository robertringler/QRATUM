# QuASIM TERC-OBS CLI - TERC Observable Emission

The `quasim-terc-obs` command-line interface provides tools for extracting and emitting observables from REVULTRA/QGH algorithms for TERC (Testing, Evaluation, Research, and Compliance) validation tiers.

## Installation

Install the QuASIM package to get access to the CLI:

```bash
pip install -e .
```

## Commands

### 1. Emit Observables

Extract and emit TERC observables from QuASIM state or text input.

```bash
quasim-terc-obs emit --text "ATTACKATDAWN" --out observables.json
quasim-terc-obs emit --state-file state.json --out obs.json --observable beta_metrics
```

**Options:**
- `--state-file` - Path to QuASIM state file (either this or `--text` required)
- `--text` - Text/ciphertext input (either this or `--state-file` required)
- `--out` - Output JSON file (required)
- `--observable` - Specific observable to compute (optional, computes all if not specified)

**Available observables:**
- `beta_metrics` - Beta metrics from cipher analysis
- `ioc_periods` - Index of Coincidence period candidates
- `emergent_complexity` - Emergent complexity score

**Example output:**
```
Computing REVULTRA observables...
Observables emitted to: observables.json
{
  "format_version": "1.0",
  "source": "quasim.terc_bridge",
  "timestamp": "2025-12-14T11:52:00Z",
  "observables": {
    "beta_metrics": {
      "beta_0": 1.234,
      "beta_1": 0.567
    },
    "ioc_periods": [5, 10, 15],
    "emergent_complexity": {
      "score": 3.45,
      "entropy": 3.84
    }
  }
}
```

### 2. List Observables

List all registered TERC observables.

```bash
quasim-terc-obs list
```

**Example output:**
```
Registered TERC Observables:
========================================
  - beta_metrics
  - ioc_periods
  - emergent_complexity
  - qgh_consensus_status
  - holographic_entropy
  - spectral_peaks

Total: 6 observables
```

### 3. Compute Consensus

Compute consensus status observable for QGH.

```bash
quasim-terc-obs consensus --num-nodes 10 --state-dim 5 --out consensus.json
```

**Options:**
- `--num-nodes` - Number of nodes (default: 5)
- `--state-dim` - State dimension (default: 3)
- `--out` - Output JSON file (optional)

**Example output:**
```
Computing consensus for 10 nodes, dimension 5...

Summary: Converged=True, Stability=0.985

Results written to: consensus.json
```

**Output format:**
```json
{
  "format_version": "1.0",
  "source": "quasim.terc_bridge",
  "observables": {
    "consensus_status": {
      "converged": true,
      "stability": 0.985,
      "iterations": 15,
      "final_variance": 0.0023
    }
  }
}
```

### 4. Validate Observable File

Validate observable JSON format for TERC compliance.

```bash
quasim-terc-obs validate --obs-file observables.json
```

**Options:**
- `--obs-file` - Observable JSON file to validate (required)

**Example output:**
```
Validating observable file...
✓ Observable file is valid
Format version: 1.0
Source: quasim.terc_bridge
Observable count: 3

Observables present:
  - beta_metrics
  - ioc_periods
  - emergent_complexity
```

**Validation checks:**
- Required fields present (`observables`, `format_version`, `source`)
- Valid JSON structure
- Observable format compliance

**Error example:**
```
❌ Missing 'observables' field
```

## Workflow Example

Complete TERC observable workflow:

```bash
# 1. Emit all observables from text input
quasim-terc-obs emit --text "ENCRYPTEDMESSAGE" --out step1_all.json

# 2. Emit specific observable
quasim-terc-obs emit --text "ENCRYPTEDMESSAGE" --out step2_beta.json --observable beta_metrics

# 3. Compute consensus observable
quasim-terc-obs consensus --num-nodes 10 --state-dim 5 --out step3_consensus.json

# 4. Validate all observable files
quasim-terc-obs validate --obs-file step1_all.json
quasim-terc-obs validate --obs-file step2_beta.json
quasim-terc-obs validate --obs-file step3_consensus.json

# 5. List all available observables
quasim-terc-obs list
```

## Observable Descriptions

### Beta Metrics

Extracts beta metrics from cipher analysis.

**Computed from:**
- Frequency analysis
- Statistical properties
- Distribution characteristics

**Use cases:**
- Cipher strength assessment
- Randomness testing
- Statistical validation

### IoC Periods

Identifies period candidates from Index of Coincidence analysis.

**Computed from:**
- IoC tensor analysis
- Peak detection
- Period inference

**Use cases:**
- Key length detection
- Polyalphabetic cipher analysis
- Periodicity detection

### Emergent Complexity

Quantifies the emergent complexity of ciphertext.

**Computed from:**
- Shannon entropy
- Pattern complexity
- Information measures

**Use cases:**
- Encryption strength
- Complexity classification
- Security assessment

### QGH Consensus Status

Computes distributed consensus status across quantum nodes.

**Computed from:**
- Node state analysis
- Convergence detection
- Stability metrics

**Use cases:**
- Distributed validation
- Consensus verification
- Network health monitoring

## TERC Validation Tiers

TERC observables support multiple validation tiers:

### Tier 1: Basic Validation
- Format compliance
- Required fields present
- Basic integrity checks

### Tier 2: Statistical Validation
- Metric ranges
- Distribution properties
- Statistical significance

### Tier 3: Cross-Validation
- Multiple observable consistency
- Temporal consistency
- Comparative analysis

### Tier 4: Certification
- Compliance documentation
- Reproducibility verification
- Audit trail validation

## Observable Format Specification

Standard TERC observable JSON format:

```json
{
  "format_version": "1.0",
  "source": "quasim.terc_bridge",
  "timestamp": "ISO-8601 timestamp",
  "metadata": {
    "input_type": "text|state|simulation",
    "input_length": 123,
    "seed": 42
  },
  "observables": {
    "observable_name": {
      "metric1": value1,
      "metric2": value2
    }
  }
}
```

**Required fields:**
- `format_version` - Observable format version
- `source` - Source system identifier
- `observables` - Dictionary of observable data

**Optional fields:**
- `timestamp` - Emission timestamp
- `metadata` - Additional metadata

## Integration with QuASIM

TERC observables integrate with QuASIM validation:

- **REVULTRA Integration**: Extract observables from cryptanalysis
- **QGH Integration**: Consensus and distributed observables
- **State Bridge**: Convert QuASIM states to observables
- **Validation Pipeline**: Automated TERC tier validation

## Performance Characteristics

- **Emission Time**: < 100ms for typical inputs
- **File Size**: 1-10 KB for standard observables
- **Validation Time**: < 10ms
- **Memory**: O(n) for input size n

## Advanced Usage

### Batch Observable Emission

Process multiple inputs:

```bash
for file in inputs/*.txt; do
  quasim-terc-obs emit --text "$(cat $file)" --out "observables/$(basename $file .txt).json"
done
```

### Pipeline Integration

Integrate with analysis pipelines:

```bash
# Analyze with REVULTRA
quasim-revultra analyze --file cipher.txt --export analysis.json

# Extract observables
python extract_from_analysis.py analysis.json | quasim-terc-obs emit --text - --out obs.json

# Validate
quasim-terc-obs validate --obs-file obs.json
```

### Custom Observable Processing

```python
# Load observable file
import json

with open('observables.json') as f:
    obs_data = json.load(f)

# Process observables
beta_0 = obs_data['observables']['beta_metrics']['beta_0']
complexity = obs_data['observables']['emergent_complexity']['score']

# Custom validation logic
if complexity > 3.0 and beta_0 > 1.0:
    print("High-complexity cipher detected")
```

## Troubleshooting

### Must specify state-file or text
```bash
Error: Must specify either --state-file or --text
```
**Solution:** Provide input via `--state-file` or `--text`.

### Unknown observable
```bash
Unknown observable: custom_metric
Available: ['beta_metrics', 'ioc_periods', 'emergent_complexity']
```
**Solution:** Use one of the available observables listed, or compute all by omitting `--observable`.

### Invalid JSON format
```bash
❌ Invalid JSON: Expecting property name enclosed in double quotes
```
**Solution:** Fix JSON syntax in the input file.

### Missing required field
```bash
❌ Missing 'observables' field
```
**Solution:** Ensure the observable file has the required structure. Re-emit observables if needed.

## Support

For issues or questions, refer to:
- Main README: [../README.md](../README.md)
- TERC Bridge Documentation: [terc_bridge.md](terc_bridge.md)
