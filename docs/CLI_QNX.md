# QNX CLI - Multi-Backend Quantum Simulation

The `qnx` command-line interface provides a unified substrate for running quantum simulations across multiple backend implementations with deterministic reproducibility and carbon emission tracking.

## Installation

Install the QuASIM package to get access to the CLI:

```bash
pip install -e .
```

## Commands

### simulate

Run a quantum simulation on a specified backend.

```bash
qnx simulate --scenario smoke --timesteps 10 --backend quasim_modern
qnx simulate --scenario aerospace_traj --timesteps 100 --seed 42 --backend quasim_modern
```

**Options:**
- `--scenario` - Scenario identifier (required)
- `--timesteps` - Number of timesteps to simulate (default: 1)
- `--seed` - Optional random seed for deterministic execution
- `--backend` - Backend to execute (default: quasim_modern)

**Available backends:**
- `quasim_modern` - Modern QuASIM implementation (default)
- `quasim_legacy_v1_2_0` - Legacy QuASIM v1.2.0 for compatibility
- `qvr_win` - QVR Windows-based backend

**Example output:**
```json
{
  "backend": "quasim_modern",
  "hash": "a1b2c3d4e5f6",
  "execution_time_ms": 123.45,
  "carbon_emissions_kg": 0.0023
}
```

## Usage Examples

### Basic Simulation

Run a simple smoke test scenario:

```bash
qnx simulate --scenario smoke --timesteps 1
```

**Output:**
```json
{
  "backend": "quasim_modern",
  "hash": "abc123def456",
  "execution_time_ms": 45.23,
  "carbon_emissions_kg": 0.0008
}
```

### Deterministic Simulation

Run with a specific seed for reproducibility:

```bash
qnx simulate --scenario aerospace_traj --timesteps 50 --seed 42
```

**Output:**
```json
{
  "backend": "quasim_modern",
  "hash": "42seed789abc",
  "execution_time_ms": 567.89,
  "carbon_emissions_kg": 0.0045
}
```

### Legacy Backend Comparison

Compare results across backends:

```bash
# Modern backend
qnx simulate --scenario test_scenario --timesteps 10 --seed 100 --backend quasim_modern

# Legacy backend
qnx simulate --scenario test_scenario --timesteps 10 --seed 100 --backend quasim_legacy_v1_2_0
```

**Use case:** Verify backward compatibility and validate migration from legacy systems.

### Long-Running Simulation

Execute extended simulation:

```bash
qnx simulate --scenario complex_system --timesteps 1000 --seed 12345 --backend quasim_modern
```

**Output:**
```json
{
  "backend": "quasim_modern",
  "hash": "longrun12345",
  "execution_time_ms": 15234.67,
  "carbon_emissions_kg": 0.0234
}
```

## Scenarios

### Built-in Scenarios

#### smoke
Quick validation scenario for testing infrastructure.

```bash
qnx simulate --scenario smoke --timesteps 1
```

**Characteristics:**
- Execution time: < 100ms
- Minimal resource usage
- Ideal for CI/CD validation

#### aerospace_traj
Aerospace trajectory optimization scenario.

```bash
qnx simulate --scenario aerospace_traj --timesteps 100 --seed 42
```

**Characteristics:**
- DO-178C compliance
- Deterministic execution
- Telemetry output

#### complex_system
Multi-domain complex system simulation.

```bash
qnx simulate --scenario complex_system --timesteps 500
```

**Characteristics:**
- Long-running
- High computational load
- Comprehensive state tracking

### Custom Scenarios

Custom scenarios can be defined in the QuASIM configuration. Contact support for custom scenario development.

## Backend Details

### quasim_modern (Default)

Modern QuASIM implementation with latest optimizations.

**Features:**
- GPU acceleration support (CUDA/ROCm)
- Optimized tensor operations
- Full compliance validation
- Enhanced observability

**Performance:**
- Typical latency: 50-500ms
- Throughput: 1000+ ops/sec
- Memory: O(n log n)

**Use when:**
- Running production workloads
- Requiring latest features
- Need GPU acceleration

### quasim_legacy_v1_2_0

Legacy QuASIM v1.2.0 backend for backward compatibility.

**Features:**
- Compatibility with v1.2.0
- Frozen feature set
- Deterministic matching with legacy systems

**Performance:**
- Typical latency: 100-1000ms
- CPU-only execution
- Memory: O(n²)

**Use when:**
- Validating legacy results
- Migration verification
- Regression testing

### qvr_win

QVR Windows-based backend for specialized scenarios.

**Features:**
- Windows-specific optimizations
- Integration with Windows tooling
- Specialized use cases

**Performance:**
- Platform-specific
- Windows-optimized

**Use when:**
- Windows-only deployments
- QVR-specific requirements
- Legacy Windows integration

## Output Format

QNX outputs JSON with the following fields:

```json
{
  "backend": "backend_identifier",
  "hash": "simulation_hash",
  "execution_time_ms": 123.45,
  "carbon_emissions_kg": 0.0023
}
```

### Field Descriptions

- **backend**: Backend identifier that executed the simulation
- **hash**: First 12 characters of SHA-256 simulation hash for verification
- **execution_time_ms**: Execution time in milliseconds
- **carbon_emissions_kg**: Estimated carbon emissions in kilograms CO₂

## Workflow Examples

### Development Workflow

```bash
# 1. Quick smoke test
qnx simulate --scenario smoke --timesteps 1

# 2. Development testing
qnx simulate --scenario dev_test --timesteps 10 --seed 42

# 3. Integration testing
qnx simulate --scenario integration --timesteps 50 --seed 123

# 4. Performance testing
qnx simulate --scenario perf_test --timesteps 1000
```

### CI/CD Pipeline

```bash
#!/bin/bash
# QNX CI/CD validation script

# Smoke test
echo "Running smoke test..."
qnx simulate --scenario smoke --timesteps 1 > smoke_result.json

# Verify hash
EXPECTED_HASH="abc123def456"
ACTUAL_HASH=$(cat smoke_result.json | jq -r '.hash')

if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
  echo "Hash mismatch! Expected: $EXPECTED_HASH, Got: $ACTUAL_HASH"
  exit 1
fi

echo "✓ Smoke test passed"

# Run deterministic tests
qnx simulate --scenario test1 --timesteps 10 --seed 42
qnx simulate --scenario test2 --timesteps 10 --seed 43

echo "✓ All tests passed"
```

### Backend Comparison

```bash
#!/bin/bash
# Compare backends for consistency

SCENARIO="comparison_test"
TIMESTEPS=50
SEED=42

echo "Modern backend:"
qnx simulate --scenario $SCENARIO --timesteps $TIMESTEPS --seed $SEED --backend quasim_modern

echo ""
echo "Legacy backend:"
qnx simulate --scenario $SCENARIO --timesteps $TIMESTEPS --seed $SEED --backend quasim_legacy_v1_2_0
```

### Carbon Tracking

```bash
#!/bin/bash
# Track carbon emissions across runs

SCENARIOS=("smoke" "test1" "test2" "test3")
TOTAL_CARBON=0

for scenario in "${SCENARIOS[@]}"; do
  result=$(qnx simulate --scenario $scenario --timesteps 10)
  carbon=$(echo $result | jq -r '.carbon_emissions_kg')
  TOTAL_CARBON=$(echo "$TOTAL_CARBON + $carbon" | bc)
  echo "Scenario $scenario: ${carbon}kg CO₂"
done

echo "Total carbon emissions: ${TOTAL_CARBON}kg CO₂"
```

## Determinism and Reproducibility

### Seed-Based Reproducibility

QNX ensures deterministic execution when a seed is provided:

```bash
# Run 1
qnx simulate --scenario test --timesteps 10 --seed 42 > run1.json

# Run 2 (same seed)
qnx simulate --scenario test --timesteps 10 --seed 42 > run2.json

# Hashes should match
diff <(jq '.hash' run1.json) <(jq '.hash' run2.json)
```

**Guarantee:** Same scenario, timesteps, seed, and backend → identical hash.

### Hash Verification

The simulation hash provides cryptographic verification:

```bash
# Store expected hash
EXPECTED_HASH="abc123def456"

# Run simulation
RESULT=$(qnx simulate --scenario test --timesteps 10 --seed 42)
ACTUAL_HASH=$(echo $RESULT | jq -r '.hash')

# Verify
if [ "$ACTUAL_HASH" = "$EXPECTED_HASH" ]; then
  echo "✓ Simulation verified"
else
  echo "✗ Simulation mismatch"
  exit 1
fi
```

## Integration with QuASIM

QNX serves as a substrate layer for QuASIM:

- **Backend Abstraction**: Unified interface across implementations
- **Scenario Library**: Reusable simulation scenarios
- **Telemetry**: Integrated with QuASIM observability
- **Compliance**: DO-178C validation for all backends

## Performance Characteristics

### Latency

| Backend | Typical Latency | Range |
|---------|----------------|-------|
| quasim_modern | 50-500ms | 10ms - 2s |
| quasim_legacy_v1_2_0 | 100-1000ms | 50ms - 5s |
| qvr_win | Platform-dependent | Varies |

### Throughput

- **Modern**: 1000+ simulations/sec (parallel)
- **Legacy**: 100-500 simulations/sec
- **QVR**: Varies by configuration

### Carbon Emissions

- **Smoke test**: ~0.0008 kg CO₂
- **Standard run**: ~0.002-0.005 kg CO₂
- **Long run**: ~0.01-0.05 kg CO₂

*Based on average grid carbon intensity*

## Troubleshooting

### Invalid scenario
```bash
Error: Unknown scenario 'invalid_name'
```
**Solution:** Use a valid scenario name from the built-in scenarios or contact support for custom scenarios.

### Backend not available
```bash
Error: Backend 'unavailable_backend' not found
```
**Solution:** Use one of the available backends: quasim_modern, quasim_legacy_v1_2_0, or qvr_win.

### Simulation timeout
```bash
Error: Simulation timeout after 300s
```
**Solution:** Reduce `--timesteps` or increase timeout configuration.

### Hash mismatch in CI
```bash
Hash mismatch! Expected: abc123, Got: def456
```
**Solution:** Ensure the same seed, scenario, timesteps, and backend are used. Check for environmental differences.

## Support

For issues or questions, refer to:
- Main README: [../README.md](../README.md)
- QNX Documentation: [qnx_integration/01_qnx_architecture.md](qnx_integration/01_qnx_architecture.md)
- QNX Scenarios: [qnx_integration/02_qnx_scenarios.md](qnx_integration/02_qnx_scenarios.md)
