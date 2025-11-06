# QuASIM HCAL CLI - Quick Reference

## Installation

```bash
pip install -e .
```

## Common Usage Examples

### Basic Discovery and Configuration

```bash
# Discover hardware
quasim-hcal discover --json

# Plan reconfiguration
quasim-hcal plan --profile low-latency --devices GPU0,GPU1

# Validate plan (dry-run)
quasim-hcal apply --plan plan.json

# Apply configuration
quasim-hcal apply --plan plan.json --enable-actuation --require-approval token
```

### Calibration

```bash
# Standard calibration
quasim-hcal calibrate --device GPU0 --routine power_sweep --max-iters 20

# Available routines
quasim-hcal calibrate --device GPU0 --routine power_sweep
quasim-hcal calibrate --device GPU0 --routine thermal_test
quasim-hcal calibrate --device GPU0 --routine memory_bandwidth
```

### Emergency Controls

```bash
# Stop all devices
quasim-hcal stop --all

# Stop specific device
quasim-hcal stop --device GPU0
```

## Configuration Profiles

- **low-latency** - Maximum performance, optimized for minimal latency
- **high-throughput** - Balanced mode, optimized for throughput
- **power-efficient** - Low power mode, optimized for efficiency

## See Also

- Full documentation: [docs/CLI_HCAL.md](docs/CLI_HCAL.md)
- Test examples: [tests/test_cli_hcal.py](tests/test_cli_hcal.py)
