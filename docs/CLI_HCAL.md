# QuASIM HCAL CLI - Hardware Configuration and Calibration

The `quasim-hcal` command-line interface provides hardware configuration and calibration tools for QuASIM simulation infrastructure.

## Installation

Install the QuASIM package to get access to the CLI:

```bash
pip install -e .
```

## Commands

### 1. Discover Hardware

Scan the system for compatible GPU and accelerator devices.

```bash
# JSON output
quasim-hcal discover --json

# Human-readable output
quasim-hcal discover
```

**Example output (JSON):**
```json
{
  "devices": [
    {
      "id": "GPU0",
      "type": "NVIDIA_A100",
      "memory_gb": 40,
      "compute_capability": "8.0",
      "utilization_percent": 0,
      "status": "available"
    }
  ],
  "discovered_at": "2025-11-06T10:17:50Z",
  "platform": "linux",
  "driver_version": "525.105.17"
}
```

### 2. Plan Reconfiguration

Generate a reconfiguration plan based on a profile and target devices.

```bash
quasim-hcal plan --profile low-latency --devices GPU0,GPU1 [--output plan.json]
```

**Available profiles:**
- `low-latency` - Optimized for minimal latency (max performance mode)
- `high-throughput` - Optimized for maximum throughput (balanced mode)
- `power-efficient` - Optimized for power efficiency (low power mode)

**Example output:**
```
Reconfiguration plan created: plan.json
Profile: low-latency
Target devices: GPU0, GPU1
Estimated downtime: 30s

To apply this plan:
  quasim-hcal apply --plan plan.json
```

### 3. Apply Reconfiguration Plan

Apply a reconfiguration plan with dry-run or actuation mode.

```bash
# Dry-run (default) - validates without making changes
quasim-hcal apply --plan plan.json

# Actuation mode - applies actual changes (requires approval token)
quasim-hcal apply --plan plan.json --enable-actuation --require-approval <token>
```

**Dry-run mode:**
- Validates the plan without making changes
- Shows planned actions and estimated downtime
- Safe to run in production

**Actuation mode:**
- Applies actual hardware configuration changes
- Requires approval token for safety
- Shows progress and completion status

### 4. Calibrate Device

Run calibration routines to optimize device performance.

```bash
quasim-hcal calibrate --device GPU0 --routine power_sweep --max-iters 20

# Use --force to run unknown/custom routines
quasim-hcal calibrate --device GPU0 --routine custom_routine --max-iters 10 --force
```

**Available routines:**
- `power_sweep` - Power consumption sweep across operating points
- `thermal_test` - Thermal stability validation
- `memory_bandwidth` - Memory bandwidth optimization

**Note:** Unknown routines require the `--force` flag for safety.

**Example output:**
```
Starting Hardware Calibration
==================================================
Device: GPU0
Routine: power_sweep
Max iterations: 20

Routine: Power consumption sweep across operating points
  Iteration 1/20: Convergence 25.0%
  ...
  Iteration 15/20: Convergence 95.0%

✓ Calibration completed successfully
Final status: Converged after 15 iterations
```

### 5. Emergency Stop

Immediately halt operations on devices.

```bash
# Stop all devices
quasim-hcal stop --all

# Stop specific device
quasim-hcal stop --device GPU0
```

**Use cases:**
- Emergency situations requiring immediate shutdown
- Maintenance windows
- Safety protocols

## Workflow Example

Complete workflow for hardware reconfiguration:

```bash
# 1. Discover available hardware
quasim-hcal discover --json > hardware_inventory.json

# 2. Plan reconfiguration for low-latency profile
quasim-hcal plan --profile low-latency --devices GPU0,GPU1

# 3. Validate the plan (dry-run)
quasim-hcal apply --plan plan.json

# 4. Apply the configuration
quasim-hcal apply --plan plan.json --enable-actuation --require-approval token

# 5. Calibrate devices after reconfiguration
quasim-hcal calibrate --device GPU0 --routine power_sweep --max-iters 20
quasim-hcal calibrate --device GPU1 --routine power_sweep --max-iters 20

# 6. If issues occur, emergency stop
# quasim-hcal stop --all
```

## Safety Features

1. **Dry-run by default** - Apply command defaults to dry-run mode
2. **Approval tokens** - Actuation requires explicit approval token
3. **Plan validation** - Plans are validated before application
4. **Emergency stop** - Quick shutdown capability for all devices
5. **Status reporting** - Clear progress and completion messages

## Integration with QuASIM

The HCAL CLI is integrated with the QuASIM simulation platform:

- Hardware discovery informs resource allocation
- Configuration profiles optimize for specific workloads
- Calibration ensures optimal performance
- Emergency controls protect infrastructure

## Troubleshooting

### Plan file not found
```bash
Error: Plan file not found: plan.json
```
**Solution:** Create a plan first with `quasim-hcal plan` command.

### Unknown profile
```bash
Error: Unknown profile 'invalid-profile'
Available profiles: low-latency, high-throughput, power-efficient
```
**Solution:** Use one of the available profiles listed.

### Unknown calibration routine
```bash
Error: Unknown routine 'custom_routine'
Available routines: power_sweep, thermal_test, memory_bandwidth
Use --force to run unknown routines
```
**Solution:** Use a known routine or add `--force` flag to run custom routines.

### Actuation without approval
```bash
Error: --require-approval token required for actuation
```
**Solution:** Provide approval token with `--require-approval <token>`.

## Support

For issues or questions, refer to:
- Main README: [../README.md](../README.md)
- Tests: [../tests/test_cli_hcal.py](../tests/test_cli_hcal.py)
