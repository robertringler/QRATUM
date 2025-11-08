# HCAL - Hardware Control Abstraction Layer

## Overview

HCAL (Hardware Control Abstraction Layer) provides a unified interface for controlling and monitoring physical hardware devices in quantum computing and simulation environments. It abstracts the complexity of hardware interactions and provides policy-based control mechanisms.

## Features

- **Device Discovery**: Automatically discover and enumerate available hardware devices
- **Policy Enforcement**: Define and enforce hardware access policies based on environment
- **Backend Abstraction**: Support multiple compute backends (CPU, CUDA, ROCm)
- **Resource Management**: Track and limit resource usage
- **CLI Tools**: Command-line utilities for device management

## Architecture

### Components

1. **Device Manager** (`quasim.hcal.device`)
   - Device discovery and enumeration
   - Device lifecycle management
   - Status monitoring

2. **Policy Validator** (`quasim.hcal.policy`)
   - Policy configuration loading
   - Backend validation
   - Resource limit enforcement

3. **CLI Interface** (`quasim.hcal.cli`)
   - `quasim-hcal discover`: List available devices
   - `quasim-hcal validate-policy`: Validate policy files

## Installation

```bash
pip install -e .
```

## Usage

### Device Discovery

```python
from quasim.hcal.device import DeviceManager

manager = DeviceManager()
devices = manager.discover()

for device in devices:
    print(f"{device.name} ({device.id}): {device.status}")
```

### Policy Validation

```python
from pathlib import Path
from quasim.hcal.policy import PolicyValidator

validator = PolicyValidator.from_file(Path("configs/hcal/policy.example.yaml"))

# Check if a backend is allowed
if validator.validate_backend("cuda"):
    print("CUDA backend is allowed")

# Check resource limits
if validator.check_limits("max_qubits", 15):
    print("Within qubit limit")
```

### CLI Usage

```bash
# Discover devices
quasim-hcal discover

# Discover devices (JSON output)
quasim-hcal discover --json

# Validate policy file
quasim-hcal validate-policy configs/hcal/policy.example.yaml
# HCAL: Hardware Control & Calibration Layer

**Version:** 0.1.0  
**Status:** Initial Release

## Overview

The Hardware Control & Calibration Layer (HCAL) provides a unified API for controlling and calibrating hardware in QuASIM systems. HCAL focuses on safety, auditability, and policy-driven control.

## Key Features

- **Dry-run by default**: All operations are simulated unless explicitly enabled
- **Policy-driven safety**: Declarative YAML policies enforce limits and restrictions
- **Automatic discovery**: Hardware topology detection for CPUs, GPUs, FPGAs
- **Real-time telemetry**: Sensor data collection with caching
- **Closed-loop calibration**: Automated optimization routines
- **Audit logging**: Tamper-evident cryptographic chain for all operations

## Quick Start

### Installation

```bash
pip install pyyaml click numpy
# Optional: For NVIDIA GPU support
pip install pynvml
```

### Basic Usage

```python
from quasim.hcal import HCAL

# Initialize HCAL (dry-run by default)
hcal = HCAL()

# Discover hardware
topology = hcal.discover_topology()
print(f"Found {len(topology.devices)} devices")

# Read telemetry
reading = hcal.read_telemetry("gpu0")
print(f"GPU0 Power: {reading.metrics['power_watts']} W")

# Apply setpoint (dry-run)
setpoint = {"power_limit_watts": 250.0}
success = hcal.apply_setpoint("gpu0", setpoint)
```

### CLI Usage

```bash
# Discover hardware topology
quasim-hcal discover

# Plan a reconfiguration
quasim-hcal plan --profile low-latency --device gpu0

# Apply a profile (dry-run by default)
quasim-hcal apply --profile balanced --device gpu0

# Actually apply changes (requires policy approval)
quasim-hcal apply --profile balanced --device gpu0 --actuate

# Read telemetry
quasim-hcal telemetry --device gpu0

# Run calibration
quasim-hcal calibrate --device gpu0 --iterations 20
```

## Python API Reference

### HCAL Class

Main interface for hardware control and calibration.

```python
class HCAL:
    def __init__(
        self,
        policy_path: Optional[Path] = None,
        dry_run: bool = True,
        audit_log_path: Optional[Path] = None,
    ):
        """Initialize HCAL with optional policy and audit log."""
```

**Methods:**

- `discover_topology()`: Discover hardware topology
- `apply_setpoint(device_id, setpoint)`: Apply configuration to device
- `read_telemetry(device_id)`: Read sensor data from device
- `calibrate_bias_trim(device_id, max_iterations)`: Run bias trim calibration
- `run_power_sweep(device_id, power_range, steps)`: Run power sweep
- `emergency_stop()`: Halt all operations
- `get_audit_log()`: Retrieve audit log entries
- `verify_audit_chain()`: Verify audit log integrity

### PolicyEngine Class

Policy validation and enforcement.

```python
class PolicyEngine:
    def __init__(self, policy_path: Optional[Path] = None):
        """Load policy from YAML file."""
    
    def check_device_allowed(self, device_id: str) -> bool:
        """Check if device is allowed by policy."""
    
    def check_backend_allowed(self, backend: str) -> bool:
        """Check if backend is allowed."""
    
    def check_limits(self, device_id: str, setpoint: Dict) -> bool:
        """Validate setpoint against limits."""
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
```

### TopologyDiscovery Class

Hardware topology discovery.

```python
class TopologyDiscovery:
    def discover(self) -> Topology:
        """Discover hardware topology."""
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID."""
    
    def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        """Get devices by type."""
```

## Configuration

Policy configuration files are YAML-based and must include:

- `environment`: Environment type (DEV, LAB, PROD)
- `allowed_backends`: List of permitted compute backends
- `limits`: Resource limits (max_qubits, max_circuits, etc.)

Example policy file:

```yaml
environment: DEV
allowed_backends:
  - cpu
  - cuda
limits:
  max_qubits: 20
  max_circuits: 100
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/hcal/ -v

# Run with coverage
pytest tests/hcal/ --cov=quasim.hcal --cov-report=term

# Run only non-hardware tests
pytest tests/hcal/ -m "not requires_hardware"
```

## Development

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use Ruff for linting and formatting

### Linting

```bash
# Check code style
ruff check quasim/hcal/ tests/hcal/

# Format code
ruff format quasim/hcal/ tests/hcal/

# Type checking
mypy quasim/hcal/ --ignore-missing-imports
```

## Safety Considerations

See [Safety-and-Compliance.md](./Safety-and-Compliance.md) for detailed safety guidelines and compliance requirements.

## License

Copyright © 2024 QuASIM Project
### Policy File

Create a policy file (e.g., `policy.yaml`):

```yaml
environment: dev
dry_run_default: true

device_allowlist:
  - "*"

backend_restrictions: []

device_limits:
  gpu0:
    max_power_watts: 300.0
    max_temp_celsius: 85.0

rate_limit:
  commands_per_minute: 30
  window_seconds: 60

approval_gate:
  required: false
  min_approvers: 2
  methods:
    - github_pr
```

Use the policy:

```python
hcal = HCAL(policy_path=Path("policy.yaml"))
```

See `configs/hcal/policy.example.yaml` for a complete example.

## Reconfiguration Profiles

HCAL provides pre-defined optimization profiles:

- **low-latency**: Maximize performance
- **energy-cap**: Minimize power consumption
- **coherence**: Optimize for quantum coherence
- **balanced**: Balance performance and efficiency

```python
from quasim.hcal.loops.reconfig_profiles import ProfileManager

mgr = ProfileManager()
setpoints = mgr.apply_profile("low-latency", "gpu", "gpu0")
hcal.apply_setpoint("gpu0", setpoints)
```

## Calibration Routines

### Bias Trim

Optimize power consumption while maintaining performance:

```python
result = hcal.calibrate_bias_trim("gpu0", max_iterations=20)
print(f"Status: {result.status}")
print(f"Best objective: {result.best_objective}")
print(f"Final setpoint: {result.final_setpoint}")
```

### Power Sweep

Characterize performance across power levels:

```python
results = hcal.run_power_sweep("gpu0", power_range=(150, 300), steps=10)
for power, telemetry in results:
    print(f"Power: {power}W, Temp: {telemetry['temperature_celsius']}°C")
```

## Backends

### NVIDIA NVML Backend

Fully implemented backend for NVIDIA GPUs:

- Power cap control
- Clock frequency management
- Temperature monitoring
- Utilization tracking
- Memory usage monitoring

### Other Backends (Stubs)

The following backends are planned for future implementation:

- AMD ROCm (AMD GPUs)
- Linux sysfs (CPU governors, DVFS)
- Xilinx XRT (FPGA reconfiguration)
- Intel OPAE (FPGA AFU management)
- Intel IPMI (Server chassis control)

## Safety Features

1. **Dry-run by default**: All operations are simulated unless explicitly enabled
2. **No firmware writes**: HCAL never modifies firmware or microcode
3. **Vendor API boundaries**: Strictly uses vendor-provided APIs
4. **Audit logging**: All operations logged with SHA256 chain
5. **Automatic rollback**: Failed operations trigger rollback to baseline
6. **Emergency stop**: Can halt all operations immediately
7. **Rate limiting**: Prevents runaway automation
8. **Policy enforcement**: All operations validated against policy

## Troubleshooting

### "NVML not available"

Install pynvml: `pip install pynvml`

Ensure NVIDIA drivers are installed.

### "Device not allowed by policy"

Check your policy file's `device_allowlist`. Use `"*"` to allow all devices or add specific device IDs.

### "Rate limit exceeded"

Adjust `rate_limit.commands_per_minute` in your policy file, or wait for the rate limit window to reset.

### "No backend available for device"

The device type may not be supported yet. Check that you're using a supported device ID format (e.g., `gpu0` for NVIDIA GPUs).

## Known Limitations

- Only NVIDIA backend is fully implemented
- Test coverage at ~60% (target: 90%)
- No Prometheus metrics export yet
- Limited calibration routines
- No hardware-in-the-loop test infrastructure

See the CHANGELOG for a complete list of limitations.

## Contributing

See CONTRIBUTING.md for guidelines.

## License

See LICENSE file for details.

## See Also

- [Safety & Compliance Guide](Safety-and-Compliance.md)
- [HCAL CHANGELOG](../../CHANGELOG_HCAL.md)
- Policy Examples: `configs/hcal/policy.example.yaml`
