# QuASIM Hardware Reconfiguration Profiles

This module provides hardware reconfiguration profiles for optimizing hardware settings based on workload characteristics.

## Features

- **Device-agnostic profiles**: Support for GPUs, CPUs, and cryogenic systems
- **Policy enforcement**: Automatic validation and limit application
- **Built-in profiles**: 4 pre-configured profiles for common scenarios
- **Custom profiles**: Create your own optimization profiles
- **Dynamic planning**: Generate device-specific plans with constraint merging

## Built-in Profiles

### 1. `low-latency`

Maximizes clocks and minimizes latency for interactive workloads.

- High GPU clocks (2100 MHz SM, 2619 MHz memory)
- High power limit (350W)
- Performance CPU governor
- Target: Interactive and real-time applications

### 2. `energy-cap`

Minimizes power consumption for sustained workloads.

- Reduced GPU clocks (1410 MHz SM, 1593 MHz memory)
- Lower power limit (200W)
- Powersave CPU governor
- Target: Long-running batch processing

### 3. `coherence`

Optimizes for quantum coherence and low noise.

- Minimal GPU power (150W)
- Conservative clocks (1200 MHz)
- ECC enabled
- Cryogenic system control (15mK, 0mV bias)
- Target: Quantum computing applications

### 4. `balanced`

Balanced performance and efficiency.

- Moderate GPU settings (250W, 1800 MHz)
- Schedutil CPU governor
- Target: General-purpose workloads

## Quick Start

### Using Built-in Profiles

```python
from quasim.hw import (
    ReconfigurationProfile,
    TopologyDiscovery,
    PolicyEngine,
    DeviceLimits,
)

# Setup topology
topology = TopologyDiscovery()
topology.add_device("gpu0", "gpu", "NVIDIA A100")
topology.add_device("cpu0", "cpu", "Intel Xeon")

# Setup policy
policy = PolicyEngine()
policy.set_device_limits("gpu0", DeviceLimits(
    device_id="gpu0",
    power_watts_max=400,
    clock_mhz_range=(1000, 2500),
    fan_percent_range=(0, 100)
))

# Load and apply profile
profile = ReconfigurationProfile.load("low-latency")
plan = profile.plan(topology, policy)

# Use the plan with your actuator
print(f"Plan for {len(plan['devices'])} devices")
for device_id, setpoints in plan['devices'].items():
    print(f"{device_id}: {setpoints}")
```

### Creating Custom Profiles

```python
from quasim.hw import create_custom_profile, ReconfigurationProfile

# Create custom profile
custom = create_custom_profile(
    name="my-optimization",
    description="Custom workload optimization",
    setpoints={
        "gpu": {
            "power_limit_w": 275,
            "sm_clock_mhz": 1950,
            "mem_clock_mhz": 2200,
            "fan_percent": 65,
        },
        "cpu": {
            "governor": "ondemand",
        },
    },
    constraints={
        "temp_c_max": 75,
        "power_watts_max": 300,
    }
)

# Use custom profile
profile = ReconfigurationProfile(custom)
plan = profile.plan(topology, policy)
```

### Filtering Devices

```python
# Generate plan for specific devices only
plan = profile.plan(
    topology,
    policy,
    devices=["gpu0"],  # Only configure gpu0
)
```

### Adding Constraints

```python
# Add runtime constraints to profile
plan = profile.plan(
    topology,
    policy,
    constraints={
        "power_watts_max": 250,  # Override or add constraints
        "temp_c_max": 70,
    }
)
```

## API Reference

### Profile

Dataclass representing a hardware reconfiguration profile.

**Attributes:**

- `name` (str): Profile name
- `description` (str): Profile description
- `setpoints` (Dict[str, Dict[str, Any]]): Device type to parameter mappings
- `constraints` (Optional[Dict[str, Any]]): Global constraints

### ReconfigurationProfile

Main class for profile management and planning.

**Methods:**

- `load(profile_name: str)`: Load built-in profile by name
- `list_profiles()`: List available profile names
- `plan(topology, policy, devices=None, constraints=None)`: Generate reconfiguration plan

### TopologyDiscovery

Hardware topology management.

**Methods:**

- `add_device(device_id, device_type, name=None, capabilities=None)`: Add device
- `get_device(device_id)`: Get device information
- `get_devices_by_type(device_type)`: Get all devices of a type

### PolicyEngine

Policy enforcement and validation.

**Methods:**

- `set_device_limits(device_id, limits)`: Set device limits
- `get_device_limits(device_id)`: Get device limits
- `validate_operation(device_id, operation, setpoints, enable_actuation)`: Validate operation

### DeviceLimits

Dataclass for device operational limits.

**Attributes:**

- `device_id` (str): Device identifier
- `power_watts_max` (Optional[float]): Maximum power limit
- `clock_mhz_range` (Optional[Tuple[int, int]]): Clock frequency range (min, max)
- `fan_percent_range` (Optional[Tuple[int, int]]): Fan speed range (min, max)
- `temp_c_max` (Optional[float]): Maximum temperature

### PolicyViolation

Exception raised when operations violate policy constraints.

## Integration with Actuators

The generated plans are designed to be consumed by hardware actuators:

```python
plan = profile.plan(topology, policy)

# Plan structure:
{
    "profile": "low-latency",
    "description": "Maximize clocks...",
    "plan_id": "low-latency-12345",
    "devices": {
        "gpu0": {
            "sm_clock_mhz": 2100,
            "mem_clock_mhz": 2619,
            "power_limit_w": 350,
            "fan_percent": 80,
        },
        "cpu0": {
            "governor": "performance",
            "min_freq_mhz": 3000,
        }
    },
    "constraints": {
        "temp_c_max": 85,
    }
}
```

## Error Handling

```python
from quasim.hw import PolicyViolation

try:
    plan = profile.plan(topology, policy)
except PolicyViolation as e:
    print(f"Policy violation: {e}")
except ValueError as e:
    print(f"Invalid profile: {e}")
```

## Testing

Run the test suite:

```bash
pytest tests/software/test_hw_profiles.py -v
```

## License

Part of the QuASIM project. See main repository LICENSE file.
