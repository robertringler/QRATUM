# QuASIM Hardware Backends

This module provides hardware backends for GPU and accelerator management in QuASIM.

## NVIDIA NVML Backend

The NVML (NVIDIA Management Library) backend provides safe GPU reconfiguration capabilities through the `pynvml` library.

### Features

- **Device Management**
  - List available NVIDIA GPUs
  - Query device state (power, temperature, clocks, fan, ECC)

- **Safe Reconfiguration**
  - Power limit adjustment
  - Clock frequency control (SM and memory)
  - Fan speed control
  - ECC enable/disable
  - Reset to default configuration

- **Safety Features**
  - Dry-run mode for testing
  - Proper error handling
  - No firmware writes
  - Safe operations only

### Installation

To use the NVML backend, install the `pynvml` package:

```bash
pip install pynvml
```

### Usage

```python
from quasim.hardware import NVMLBackend, NVML_AVAILABLE

# Check availability
if not NVML_AVAILABLE:
    print("pynvml not available")
    exit(1)

# Initialize backend
backend = NVMLBackend()

if not backend.initialized:
    print("Failed to initialize")
    exit(1)

# List devices
devices = backend.list_devices()
for device in devices:
    print(f"{device['id']}: {device['name']}")

# Get state
device_id = "GPU0"
state = backend.get_state(device_id)
print(f"Power: {state['power_mw']} mW")
print(f"Temperature: {state['temp_c']} °C")

# Apply setpoint (dry-run)
result = backend.apply_setpoint(
    device_id, 
    "power_limit_w", 
    250, 
    dry_run=True
)
print(f"Dry-run result: {result}")

# Apply setpoint (actual)
result = backend.apply_setpoint(
    device_id,
    "power_limit_w",
    250,
    dry_run=False
)
if result["success"]:
    print("Power limit set successfully")

# Reset to defaults
result = backend.reset_to_defaults(device_id, dry_run=False)
```

### Supported Parameters

The `apply_setpoint` method supports the following parameters:

- `power_limit_w`: Power cap in watts
- `sm_clock_mhz`: SM (streaming multiprocessor) clock frequency in MHz
- `mem_clock_mhz`: Memory clock frequency in MHz
- `fan_percent`: Fan speed percentage (0-100)
- `ecc_enabled`: Enable (True) or disable (False) ECC memory

### Example

See `examples/nvml_backend_example.py` for a complete example.

### Testing

Run the test suite:

```bash
pytest tests/hardware/test_nvml_backend.py -v
```

### Requirements

- Python 3.8+
- `pynvml` package (optional, backend disabled if not available)
- NVIDIA GPU with driver installed

### Limitations

- Requires NVIDIA GPU with driver installed
- Some operations may require elevated privileges
- Not all GPUs support all features (e.g., ECC on consumer cards)
- MIG (Multi-Instance GPU) support not yet implemented

### Safety

This backend only performs safe, reversible operations:

- No firmware modifications
- No permanent changes
- All settings can be reset to defaults
- Dry-run mode available for testing

### Error Handling

The backend gracefully handles:

- Missing pynvml package (NVML_AVAILABLE flag)
- Initialization failures
- Device access errors
- Partial state retrieval failures
- Invalid parameters

All errors are logged and returned in result dictionaries.
