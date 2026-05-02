#!/usr/bin/env python3
"""

Example usage of the NVIDIA NVML backend.

This demonstrates how to use the NVMLBackend for GPU management.
"""

from quasim.hardware import NVML_AVAILABLE, NVMLBackend


def main():
    """Demonstrate NVML backend usage."""

    print("NVIDIA NVML Backend Example")
    print("=" * 50)

    # Check if NVML is available
    if not NVML_AVAILABLE:
        print("WARNING: pynvml is not available - backend disabled")
        print("Install with: pip install pynvml")
        return

    # Initialize backend
    backend = NVMLBackend()

    if not backend.initialized:
        print("❌ Failed to initialize NVML backend")
        return

    print("✓ NVML backend initialized successfully\n")

    # List available GPUs
    print("Available GPUs:")
    devices = backend.list_devices()
    for device in devices:
        print(f"  - {device['id']}: {device['name']} (Serial: {device['serial']})")
    print()

    if not devices:
        print("No NVIDIA GPUs found")
        return

    # Get state of first GPU
    device_id = devices[0]["id"]
    print(f"State of {device_id}:")
    state = backend.get_state(device_id)
    if state:
        print(f"  Power: {state.get('power_mw', 'N/A')} mW")
        print(f"  Power Limit: {state.get('power_limit_mw', 'N/A')} mW")
        print(f"  Temperature: {state.get('temp_c', 'N/A')} °C")
        print(f"  SM Clock: {state.get('sm_clock_mhz', 'N/A')} MHz")
        print(f"  Memory Clock: {state.get('mem_clock_mhz', 'N/A')} MHz")
        print(f"  Fan Speed: {state.get('fan_percent', 'N/A')}%")
        print(f"  ECC Enabled: {state.get('ecc_enabled', 'N/A')}")
    print()

    # Demonstrate dry-run setpoint
    print("Testing dry-run setpoint:")
    result = backend.apply_setpoint(device_id, "power_limit_w", 250, dry_run=True)
    if result["success"]:
        print(f"  ✓ Dry-run successful: {result}")
    else:
        print(f"  ❌ Dry-run failed: {result}")
    print()

    # Demonstrate dry-run reset
    print("Testing dry-run reset:")
    result = backend.reset_to_defaults(device_id, dry_run=True)
    if result["success"]:
        print(f"  ✓ Dry-run reset successful: {result}")
    else:
        print(f"  ❌ Dry-run reset failed: {result}")
    print()

    print("=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
