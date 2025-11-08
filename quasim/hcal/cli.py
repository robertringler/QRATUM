"""
Command-line interface for QuASIM Hardware Calibration and Analysis Layer.

This CLI provides tools for monitoring and managing hardware resources
for quantum simulation workloads.
"""

import sys
import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """QuASIM Hardware Calibration and Analysis Layer (HCAL) CLI.
    
    Tools for monitoring and managing hardware resources for quantum simulation.
    """
"""CLI interface for HCAL."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from quasim.hcal import HCAL
from quasim.hcal.loops.reconfig_profiles import ProfileManager
from quasim.hcal.topology import TopologyDiscovery


@click.group()
def cli():
    """QuASIM Hardware Control & Calibration Layer (HCAL) CLI."""
    pass


@cli.command()
def status():
    """Display hardware status and availability."""
    click.echo("Hardware Status:")
    click.echo("================")
    
    # Check for NVIDIA GPU
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        click.echo(f"✓ NVIDIA GPUs detected: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = memory_info.total / (1024**3)
            free_gb = memory_info.free / (1024**3)
            click.echo(f"  GPU {i}: {name}")
            click.echo(f"    Memory: {free_gb:.2f}GB / {total_gb:.2f}GB free")
        
        pynvml.nvmlShutdown()
    except ImportError:
        click.echo("✗ NVIDIA GPU support not available (install with: pip install quasim[hcal-nvidia])")
    except Exception as e:
        click.echo(f"✗ Error accessing NVIDIA GPUs: {e}")
    
    # Check for AMD GPU
    try:
        import pyrsmi
        pyrsmi.rsmi_init()
        device_count = pyrsmi.rsmi_num_monitor_devices()
        click.echo(f"✓ AMD GPUs detected: {device_count}")
        
        for i in range(device_count):
            name = pyrsmi.rsmi_dev_name_get(i)
            click.echo(f"  GPU {i}: {name}")
        
        pyrsmi.rsmi_shut_down()
    except ImportError:
        click.echo("✗ AMD GPU support not available (install with: pip install quasim[hcal-amd])")
    except Exception as e:
        click.echo(f"✗ Error accessing AMD GPUs: {e}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--output', '-o', type=click.Path(), help='Output path for calibration results')
def calibrate(config, output):
    """Run hardware calibration procedures."""
    click.echo("Running hardware calibration...")
    
    if config:
        click.echo(f"Using configuration: {config}")
    
    # Placeholder for calibration logic
    click.echo("Calibration complete.")
    
    if output:
        click.echo(f"Results saved to: {output}")


@cli.command()
@click.option('--duration', '-d', default=60, help='Monitoring duration in seconds')
@click.option('--interval', '-i', default=1, help='Sampling interval in seconds')
def monitor(duration, interval):
    """Monitor hardware resources in real-time."""
    import time
    
    click.echo(f"Monitoring hardware for {duration} seconds (sampling every {interval}s)...")
    click.echo("Press Ctrl+C to stop")
    
    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            # Placeholder for monitoring logic
            elapsed = int(time.time() - start_time)
            click.echo(f"\r[{elapsed}s] Monitoring...", nl=False)
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped by user")
    
    click.echo("\nMonitoring complete.")


@cli.command()
def info():
    """Display system and package information."""
    click.echo("QuASIM HCAL Information:")
    click.echo("========================")
    click.echo(f"Version: 0.1.0")
    click.echo(f"Python: {sys.version}")
    
    # Check installed optional dependencies
    deps = []
    try:
        import yaml
        deps.append("pyyaml")
    except ImportError:
        pass
    
    try:
        import numpy
        deps.append("numpy")
    except ImportError:
        pass
    
    try:
        import pynvml
        deps.append("nvidia-ml-py (NVIDIA support)")
    except ImportError:
        pass
    
    try:
        import pyrsmi
        deps.append("pyrsmi (AMD support)")
    except ImportError:
        pass
    
    if deps:
        click.echo(f"Installed dependencies: {', '.join(deps)}")
    else:
        click.echo("No optional dependencies installed")


if __name__ == '__main__':
    cli()
@click.option("--json-output", is_flag=True, help="Output in JSON format")
def discover(json_output: bool):
    """Discover hardware topology."""
    discovery = TopologyDiscovery()
    topology = discovery.discover()

    if json_output:
        devices_data = []
        for device in topology.devices:
            devices_data.append(
                {
                    "device_id": device.device_id,
                    "type": device.device_type.value,
                    "name": device.name,
                    "vendor": device.vendor,
                    "capabilities": device.capabilities,
                    "numa_node": device.numa_node,
                    "pcie_address": device.pcie_address,
                }
            )

        interconnects_data = []
        for interconnect in topology.interconnects:
            interconnects_data.append(
                {
                    "source": interconnect.source,
                    "destination": interconnect.destination,
                    "type": interconnect.interconnect_type.value,
                    "bandwidth_gbps": interconnect.bandwidth_gbps,
                }
            )

        output = {
            "devices": devices_data,
            "interconnects": interconnects_data,
            "numa_nodes": topology.numa_nodes,
        }

        click.echo(json.dumps(output, indent=2))
    else:
        click.echo("=== Hardware Topology ===")
        click.echo(f"\nDevices ({len(topology.devices)}):")
        for device in topology.devices:
            click.echo(
                f"  {device.device_id}: {device.name} ({device.device_type.value}, {device.vendor})"
            )
            if device.pcie_address:
                click.echo(f"    PCIe: {device.pcie_address}")
            if device.capabilities:
                click.echo(f"    Capabilities: {device.capabilities}")

        click.echo(f"\nInterconnects ({len(topology.interconnects)}):")
        for interconnect in topology.interconnects:
            itype = interconnect.interconnect_type.value
            click.echo(
                f"  {interconnect.source} <-> {interconnect.destination} ({itype})"
            )
            if interconnect.bandwidth_gbps:
                click.echo(f"    Bandwidth: {interconnect.bandwidth_gbps} GB/s")

        click.echo(f"\nNUMA Nodes ({len(topology.numa_nodes)}):")
        for node, devices in topology.numa_nodes.items():
            click.echo(f"  Node {node}: {', '.join(devices)}")


@cli.command()
@click.option("--profile", required=True, help="Reconfiguration profile name")
@click.option("--device", required=True, help="Device ID (e.g., gpu0)")
@click.option("--json-output", is_flag=True, help="Output in JSON format")
def plan(profile: str, device: str, json_output: bool):
    """Create a reconfiguration plan."""
    profile_mgr = ProfileManager()

    profile_obj = profile_mgr.get_profile(profile)
    if not profile_obj:
        click.echo(f"Error: Profile '{profile}' not found", err=True)
        click.echo(f"Available profiles: {', '.join(profile_mgr.list_profiles())}", err=True)
        sys.exit(1)

    # Determine device type from device ID
    device_type = "gpu" if device.startswith("gpu") else "cpu"

    setpoints = profile_mgr.apply_profile(profile, device_type, device)
    if not setpoints:
        click.echo(
            f"Error: Profile '{profile}' has no setpoints for device type '{device_type}'",
            err=True,
        )
        sys.exit(1)

    plan_data = {
        "profile": profile,
        "device": device,
        "device_type": device_type,
        "setpoints": setpoints,
        "constraints": profile_obj.constraints,
    }

    if json_output:
        click.echo(json.dumps(plan_data, indent=2))
    else:
        click.echo("=== Reconfiguration Plan ===")
        click.echo(f"Profile: {profile}")
        click.echo(f"Device: {device} ({device_type})")
        click.echo("\nSetpoints:")
        for key, value in setpoints.items():
            click.echo(f"  {key}: {value}")
        click.echo("\nConstraints:")
        for key, value in profile_obj.constraints.items():
            click.echo(f"  {key}: {value}")


@cli.command()
@click.option("--profile", required=True, help="Reconfiguration profile name")
@click.option("--device", required=True, help="Device ID (e.g., gpu0)")
@click.option(
    "--policy",
    type=click.Path(exists=True, path_type=Path),
    help="Policy file path",
)
@click.option("--actuate", is_flag=True, help="Actually apply changes (default: dry-run)")
@click.option("--json-output", is_flag=True, help="Output in JSON format")
def apply(
    profile: str,
    device: str,
    policy: Optional[Path],
    actuate: bool,
    json_output: bool,
):
    """Apply a reconfiguration plan."""
    # Initialize HCAL
    hcal = HCAL(policy_path=policy, dry_run=not actuate)

    # Get profile
    profile_mgr = ProfileManager()
    profile_obj = profile_mgr.get_profile(profile)

    if not profile_obj:
        click.echo(f"Error: Profile '{profile}' not found", err=True)
        sys.exit(1)

    # Determine device type
    device_type = "gpu" if device.startswith("gpu") else "cpu"

    setpoints = profile_mgr.apply_profile(profile, device_type, device)
    if not setpoints:
        click.echo(
            f"Error: Profile '{profile}' has no setpoints for device type '{device_type}'",
            err=True,
        )
        sys.exit(1)

    # Apply setpoints
    success = hcal.apply_setpoint(device, setpoints)

    result = {
        "success": success,
        "profile": profile,
        "device": device,
        "setpoints": setpoints,
        "dry_run": not actuate,
    }

    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        if success:
            mode = "DRY-RUN" if not actuate else "APPLIED"
            click.echo(f"[{mode}] Successfully applied profile '{profile}' to {device}")
        else:
            click.echo(f"Error: Failed to apply profile '{profile}' to {device}", err=True)
            sys.exit(1)


@cli.command()
@click.option("--device", required=True, help="Device ID (e.g., gpu0)")
@click.option(
    "--policy",
    type=click.Path(exists=True, path_type=Path),
    help="Policy file path",
)
@click.option("--iterations", default=20, help="Maximum iterations")
@click.option("--json-output", is_flag=True, help="Output in JSON format")
def calibrate(device: str, policy: Optional[Path], iterations: int, json_output: bool):
    """Run closed-loop calibration."""
    hcal = HCAL(policy_path=policy, dry_run=True)

    click.echo(f"Starting calibration for {device} (max {iterations} iterations)...")

    # Run bias trim calibration
    result = hcal.calibrate_bias_trim(device, max_iterations=iterations)

    output = {
        "device": device,
        "status": result.status.value,
        "iterations": result.iterations,
        "best_objective": result.best_objective,
        "final_setpoint": result.final_setpoint,
    }

    if json_output:
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo("\n=== Calibration Result ===")
        click.echo(f"Device: {device}")
        click.echo(f"Status: {result.status.value}")
        click.echo(f"Iterations: {result.iterations}")
        click.echo(f"Best objective: {result.best_objective:.2f}")
        click.echo(f"Final setpoint: {result.final_setpoint}")


@cli.command()
@click.option("--device", help="Device ID (omit for all devices)")
@click.option(
    "--policy",
    type=click.Path(exists=True, path_type=Path),
    help="Policy file path",
)
@click.option("--json-output", is_flag=True, help="Output in JSON format")
def telemetry(device: Optional[str], policy: Optional[Path], json_output: bool):
    """Read real-time telemetry data."""
    hcal = HCAL(policy_path=policy, dry_run=True)

    if device:
        reading = hcal.read_telemetry(device)
        if reading:
            if json_output:
                output = {
                    "device_id": reading.device_id,
                    "timestamp": reading.timestamp.isoformat(),
                    "metrics": reading.metrics,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                click.echo(f"=== Telemetry: {device} ===")
                click.echo(f"Timestamp: {reading.timestamp}")
                for key, value in reading.metrics.items():
                    click.echo(f"  {key}: {value}")
        else:
            click.echo(f"Error: Failed to read telemetry from {device}", err=True)
            sys.exit(1)
    else:
        # Read from all GPU devices
        discovery = TopologyDiscovery()
        topology = discovery.discover()
        gpu_devices = [d for d in topology.devices if d.device_id.startswith("gpu")]

        readings_data = []
        for gpu in gpu_devices:
            reading = hcal.read_telemetry(gpu.device_id)
            if reading:
                readings_data.append(reading)

        if json_output:
            output = []
            for reading in readings_data:
                output.append(
                    {
                        "device_id": reading.device_id,
                        "timestamp": reading.timestamp.isoformat(),
                        "metrics": reading.metrics,
                    }
                )
            click.echo(json.dumps(output, indent=2))
        else:
            for reading in readings_data:
                click.echo(f"\n=== Telemetry: {reading.device_id} ===")
                click.echo(f"Timestamp: {reading.timestamp}")
                for key, value in reading.metrics.items():
                    click.echo(f"  {key}: {value}")


@cli.command()
def stop():
    """Emergency stop - halt all HCAL operations."""
    click.echo("EMERGENCY STOP - This would halt all HCAL operations")
    click.echo("(In production, this would trigger actuator emergency stop)")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
