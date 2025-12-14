"""Command-line interface for HCAL.

Command-line interface for QuASIM Hardware Calibration and Analysis Layer.

This CLI provides tools for monitoring and managing hardware resources
for quantum simulation workloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import click

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False
    click = None  # type: ignore

from quasim.hcal import HCAL
from quasim.hcal.device import DeviceManager
from quasim.hcal.loops.reconfig_profiles import ProfileManager
from quasim.hcal.policy import Policy, PolicyValidator
from quasim.hcal.topology import TopologyDiscovery


if HAS_CLICK and click is not None:

    @click.group()
    @click.version_option(version="0.1.0")
    def cli():
        """QuASIM Hardware Calibration and Analysis Layer (HCAL) CLI.

        Tools for monitoring and managing hardware resources for quantum simulation.
        """
        pass

    @cli.command()
    @click.option("--json", "output_json", is_flag=True, help="Output as JSON")
    @click.option("--policy", type=click.Path(exists=True), help="Policy file path")
    def discover(output_json: bool, policy: Optional[str]):
        """Discover hardware topology."""
        try:
            if policy:
                hcal = HCAL.from_policy(Path(policy))
            else:
                # Use default policy (no policy file)
                hcal = HCAL(dry_run=True)

            topology = hcal.discover(full=True)

            if output_json:
                click.echo(json.dumps(topology, indent=2))
            else:
                click.echo(f"Discovered {topology['summary']['total_devices']} devices")
                for device in topology["devices"]:
                    click.echo(f"  - {device['id']} ({device['type']})")
        except Exception as e:
            click.echo(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    @cli.command()
    @click.argument("policy_path", type=click.Path(exists=True, path_type=Path))
    def validate_policy(policy_path: Path) -> None:
        """Validate a policy configuration file."""
        try:
            validator = PolicyValidator.from_file(policy_path)
            click.echo("✓ Policy validation passed")
            if validator.policy:
                click.echo(f"  Environment: {validator.policy.environment}")
                click.echo(f"  Allowed backends: {', '.join(validator.policy.allowed_backends)}")
                click.echo(f"  Limits: {validator.policy.limits}")
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"✗ Policy validation failed: {e}", err=True)
            sys.exit(1)

    @cli.command()
    @click.option("--profile", required=True, help="Configuration profile")
    @click.option("--devices", help="Comma-separated device IDs")
    @click.option("--out", type=click.Path(), help="Output file path")
    @click.option("--policy", type=click.Path(exists=True), help="Policy file path")
    def plan(profile: str, devices: Optional[str], out: Optional[str], policy: Optional[str]):
        """Create hardware configuration plan."""
        try:
            if policy:
                hcal = HCAL.from_policy(Path(policy))
            else:
                # Use default policy (no policy file)
                hcal = HCAL(dry_run=True)

            device_list = devices.split(",") if devices and devices.strip() else None
            plan_result = hcal.plan(profile=profile, devices=device_list)

            if out:
                with open(out, "w") as f:
                    json.dump(plan_result, f, indent=2)
                click.echo(f"Plan written to {out}")
            else:
                click.echo(json.dumps(plan_result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    @cli.command()
    @click.argument("plan_file", type=click.Path(exists=True))
    @click.option("--dry-run", is_flag=True, help="Dry run mode")
    @click.option("--policy", type=click.Path(exists=True), help="Policy file path")
    def apply(plan_file: str, dry_run: bool, policy: Optional[str]):
        """Apply hardware configuration plan."""
        try:
            if policy:
                hcal = HCAL.from_policy(Path(policy))
            else:
                # Use default policy (no policy file)
                hcal = HCAL(dry_run=True)

            with open(plan_file) as f:
                plan_data = json.load(f)

            result = hcal.apply(plan_data, enable_actuation=not dry_run)
            click.echo(json.dumps(result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

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
                    name = name.decode("utf-8")
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = memory_info.total / (1024**3)
                free_gb = memory_info.free / (1024**3)
                click.echo(f"  GPU {i}: {name}")
                click.echo(f"    Memory: {free_gb:.2f}GB / {total_gb:.2f}GB free")

            pynvml.nvmlShutdown()
        except ImportError:
            click.echo(
                "✗ NVIDIA GPU support not available (install with: pip install quasim[hcal-nvidia])"
            )
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
    @click.option("--duration", "-d", default=60, help="Monitoring duration in seconds")
    @click.option("--interval", "-i", default=1, help="Sampling interval in seconds")
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
        click.echo("Version: 0.1.0")
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
    if click is None:
        print("Error: click package not installed", file=sys.stderr)
        print("Install with: pip install click", file=sys.stderr)
        sys.exit(1)
    cli()


if __name__ == "__main__":
    main()
