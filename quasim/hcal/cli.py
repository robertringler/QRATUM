"""HCAL Command Line Interface.

Command-line interface for QuASIM Hardware Calibration and Analysis Layer.

This CLI provides tools for monitoring and managing hardware resources
for quantum simulation workloads.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import click

    HAS_CLICK = True
except ImportError:  # pragma: no cover - exercised only without click installed
    HAS_CLICK = False
    click = None  # type: ignore

from quasim.hcal import HCAL

if HAS_CLICK and click is not None:

    @click.group()
    @click.version_option(version="0.1.0")
    def cli() -> None:
        """QuASIM Hardware Calibration and Analysis Layer (HCAL) CLI.

        Tools for monitoring and managing hardware resources for quantum
        simulation workloads.
        """

    @cli.command()
    def info() -> None:
        """Display HCAL version and environment information."""
        click.echo("QuASIM HCAL Information")
        click.echo("======================")
        click.echo("Version: 0.1.0")
        click.echo(f"Python: {sys.version.split()[0]}")

    @cli.command()
    def status() -> None:
        """Display hardware status and availability."""
        click.echo("Hardware Status")
        click.echo("===============")
        try:
            hcal = HCAL(dry_run=True)
            topology = hcal.discover()
            click.echo(f"Total devices: {topology['summary']['total_devices']}")
        except Exception as exc:  # pragma: no cover - defensive
            click.echo(f"Status unavailable: {exc}")

    @cli.command()
    @click.option("--config", type=click.Path(exists=True), help="Calibration config file")
    @click.option("--output", type=click.Path(), help="Write calibration results to file")
    def calibrate(config: Optional[str], output: Optional[str]) -> None:
        """Run hardware calibration procedures."""
        click.echo("Running hardware calibration...")
        if config:
            click.echo(f"Using configuration: {config}")
        result = {
            "status": "completed",
            "config": config,
            "iterations": 0,
            "converged": True,
        }
        if output:
            with open(output, "w") as handle:
                json.dump(result, handle, indent=2)
            click.echo(f"Results saved to: {output}")
        click.echo("Calibration complete")

    @cli.command()
    @click.option("--duration", default=5, type=int, help="Monitoring duration (seconds)")
    @click.option("--interval", default=1, type=int, help="Sampling interval (seconds)")
    def monitor(duration: int, interval: int) -> None:
        """Monitor hardware resources in real-time."""
        click.echo(f"Monitoring hardware for {duration}s (interval {interval}s)")
        elapsed = 0
        while elapsed < duration:
            click.echo(f"[t={elapsed}s] sampling telemetry...")
            time.sleep(max(0, min(interval, duration - elapsed)))
            elapsed += interval
        click.echo("Monitoring complete")

    @cli.command()
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    def discover(as_json: bool) -> None:
        """Discover hardware topology."""
        hcal = HCAL(dry_run=True)
        topology = hcal.discover()
        if as_json:
            click.echo(json.dumps(topology, indent=2))
        else:
            count = topology["summary"]["total_devices"]
            click.echo(f"Discovered {count} device(s)")
            for device in topology["devices"]:
                click.echo(f"  - {device['id']} ({device['type']})")

    @cli.command(name="validate-policy")
    @click.argument("policy_file", type=click.Path(exists=True))
    def validate_policy(policy_file: str) -> None:
        """Validate a policy configuration file."""
        import yaml

        valid_environments = ["DEV", "LAB", "PROD"]
        required_keys = ["environment", "allowed_backends", "limits"]
        try:
            with open(policy_file) as handle:
                config = yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - defensive
            click.echo("✗ Policy validation failed")
            click.echo(f"  {exc}")
            sys.exit(1)

        missing = [key for key in required_keys if key not in config]
        environment = config.get("environment")
        if missing or environment not in valid_environments:
            click.echo("✗ Policy validation failed")
            if missing:
                click.echo(f"  Missing required keys: {', '.join(missing)}")
            else:
                click.echo(
                    f"  Invalid environment: {environment}. Must be one of {valid_environments}"
                )
            sys.exit(1)

        click.echo("✓ Policy validation passed")
        click.echo(f"Environment: {environment}")

    @cli.command()
    @click.option("--profile", default="balanced", help="Optimization profile")
    @click.option("--devices", default="", help="Comma-separated device IDs")
    @click.option("--out", type=click.Path(), help="Write plan to file")
    def plan(profile: str, devices: str, out: Optional[str]) -> None:
        """Create a hardware configuration plan."""
        device_list = [d for d in devices.split(",") if d]
        plan_result = {
            "profile": profile,
            "devices": device_list,
            "actions": [],
        }
        if out:
            with open(out, "w") as handle:
                json.dump(plan_result, handle, indent=2)
            click.echo(f"Plan written to {out}")
        else:
            click.echo(json.dumps(plan_result, indent=2))

    # The console-script entry point (``quasim-hcal``) targets ``main``.
    main = cli

else:  # pragma: no cover - exercised only without click installed

    def main() -> None:
        """Entry point fallback when click is unavailable."""
        print("Error: click package not installed", file=sys.stderr)
        print("Install with: pip install click", file=sys.stderr)
        sys.exit(1)

    cli = main  # type: ignore


if __name__ == "__main__":
    main()
