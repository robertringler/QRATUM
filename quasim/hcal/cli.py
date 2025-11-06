"""HCAL Command Line Interface."""

import json
import sys
from pathlib import Path
from typing import Optional

from . import HCAL, Policy

try:
    import click

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False
    click = None  # type: ignore

if HAS_CLICK and click is not None:
    @click.group()
    def cli():
        """HCAL - Hardware Control Abstraction Layer CLI."""
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
                # Use default policy
                default_policy = Policy({
                    "environment": "DEV",
                    "allowed_backends": ["nvml", "rocm_smi"],
                    "device_allowlist": [],
                })
                hcal = HCAL(default_policy)

            topology = hcal.discover(full=True)

            if output_json:
                print(json.dumps(topology, indent=2))
            else:
                print(f"Discovered {topology['summary']['total_devices']} devices")
                for device in topology["devices"]:
                    print(f"  - {device['id']} ({device['type']})")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
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
                default_policy = Policy({
                    "environment": "DEV",
                    "allowed_backends": ["nvml"],
                    "device_allowlist": ["GPU0", "GPU1"],
                })
                hcal = HCAL(default_policy)

            device_list = devices.split(",") if devices else None
            plan_result = hcal.plan(profile=profile, devices=device_list)

            if out:
                with open(out, "w") as f:
                    json.dump(plan_result, f, indent=2)
                print(f"Plan written to {out}")
            else:
                print(json.dumps(plan_result, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
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
                default_policy = Policy({
                    "environment": "DEV",
                    "allowed_backends": ["nvml"],
                    "device_allowlist": ["GPU0", "GPU1"],
                    "limits": {"power_watts_max": 300},
                })
                hcal = HCAL(default_policy)

            with open(plan_file) as f:
                plan_data = json.load(f)

            result = hcal.apply(plan_data, enable_actuation=not dry_run)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    """CLI entry point."""
    if click is None:
        print("Error: click package not installed", file=sys.stderr)
        print("Install with: pip install click", file=sys.stderr)
        sys.exit(1)

    cli()


if __name__ == "__main__":
    main()
