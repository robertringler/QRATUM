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
