"""Command-line interface for QUBIC visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from qubic.visualization.adapters.mesh import MeshAdapter
from qubic.visualization.adapters.quantum import QuantumSimulationAdapter
from qubic.visualization.adapters.tire import TireSimulationAdapter
from qubic.visualization.pipelines.static import StaticPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """QUBIC Visualization Subsystem - Unified rendering for simulation results."""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    help="Input simulation data file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output image file path",
)
@click.option(
    "--adapter",
    "-a",
    type=click.Choice(["tire", "quantum", "mesh"]),
    default="mesh",
    help="Simulation adapter type",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["matplotlib", "headless", "gpu"]),
    default="headless",
    help="Rendering backend",
)
@click.option(
    "--field",
    "-f",
    type=str,
    help="Scalar field to visualize",
)
@click.option(
    "--colormap",
    "-c",
    type=str,
    default="viridis",
    help="Colormap for scalar field",
)
@click.option(
    "--dpi",
    type=int,
    default=150,
    help="Output resolution (dots per inch)",
)
def render(
    input: str | None,
    output: str,
    adapter: str,
    backend: str,
    field: str | None,
    colormap: str,
    dpi: int,
):
    """Render a single-frame visualization."""
    logger.info(f"Rendering with {adapter} adapter and {backend} backend")

    # Initialize adapter
    if adapter == "tire":
        adapter_obj = TireSimulationAdapter()
        if not input:
            logger.info("No input provided, using synthetic tire data")
            data = adapter_obj.create_synthetic_tire()
        else:
            data = adapter_obj.load_data(input)
    elif adapter == "quantum":
        adapter_obj = QuantumSimulationAdapter()
        if not input:
            logger.info("No input provided, using synthetic quantum state")
            data = adapter_obj.create_synthetic_state(n_qubits=3)
        else:
            data = adapter_obj.load_data(input)
    else:  # mesh
        adapter_obj = MeshAdapter()
        if not input:
            logger.info("No input provided, using test sphere mesh")
            data = adapter_obj.create_test_mesh("sphere")
        else:
            data = adapter_obj.load_data(input)

    # Create pipeline
    pipeline = StaticPipeline(backend=backend, dpi=dpi)

    # Render and save
    output_path = Path(output)
    pipeline.render_and_save(
        data=data,
        output_path=output_path,
        scalar_field=field,
        colormap=colormap,
    )

    logger.info(f"Visualization saved to {output_path}")


@cli.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    help="Input time-series data file or directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output video file path (.mp4 or .gif)",
)
@click.option(
    "--field",
    "-f",
    type=str,
    help="Scalar field to visualize",
)
@click.option(
    "--fps",
    type=int,
    default=30,
    help="Frames per second",
)
@click.option(
    "--colormap",
    "-c",
    type=str,
    default="viridis",
    help="Colormap for scalar field",
)
def animate(
    input: str | None,
    output: str,
    field: str | None,
    fps: int,
    colormap: str,
):
    """Create an animation from time-series data."""
    logger.info(f"Creating animation at {fps} FPS")

    from qubic.visualization.adapters.timeseries import TimeSeriesAdapter
    from qubic.visualization.pipelines.timeseries import TimeSeriesPipeline

    # Load time-series data
    adapter = TimeSeriesAdapter()

    if not input:
        logger.info("No input provided, using synthetic time-series")
        adapter.create_synthetic_timeseries(n_steps=20)
    else:
        adapter.load_data(input)

    # Create pipeline and render
    pipeline = TimeSeriesPipeline()
    output_path = Path(output)

    # Determine format from extension
    extension = output_path.suffix.lower()
    format_type = "mp4" if extension == ".mp4" else "gif"

    pipeline.render_animation(
        adapter=adapter,
        output_path=output_path,
        scalar_field=field,
        fps=fps,
        format=format_type,
        colormap=colormap,
    )

    logger.info(f"Animation saved to {output_path}")


@cli.command()
@click.option(
    "--host",
    "-h",
    type=str,
    default="0.0.0.0",
    help="Server host address",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8765,
    help="Server port",
)
def stream(host: str, port: int):
    """Start a streaming visualization server."""
    logger.info(f"Starting streaming server on ws://{host}:{port}")

    from qubic.visualization.pipelines.streaming import StreamingPipeline

    pipeline = StreamingPipeline()

    # Create and start server
    try:
        import asyncio

        server = pipeline.create_server(host=host, port=port)

        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        pipeline.stop_streaming()


@cli.command()
@click.argument(
    "example_type",
    type=click.Choice(["tire", "quantum", "streaming"]),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./viz_output",
    help="Output directory for examples",
)
def example(example_type: str, output_dir: str):
    """Run a visualization example (tire, quantum, or streaming)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running {example_type} example, output to {output_path}")

    if example_type == "tire":
        from qubic.visualization.examples.render_tire import main as tire_main

        tire_main(output_path)

    elif example_type == "quantum":
        from qubic.visualization.examples.render_quantum import \
            main as quantum_main

        quantum_main(output_path)

    elif example_type == "streaming":
        from qubic.visualization.examples.stream_simulation import \
            main as streaming_main

        streaming_main()

    logger.info(f"{example_type} example completed successfully")


if __name__ == "__main__":
    cli()
