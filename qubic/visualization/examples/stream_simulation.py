"""Streaming visualization server example.

Demonstrates real-time WebSocket-based visualization streaming.
"""

from __future__ import annotations

import asyncio
import logging
import signal

import numpy as np

from qubic.visualization.adapters.mesh import MeshAdapter
from qubic.visualization.pipelines.streaming import StreamingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_animated_sphere(t: float) -> dict:
    """Create animated sphere data.

    Args:
        t: Time parameter [0, 1]

    Returns:
        Dictionary with sphere mesh data
    """

    resolution = 20

    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    # Radius varies with time and position
    radius = 1.0 + 0.2 * np.sin(2 * np.pi * t) + 0.1 * np.cos(phi * 3)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Generate faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1)
            v2 = (i + 1) * resolution + (j + 1)
            v3 = (i + 1) * resolution + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = np.array(faces)

    # Field that varies with time
    field = np.sin(2 * np.pi * t + vertices[:, 2] * 3)

    return {"vertices": vertices, "faces": faces, "field": field}


def main() -> None:
    """Run streaming visualization example."""

    logger.info("=" * 60)
    logger.info("QUBIC Streaming Visualization Example")
    logger.info("=" * 60)

    # Create streaming pipeline
    pipeline = StreamingPipeline(figsize=(10, 8), dpi=100, max_fps=10)

    # Create mesh adapter
    adapter = MeshAdapter()

    # Time counter for animation
    time_counter = [0.0]  # Use list to make it mutable in closure

    def data_generator():
        """Generate animated data."""

        # Update time
        time_counter[0] = (time_counter[0] + 0.05) % 1.0
        t = time_counter[0]

        # Create animated mesh
        mesh_data = create_animated_sphere(t)

        # Load through adapter
        vis_data = adapter.load_data(mesh_data)
        vis_data.metadata["title"] = f"Streaming Demo (t={t:.2f})"

        return vis_data

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down streaming server...")
        pipeline.stop_streaming()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    logger.info("Starting WebSocket server on ws://0.0.0.0:8765")
    logger.info("Connect with a WebSocket client to receive frames")
    logger.info("Press Ctrl+C to stop")
    logger.info("-" * 60)

    try:
        # Create WebSocket server
        server = pipeline.create_server(host="0.0.0.0", port=8765)

        # Start event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(server)

        # Start streaming in parallel
        loop.create_task(
            pipeline.stream_data(
                data_generator=data_generator,
                scalar_field="field",
                colormap="viridis",
            )
        )

        # Run forever
        loop.run_forever()

    except Exception as e:
        logger.error(f"Server error: {e}")

    finally:
        pipeline.stop_streaming()
        logger.info("=" * 60)
        logger.info("Streaming example stopped")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
