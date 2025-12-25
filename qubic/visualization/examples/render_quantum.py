"""Quantum amplitude visualization example.

Demonstrates quantum circuit state visualization with amplitude
and phase representations.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qubic.visualization.adapters.quantum import QuantumSimulationAdapter
from qubic.visualization.exporters.image import ImageExporter
from qubic.visualization.pipelines.static import StaticPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(output_dir: Path = Path("./viz_output")) -> None:
    """Run quantum visualization example.

    Args:
        output_dir: Directory for output files
    """

    logger.info("=" * 60)
    logger.info("QUBIC Quantum Visualization Example")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create quantum adapter
    adapter = QuantumSimulationAdapter()

    # Generate different quantum states
    states = [
        (3, "superposition", "Equal Superposition (3 qubits)"),
        (3, "entangled", "Entangled State (Bell-like)"),
        (4, "superposition", "Superposition (4 qubits)"),
    ]

    # Create visualization pipeline
    pipeline = StaticPipeline(backend="headless", dpi=150, figsize=(12, 8))

    for n_qubits, state_type, title in states:
        logger.info(f"Visualizing: {title}")

        # Create synthetic quantum state
        quantum_data = adapter.create_synthetic_state(n_qubits=n_qubits, state_type=state_type)

        logger.info(f"  States: {2**n_qubits}, Type: {state_type}")

        # Render probability distribution
        quantum_data.metadata["title"] = f"{title} - Probability"
        prob_output = output_dir / f"quantum_{state_type}_{n_qubits}q_prob.png"

        pipeline.render_and_save(
            data=quantum_data,
            output_path=prob_output,
            scalar_field="probability",
            colormap="Blues",
        )

        logger.info(f"  Probability saved to {prob_output}")

        # Render phase information
        quantum_data.metadata["title"] = f"{title} - Phase"
        phase_output = output_dir / f"quantum_{state_type}_{n_qubits}q_phase.png"

        pipeline.render_and_save(
            data=quantum_data,
            output_path=phase_output,
            scalar_field="phase",
            colormap="hsv",
        )

        logger.info(f"  Phase saved to {phase_output}")

    # Export high-resolution version
    logger.info("Exporting high-resolution quantum state...")
    exporter = ImageExporter(dpi=300, figsize=(14, 10))

    quantum_data = adapter.create_synthetic_state(n_qubits=3, state_type="superposition")
    quantum_data.metadata["title"] = "Quantum Amplitude Distribution - High Resolution"

    hr_output = output_dir / "quantum_highres.png"
    exporter.export_png(
        data=quantum_data,
        output_path=hr_output,
        scalar_field="probability",
        colormap="viridis",
    )

    logger.info(f"  Saved to {hr_output}")

    # Export interactive visualization
    try:
        from qubic.visualization.exporters.interactive import \
            InteractiveExporter

        logger.info("Exporting interactive HTML...")
        interactive_exporter = InteractiveExporter()

        html_output = output_dir / "quantum_interactive.html"
        interactive_exporter.export_html(
            data=quantum_data,
            output_path=html_output,
            scalar_field="probability",
            title="Interactive Quantum State Visualization",
        )

        logger.info(f"  Saved to {html_output}")
        logger.info(f"  Open {html_output} in a web browser to interact")

    except ImportError as e:
        logger.warning(f"Skipping interactive export: {e}")

    logger.info("=" * 60)
    logger.info("Quantum visualization example completed!")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
