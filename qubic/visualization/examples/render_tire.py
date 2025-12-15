"""Tire visualization example.

Demonstrates complete tire visualization workflow with thermal,
stress, and wear field rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qubic.visualization.adapters.tire import TireSimulationAdapter
from qubic.visualization.exporters.image import ImageExporter
from qubic.visualization.pipelines.static import StaticPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(output_dir: Path = Path("./viz_output")) -> None:
    """Run tire visualization example.

    Args:
        output_dir: Directory for output files
    """
    logger.info("=" * 60)
    logger.info("QUBIC Tire Visualization Example")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tire adapter and generate synthetic data
    logger.info("Creating synthetic tire mesh...")
    adapter = TireSimulationAdapter()
    tire_data = adapter.create_synthetic_tire(resolution=48, include_fields=True)

    logger.info(
        f"Tire mesh created: {len(tire_data.vertices)} vertices, "
        f"{len(tire_data.faces)} faces"
    )
    logger.info(f"Available fields: {list(tire_data.scalar_fields.keys())}")

    # Create visualization pipeline
    pipeline = StaticPipeline(backend="headless", dpi=150, figsize=(12, 10))

    # Render different field views
    fields = [
        ("temperature", "Temperature Distribution", "hot"),
        ("stress_von_mises", "Von Mises Stress", "plasma"),
        ("wear_depth", "Wear Depth", "YlOrRd"),
    ]

    for field_name, title, colormap in fields:
        logger.info(f"Rendering {title}...")

        tire_data.metadata["title"] = title
        output_path = output_dir / f"tire_{field_name}.png"

        pipeline.render_and_save(
            data=tire_data,
            output_path=output_path,
            scalar_field=field_name,
            colormap=colormap,
        )

        logger.info(f"  Saved to {output_path}")

    # Export high-resolution image
    logger.info("Exporting high-resolution image...")
    exporter = ImageExporter(dpi=300, figsize=(16, 12))

    tire_data.metadata["title"] = "Tire Temperature - High Resolution"
    hr_output = output_dir / "tire_temperature_highres.png"

    exporter.export_png(
        data=tire_data,
        output_path=hr_output,
        scalar_field="temperature",
        colormap="hot",
    )

    logger.info(f"  Saved to {hr_output}")

    # Export JPEG for web
    logger.info("Exporting JPEG for web...")
    web_output = output_dir / "tire_stress_web.jpg"

    tire_data.metadata["title"] = "Von Mises Stress"
    exporter.export_jpeg(
        data=tire_data,
        output_path=web_output,
        scalar_field="stress_von_mises",
        colormap="plasma",
        quality=90,
    )

    logger.info(f"  Saved to {web_output}")

    # Export interactive HTML
    try:
        from qubic.visualization.exporters.interactive import \
            InteractiveExporter

        logger.info("Exporting interactive HTML...")
        interactive_exporter = InteractiveExporter()

        html_output = output_dir / "tire_interactive.html"
        interactive_exporter.export_html(
            data=tire_data,
            output_path=html_output,
            scalar_field="temperature",
            title="Interactive Tire Temperature Visualization",
        )

        logger.info(f"  Saved to {html_output}")
        logger.info(f"  Open {html_output} in a web browser to interact")

    except ImportError as e:
        logger.warning(f"Skipping interactive export: {e}")

    logger.info("=" * 60)
    logger.info("Tire visualization example completed!")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
