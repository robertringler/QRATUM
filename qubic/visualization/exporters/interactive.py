"""Interactive HTML/WebGL export functionality."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qubic.visualization.core.data_model import VisualizationData

logger = logging.getLogger(__name__)


class InteractiveExporter:
    """Export interactive HTML/WebGL visualizations.

    Creates standalone HTML files with embedded 3D viewers for
    interactive exploration in web browsers.
    """

    def __init__(self) -> None:
        """Initialize interactive exporter."""

        pass

    def export_html(
        self,
        data: VisualizationData,
        output_path: Path,
        scalar_field: str | None = None,
        title: str = "Interactive Visualization",
    ) -> None:
        """Export as interactive HTML with embedded viewer.

        Args:
            data: Visualization data to export
            output_path: Output HTML file path
            scalar_field: Name of scalar field for color mapping
            title: Title for the visualization

        Raises:
            ImportError: If plotly is not available
        """

        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for interactive HTML export. Install with: pip install plotly"
            ) from None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting interactive HTML to {output_path}")

        # Create plotly mesh
        vertices = data.vertices
        faces = data.faces

        # Prepare color data
        if scalar_field and scalar_field in data.scalar_fields:
            intensity = data.scalar_fields[scalar_field]
            colorscale = "Viridis"
        else:
            intensity = None
            colorscale = None

        # Create 3D mesh
        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=intensity,
            colorscale=colorscale,
            name="mesh",
            showscale=intensity is not None,
        )

        # Create figure
        fig = go.Figure(data=[mesh])

        # Update layout
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Z",
                "aspectmode": "data",
            },
            width=1000,
            height=800,
        )

        # Save as HTML
        fig.write_html(output_path, include_plotlyjs="cdn")

        logger.info(f"Interactive HTML saved to {output_path}")

    def export_json(
        self,
        data: VisualizationData,
        output_path: Path,
        scalar_field: str | None = None,
    ) -> None:
        """Export data as JSON for custom web viewers.

        Args:
            data: Visualization data to export
            output_path: Output JSON file path
            scalar_field: Name of scalar field to include
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting JSON data to {output_path}")

        # Prepare data dictionary
        export_data = {
            "vertices": data.vertices.tolist(),
            "faces": data.faces.tolist(),
            "normals": data.normals.tolist() if data.normals is not None else None,
            "metadata": data.metadata,
        }

        # Include scalar field if specified
        if scalar_field and scalar_field in data.scalar_fields:
            export_data["scalar_field"] = {
                "name": scalar_field,
                "values": data.scalar_fields[scalar_field].tolist(),
            }

        # Write JSON
        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"JSON data saved to {output_path}")

    def export(
        self,
        data: VisualizationData,
        output_path: Path,
        scalar_field: str | None = None,
        title: str = "Interactive Visualization",
        **kwargs,
    ) -> None:
        """Export with format auto-detected from extension.

        Args:
            data: Visualization data to export
            output_path: Output file path (extension determines format)
            scalar_field: Name of scalar field for color mapping
            title: Title for the visualization
            **kwargs: Additional format-specific arguments

        Raises:
            ValueError: If file extension is not supported
        """

        output_path = Path(output_path)
        extension = output_path.suffix.lower()

        if extension == ".html":
            self.export_html(data, output_path, scalar_field, title, **kwargs)
        elif extension == ".json":
            self.export_json(data, output_path, scalar_field, **kwargs)
        else:
            raise ValueError(
                f"Unsupported interactive format: {extension}. Supported formats: .html, .json"
            )
