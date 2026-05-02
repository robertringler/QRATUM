"""Export utilities for simulation screenshots, tensor logs, and CSV metrics.

Provides:
    ScreenshotCapture — Capture matplotlib figures as PNG screenshots
    TensorLogger — Log tensor states and metrics to JSON
    MetricsCSVWriter — Stream per-step metrics to CSV
    SimulationExporter — Orchestrate all exports for a simulation run
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ================================================================
# ScreenshotCapture
# ================================================================


class ScreenshotCapture:
    """Capture matplotlib figures as PNG screenshots.

    Parameters
    ----------
    output_dir : str
        Directory to save screenshots.
    dpi : int
        Image resolution in DPI.
    """

    def __init__(self, output_dir: str, dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        self._count = 0

    def capture(self, fig: Any, prefix: str = "frame", step: int | None = None) -> str:
        """Save a matplotlib figure as PNG.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        prefix : str
            Filename prefix.
        step : int, optional
            Step number for filename (auto-increments if not given).

        Returns
        -------
        path : str
            Path to the saved screenshot.
        """
        if step is None:
            step = self._count
        filename = f"{prefix}_{step:06d}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        self._count += 1
        return path

    @property
    def count(self) -> int:
        return self._count


# ================================================================
# TensorLogger
# ================================================================


class TensorLogger:
    """Log tensor states and computation metadata to JSON.

    Parameters
    ----------
    path : str
        Output JSON file path.
    """

    def __init__(self, path: str):
        self.path = path
        self._entries: list[dict[str, Any]] = []

    def log(self, step: int, data: dict[str, Any]) -> None:
        """Append a log entry.

        Parameters
        ----------
        step : int
            Simulation step.
        data : dict
            Tensor metadata (norms, spectra, shapes, etc.).
        """
        entry = {"step": step, **data}
        self._entries.append(entry)

    def flush(self) -> None:
        """Write all entries to JSON file."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._entries, f, indent=2, default=_json_default)

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ================================================================
# MetricsCSVWriter
# ================================================================


class MetricsCSVWriter:
    """Stream per-step metrics to a CSV file.

    Parameters
    ----------
    path : str
        Output CSV file path.
    fieldnames : list of str
        Column names.
    """

    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._count = 0

    def write_row(self, row: dict[str, Any]) -> None:
        """Write a single metrics row."""
        self._writer.writerow(row)
        self._count += 1

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.close()

    @property
    def row_count(self) -> int:
        return self._count

    def __enter__(self) -> MetricsCSVWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ================================================================
# SimulationExporter
# ================================================================


@dataclass
class SimulationExporter:
    """Orchestrate all export streams for a simulation run.

    Parameters
    ----------
    output_dir : str
        Base output directory.
    dpi : int
        Screenshot DPI.
    """

    output_dir: str
    dpi: int = 150
    screenshot_capture: ScreenshotCapture | None = None
    tensor_logger: TensorLogger | None = None
    metrics_writer: MetricsCSVWriter | None = None

    def init(self, fieldnames: list[str] | None = None) -> None:
        """Initialize all export streams."""
        os.makedirs(self.output_dir, exist_ok=True)

        self.screenshot_capture = ScreenshotCapture(
            os.path.join(self.output_dir, "screenshots"), dpi=self.dpi)
        self.tensor_logger = TensorLogger(
            os.path.join(self.output_dir, "tensor_log.json"))

        if fieldnames is None:
            fieldnames = [
                "step", "total_loss", "constraint_loss", "observer_loss",
                "entropy_loss", "gradient_norm", "step_time_ms",
            ]
        self.metrics_writer = MetricsCSVWriter(
            os.path.join(self.output_dir, "metrics.csv"), fieldnames)

    def finalize(self) -> None:
        """Flush and close all export streams."""
        if self.tensor_logger:
            self.tensor_logger.flush()
        if self.metrics_writer:
            self.metrics_writer.close()


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
