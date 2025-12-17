"""Headless rendering backend for CI/cluster environments."""

from __future__ import annotations

import matplotlib

# Set non-interactive backend before importing pyplot
matplotlib.use("Agg")

from qubic.visualization.backends.matplotlib_backend import MatplotlibBackend


class HeadlessBackend(MatplotlibBackend):
    """Headless rendering backend extending MatplotlibBackend.

    Uses the Agg backend for rendering without display, suitable for
    CI environments and batch processing on clusters.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize headless backend.

        Args:
            *args: Positional arguments for MatplotlibBackend
            **kwargs: Keyword arguments for MatplotlibBackend
        """
        # Ensure Agg backend is active
        matplotlib.use("Agg")
        super().__init__(*args, **kwargs)

    def show(self) -> None:
        """Override show() to prevent display attempts.

        Raises:
            RuntimeError: Always, as headless mode cannot display
        """
        raise RuntimeError("Cannot display in headless mode. Use save() to export to file.")
