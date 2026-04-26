"""Telemetry output for QRATUM-Chess benchmarking.

Generates:
- Heatmaps of node density
- Time-per-move distributions
- Neural drift spectra
- Branching entropy curves
- Elo confidence intervals
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetryData:
    """Container for telemetry data."""

    timestamp: float = field(default_factory=time.time)
    node_density: dict[str, list[float]] = field(default_factory=dict)
    time_per_move: list[float] = field(default_factory=list)
    neural_drift: list[float] = field(default_factory=list)
    branching_entropy: list[float] = field(default_factory=list)
    elo_history: list[tuple[float, float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Enhanced fields for motif extraction
    cortex_activations: list[dict[str, float]] = field(default_factory=list)
    novelty_pressure: list[float] = field(default_factory=list)
    move_divergence: list[dict[str, Any]] = field(default_factory=list)
    pattern_inventions: list[dict[str, Any]] = field(default_factory=list)
    abstraction_signals: list[float] = field(default_factory=list)


class TelemetryOutput:
    """Telemetry generation and output for benchmarking.

    Collects and formats telemetry data from benchmark runs
    for analysis and visualization.
    """

    def __init__(self):
        """Initialize telemetry collector."""
        self.data = TelemetryData()
        self.history: list[TelemetryData] = []

    def record_node_density(self, position_fen: str, density_map: list[float]) -> None:
        """Record node density for a position.

        Args:
            position_fen: Position in FEN format.
            density_map: 64-element list of node visit densities per square.
        """
        self.data.node_density[position_fen] = density_map

    def record_time_per_move(self, time_ms: float) -> None:
        """Record time taken for a move.

        Args:
            time_ms: Time in milliseconds.
        """
        self.data.time_per_move.append(time_ms)

    def record_neural_drift(self, drift: float) -> None:
        """Record neural network evaluation drift.

        Args:
            drift: Drift value (difference from expected).
        """
        self.data.neural_drift.append(drift)

    def record_branching_entropy(self, entropy: float) -> None:
        """Record branching entropy.

        Args:
            entropy: Entropy of branching factor distribution.
        """
        self.data.branching_entropy.append(entropy)

    def record_elo(self, elo: float, low: float, high: float) -> None:
        """Record Elo rating with confidence interval.

        Args:
            elo: Elo rating.
            low: Lower confidence bound.
            high: Upper confidence bound.
        """
        self.data.elo_history.append((elo, low, high))

    def record_cortex_activation(
        self, tactical: float = 0.0, strategic: float = 0.0, conceptual: float = 0.0
    ) -> None:
        """Record cortex activation weights for a move.

        Args:
            tactical: Tactical cortex activation weight.
            strategic: Strategic cortex activation weight.
            conceptual: Conceptual cortex activation weight.
        """
        self.data.cortex_activations.append(
            {
                "tactical": tactical,
                "strategic": strategic,
                "conceptual": conceptual,
                "timestamp": time.time(),
            }
        )

    def record_novelty_pressure(self, pressure: float) -> None:
        """Record novelty pressure functional value Ω(a).

        Args:
            pressure: Novelty pressure value.
        """
        self.data.novelty_pressure.append(pressure)

    def record_move_divergence(
        self, position_fen: str, move_uci: str, engine_move: str, divergence_score: float
    ) -> None:
        """Record move divergence from engine databases.

        Args:
            position_fen: Position in FEN format.
            move_uci: Selected move in UCI notation.
            engine_move: Expected engine move in UCI notation.
            divergence_score: Divergence score (0.0 = match, 1.0 = completely different).
        """
        self.data.move_divergence.append(
            {
                "position": position_fen,
                "selected_move": move_uci,
                "engine_move": engine_move,
                "divergence": divergence_score,
                "timestamp": time.time(),
            }
        )

    def record_pattern_invention(
        self, position_fen: str, move_sequence: list[str], pattern_type: str, novelty_score: float
    ) -> None:
        """Record pattern invention event.

        Args:
            position_fen: Position in FEN format.
            move_sequence: Sequence of moves (UCI notation).
            pattern_type: Type of pattern (tactical/strategic/conceptual).
            novelty_score: Novelty score (0.0-1.0).
        """
        self.data.pattern_inventions.append(
            {
                "position": position_fen,
                "moves": move_sequence,
                "pattern_type": pattern_type,
                "novelty_score": novelty_score,
                "timestamp": time.time(),
            }
        )

    def record_abstraction_signal(self, signal: float) -> None:
        """Record abstraction learning signal.

        Args:
            signal: Abstraction learning signal value.
        """
        self.data.abstraction_signals.append(signal)

    def finalize_snapshot(self) -> TelemetryData:
        """Finalize current telemetry snapshot.

        Returns:
            Current telemetry data.
        """
        snapshot = self.data
        self.history.append(snapshot)
        self.data = TelemetryData()
        return snapshot

    def generate_node_density_heatmap(self, position_fen: str) -> dict[str, Any]:
        """Generate node density heatmap data.

        Args:
            position_fen: Position to generate heatmap for.

        Returns:
            Heatmap data in JSON-serializable format.
        """
        density = self.data.node_density.get(position_fen, [0.0] * 64)

        # Convert to 8x8 grid
        grid = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                sq = rank * 8 + file
                row.append(density[sq] if sq < len(density) else 0.0)
            grid.append(row)

        # Normalize
        max_val = max(max(row) for row in grid) if grid else 1.0
        if max_val > 0:
            grid = [[v / max_val for v in row] for row in grid]

        return {
            "position": position_fen,
            "grid": grid,
            "max_density": max_val,
            "timestamp": time.time(),
        }

    def generate_time_distribution(self) -> dict[str, Any]:
        """Generate time-per-move distribution data.

        Returns:
            Distribution statistics in JSON-serializable format.
        """
        times = self.data.time_per_move
        if not times:
            return {"error": "No time data recorded"}

        sorted_times = sorted(times)
        n = len(sorted_times)

        return {
            "count": n,
            "min": sorted_times[0],
            "max": sorted_times[-1],
            "mean": sum(times) / n,
            "median": sorted_times[n // 2],
            "p95": sorted_times[int(0.95 * n)] if n > 0 else 0,
            "p99": sorted_times[int(0.99 * n)] if n > 0 else 0,
            "histogram": self._compute_histogram(times, bins=20),
            "timestamp": time.time(),
        }

    def generate_neural_drift_spectrum(self) -> dict[str, Any]:
        """Generate neural drift spectrum.

        Returns:
            Drift spectrum in JSON-serializable format.
        """
        drift = self.data.neural_drift
        if not drift:
            return {"error": "No drift data recorded"}

        # Simple moving average for smoothing
        window = 10
        smoothed = []
        for i in range(len(drift)):
            start = max(0, i - window)
            smoothed.append(sum(drift[start : i + 1]) / (i - start + 1))

        return {
            "raw_drift": drift,
            "smoothed_drift": smoothed,
            "mean_drift": sum(drift) / len(drift),
            "max_drift": max(abs(d) for d in drift),
            "timestamp": time.time(),
        }

    def generate_branching_entropy_curve(self) -> dict[str, Any]:
        """Generate branching entropy curve.

        Returns:
            Entropy curve data in JSON-serializable format.
        """
        entropy = self.data.branching_entropy
        if not entropy:
            return {"error": "No entropy data recorded"}

        # Calculate trend (linear regression)
        n = len(entropy)
        x_mean = (n - 1) / 2
        y_mean = sum(entropy) / n

        numerator = sum((i - x_mean) * (e - y_mean) for i, e in enumerate(entropy))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 0 else 0

        return {
            "entropy_values": entropy,
            "mean_entropy": y_mean,
            "trend_slope": slope,
            "is_stabilizing": slope < 0,
            "timestamp": time.time(),
        }

    def generate_elo_confidence_plot(self) -> dict[str, Any]:
        """Generate Elo confidence interval plot data.

        Returns:
            Elo plot data in JSON-serializable format.
        """
        history = self.data.elo_history
        if not history:
            return {"error": "No Elo data recorded"}

        elos = [h[0] for h in history]
        lows = [h[1] for h in history]
        highs = [h[2] for h in history]

        return {
            "elo_values": elos,
            "confidence_low": lows,
            "confidence_high": highs,
            "final_elo": elos[-1] if elos else 0,
            "final_confidence": (highs[-1] - lows[-1]) if history else 0,
            "trend": "improving" if len(elos) > 1 and elos[-1] > elos[0] else "stable",
            "timestamp": time.time(),
        }

    def _compute_histogram(self, values: list[float], bins: int = 20) -> dict[str, Any]:
        """Compute histogram of values.

        Args:
            values: List of values.
            bins: Number of bins.

        Returns:
            Histogram data.
        """
        if not values:
            return {"bins": [], "counts": []}

        min_val = min(values)
        max_val = max(values)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1

        bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
        counts = [0] * bins

        for v in values:
            bin_idx = min(int((v - min_val) / bin_width), bins - 1)
            counts[bin_idx] += 1

        return {
            "bin_edges": bin_edges,
            "counts": counts,
            "bin_width": bin_width,
        }

    def export_json(self, filepath: str) -> None:
        """Export all telemetry data to JSON file.

        Args:
            filepath: Output file path.
        """
        output = {
            "current": {
                "node_density": self.data.node_density,
                "time_distribution": self.generate_time_distribution(),
                "neural_drift": self.generate_neural_drift_spectrum(),
                "branching_entropy": self.generate_branching_entropy_curve(),
                "elo_confidence": self.generate_elo_confidence_plot(),
                "cortex_activations": self.data.cortex_activations,
                "novelty_pressure": self.data.novelty_pressure,
                "move_divergence": self.data.move_divergence,
                "pattern_inventions": self.data.pattern_inventions,
                "abstraction_signals": self.data.abstraction_signals,
            },
            "history_count": len(self.history),
            "export_timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all telemetry data.

        Returns:
            Summary dictionary.
        """
        return {
            "positions_analyzed": len(self.data.node_density),
            "moves_timed": len(self.data.time_per_move),
            "drift_samples": len(self.data.neural_drift),
            "entropy_samples": len(self.data.branching_entropy),
            "elo_measurements": len(self.data.elo_history),
            "total_snapshots": len(self.history),
            "cortex_activations_recorded": len(self.data.cortex_activations),
            "novelty_pressure_samples": len(self.data.novelty_pressure),
            "move_divergences": len(self.data.move_divergence),
            "pattern_inventions": len(self.data.pattern_inventions),
            "abstraction_signals": len(self.data.abstraction_signals),
        }
