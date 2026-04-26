"""Anti-Holographic Indicator Panel: Stochasticity and destabilization metrics.

Features:
- Move stochasticity metric and evaluation drift tracking
- Opponent destabilization index
- Anti-reconstructibility visualization
- Model-breaking behavior tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DestabilizationLevel(Enum):
    """Opponent destabilization level."""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class StochasticityMetrics:
    """Metrics for move stochasticity."""

    current_stochasticity: float = 0.0  # 0-1
    average_stochasticity: float = 0.0
    max_stochasticity: float = 0.0
    stochasticity_history: list[float] = field(default_factory=list)

    # Move unpredictability
    move_entropy: float = 0.0
    policy_variance: float = 0.0
    unexpected_move_count: int = 0


@dataclass
class EvaluationDrift:
    """Evaluation drift metrics."""

    current_drift: float = 0.0  # How much evaluation changed
    cumulative_drift: float = 0.0
    drift_history: list[float] = field(default_factory=list)
    drift_direction: float = 0.0  # Positive = improving, negative = worsening

    # Volatility
    volatility_score: float = 0.0
    stability_index: float = 1.0


@dataclass
class DestabilizationMetrics:
    """Opponent destabilization metrics."""

    level: DestabilizationLevel = DestabilizationLevel.MINIMAL
    score: float = 0.0  # 0-1

    # Components
    position_complexity: float = 0.0  # How complex is the position
    evaluation_variance: float = 0.0  # How uncertain is the opponent
    time_pressure_factor: float = 0.0  # Time pressure induced
    style_disruption: float = 0.0  # How much we've disrupted opponent's style

    # History
    destabilization_history: list[float] = field(default_factory=list)


@dataclass
class AntiReconstructibility:
    """Anti-reconstructibility metrics - how hard to predict our moves."""

    current_score: float = 0.0  # 0-1, higher = harder to predict

    # Components
    information_hiding: float = 0.0  # How much information we hide
    move_ambiguity: float = 0.0  # How many equally good moves
    strategic_depth: float = 0.0  # How deep is our plan
    style_variation: float = 0.0  # How much we vary our style

    # History
    history: list[float] = field(default_factory=list)


@dataclass
class AntiHolographicState:
    """Complete state of the anti-holographic panel."""

    # Core metrics
    stochasticity: StochasticityMetrics = field(default_factory=StochasticityMetrics)
    drift: EvaluationDrift = field(default_factory=EvaluationDrift)
    destabilization: DestabilizationMetrics = field(default_factory=DestabilizationMetrics)
    anti_reconstructibility: AntiReconstructibility = field(default_factory=AntiReconstructibility)

    # Aggregate metrics
    overall_anti_holographic_score: float = 0.0  # 0-1
    model_breaking_index: float = 0.0  # How much we're breaking opponent models

    # Display options
    show_stochasticity: bool = True
    show_drift: bool = True
    show_destabilization: bool = True
    show_anti_reconstructibility: bool = True
    show_history_graphs: bool = True
    animate_changes: bool = True

    # Thresholds
    stochasticity_threshold: float = 0.5
    drift_warning_threshold: float = 0.3
    destabilization_target: float = 0.7


class AntiHolographicPanel:
    """Anti-Holographic Indicator Panel for visualizing unpredictability metrics.

    This panel provides:
    - Move stochasticity visualization
    - Evaluation drift tracking
    - Opponent destabilization index
    - Anti-reconstructibility metrics
    - Model-breaking behavior analysis
    """

    # Colors for different states
    COLORS = {
        "low": (0.3, 0.8, 0.3),  # Green - low stochasticity/drift
        "medium": (1.0, 0.8, 0.0),  # Yellow - medium
        "high": (1.0, 0.4, 0.0),  # Orange - high
        "maximum": (1.0, 0.0, 0.4),  # Red - maximum
        "positive": (0.2, 0.8, 1.0),  # Blue - positive effect
        "negative": (1.0, 0.2, 0.4),  # Red - negative effect
    }

    # Destabilization level thresholds
    DESTABILIZATION_THRESHOLDS = {
        DestabilizationLevel.MINIMAL: 0.2,
        DestabilizationLevel.LOW: 0.4,
        DestabilizationLevel.MODERATE: 0.6,
        DestabilizationLevel.HIGH: 0.8,
        DestabilizationLevel.MAXIMUM: 1.0,
    }

    def __init__(self, width: int = 300, height: int = 400) -> None:
        """Initialize anti-holographic panel.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
        """
        self.width = width
        self.height = height
        self.state = AntiHolographicState()

    def reset(self) -> None:
        """Reset metrics for new game."""
        self.state = AntiHolographicState()

    def update_stochasticity(
        self,
        move_probabilities: list[float],
        selected_move_rank: int,
        total_moves: int,
    ) -> None:
        """Update stochasticity metrics.

        Args:
            move_probabilities: Probability distribution over moves
            selected_move_rank: Rank of the selected move (0 = best)
            total_moves: Total number of legal moves
        """
        # Calculate move entropy
        probs = np.array(move_probabilities)
        probs = probs / (probs.sum() + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(max(1, total_moves))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Stochasticity based on selecting non-top moves
        rank_stochasticity = min(1.0, selected_move_rank / max(1, total_moves - 1))

        # Policy variance
        variance = np.var(probs)

        # Combined stochasticity
        current_stoch = (
            0.4 * normalized_entropy + 0.4 * rank_stochasticity + 0.2 * (1 - min(1, variance * 10))
        )

        # Update metrics
        self.state.stochasticity.current_stochasticity = current_stoch
        self.state.stochasticity.move_entropy = normalized_entropy
        self.state.stochasticity.policy_variance = variance

        # Update history
        self.state.stochasticity.stochasticity_history.append(current_stoch)
        if len(self.state.stochasticity.stochasticity_history) > 100:
            self.state.stochasticity.stochasticity_history = (
                self.state.stochasticity.stochasticity_history[-100:]
            )

        # Update averages
        history = self.state.stochasticity.stochasticity_history
        self.state.stochasticity.average_stochasticity = np.mean(history)
        self.state.stochasticity.max_stochasticity = np.max(history)

        # Track unexpected moves
        if selected_move_rank > 0:
            self.state.stochasticity.unexpected_move_count += 1

    def update_evaluation_drift(
        self,
        previous_eval: float,
        current_eval: float,
        expected_eval: float | None = None,
    ) -> None:
        """Update evaluation drift metrics.

        Args:
            previous_eval: Evaluation before the move
            current_eval: Evaluation after the move
            expected_eval: Expected evaluation (if available)
        """
        # Calculate drift
        drift = abs(current_eval - previous_eval)
        direction = current_eval - previous_eval

        # Unexpected drift (deviation from expected)
        if expected_eval is not None:
            unexpected_drift = abs(current_eval - expected_eval)
            drift = max(drift, unexpected_drift)

        # Update metrics
        self.state.drift.current_drift = drift
        self.state.drift.cumulative_drift += drift
        self.state.drift.drift_direction = direction

        # Update history
        self.state.drift.drift_history.append(drift)
        if len(self.state.drift.drift_history) > 100:
            self.state.drift.drift_history = self.state.drift.drift_history[-100:]

        # Calculate volatility
        if len(self.state.drift.drift_history) >= 5:
            recent = self.state.drift.drift_history[-10:]
            self.state.drift.volatility_score = np.std(recent)
            self.state.drift.stability_index = 1 / (1 + self.state.drift.volatility_score)

    def update_destabilization(
        self,
        position_complexity: float,
        opponent_eval_variance: float,
        opponent_time_used_ratio: float,
        style_disruption: float = 0.0,
    ) -> None:
        """Update opponent destabilization metrics.

        Args:
            position_complexity: Position complexity (0-1)
            opponent_eval_variance: Variance in opponent's evaluation
            opponent_time_used_ratio: Ratio of time opponent used vs expected
            style_disruption: How much we've disrupted opponent's style
        """
        # Update components
        self.state.destabilization.position_complexity = position_complexity
        self.state.destabilization.evaluation_variance = opponent_eval_variance
        self.state.destabilization.time_pressure_factor = max(
            0, min(1, opponent_time_used_ratio - 1)
        )
        self.state.destabilization.style_disruption = style_disruption

        # Calculate overall destabilization score
        score = (
            0.3 * position_complexity
            + 0.25 * opponent_eval_variance
            + 0.25 * self.state.destabilization.time_pressure_factor
            + 0.2 * style_disruption
        )
        self.state.destabilization.score = min(1.0, score)

        # Determine level
        for level, threshold in sorted(
            self.DESTABILIZATION_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if score >= threshold * 0.9:  # Allow some buffer
                self.state.destabilization.level = level
                break

        # Update history
        self.state.destabilization.destabilization_history.append(score)
        if len(self.state.destabilization.destabilization_history) > 100:
            self.state.destabilization.destabilization_history = (
                self.state.destabilization.destabilization_history[-100:]
            )

    def update_anti_reconstructibility(
        self,
        move_ambiguity: float,
        information_revealed: float,
        plan_depth: int,
        style_variation: float,
    ) -> None:
        """Update anti-reconstructibility metrics.

        Args:
            move_ambiguity: How many equally good moves (normalized 0-1)
            information_revealed: Information revealed by the move (0-1)
            plan_depth: Depth of strategic plan
            style_variation: Variation from typical style (0-1)
        """
        # Update components
        self.state.anti_reconstructibility.move_ambiguity = move_ambiguity
        self.state.anti_reconstructibility.information_hiding = 1 - information_revealed
        self.state.anti_reconstructibility.strategic_depth = min(1.0, plan_depth / 10)
        self.state.anti_reconstructibility.style_variation = style_variation

        # Calculate overall score
        score = (
            0.3 * move_ambiguity
            + 0.3 * self.state.anti_reconstructibility.information_hiding
            + 0.2 * self.state.anti_reconstructibility.strategic_depth
            + 0.2 * style_variation
        )
        self.state.anti_reconstructibility.current_score = score

        # Update history
        self.state.anti_reconstructibility.history.append(score)
        if len(self.state.anti_reconstructibility.history) > 100:
            self.state.anti_reconstructibility.history = self.state.anti_reconstructibility.history[
                -100:
            ]

    def compute_overall_score(self) -> float:
        """Compute overall anti-holographic score.

        Returns:
            Overall score (0-1)
        """
        # Weighted combination of all metrics
        score = (
            0.25 * self.state.stochasticity.current_stochasticity
            + 0.2 * min(1.0, self.state.drift.current_drift)
            + 0.3 * self.state.destabilization.score
            + 0.25 * self.state.anti_reconstructibility.current_score
        )

        self.state.overall_anti_holographic_score = score

        # Model breaking index considers sustained unpredictability
        if len(self.state.stochasticity.stochasticity_history) >= 5:
            recent_stoch = np.mean(self.state.stochasticity.stochasticity_history[-5:])
            self.state.model_breaking_index = (
                0.5 * recent_stoch
                + 0.3 * self.state.destabilization.score
                + 0.2 * self.state.anti_reconstructibility.current_score
            )

        return score

    def _get_level_color(self, value: float) -> tuple[float, float, float]:
        """Get color for a metric value.

        Args:
            value: Metric value (0-1)

        Returns:
            RGB color tuple
        """
        if value < 0.25:
            return self.COLORS["low"]
        elif value < 0.5:
            return self.COLORS["medium"]
        elif value < 0.75:
            return self.COLORS["high"]
        else:
            return self.COLORS["maximum"]

    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.

        Returns:
            Dictionary with panel state for visualization
        """
        # Compute overall score
        self.compute_overall_score()

        # Stochasticity data
        stoch_data = {
            "current": self.state.stochasticity.current_stochasticity,
            "average": self.state.stochasticity.average_stochasticity,
            "max": self.state.stochasticity.max_stochasticity,
            "entropy": self.state.stochasticity.move_entropy,
            "variance": self.state.stochasticity.policy_variance,
            "unexpected_moves": self.state.stochasticity.unexpected_move_count,
            "history": self.state.stochasticity.stochasticity_history[-50:],
            "color": self._get_level_color(self.state.stochasticity.current_stochasticity),
        }

        # Drift data
        drift_data = {
            "current": self.state.drift.current_drift,
            "cumulative": self.state.drift.cumulative_drift,
            "direction": self.state.drift.drift_direction,
            "volatility": self.state.drift.volatility_score,
            "stability": self.state.drift.stability_index,
            "history": self.state.drift.drift_history[-50:],
            "color": self._get_level_color(min(1.0, self.state.drift.current_drift)),
            "direction_color": (
                self.COLORS["positive"]
                if self.state.drift.drift_direction > 0
                else self.COLORS["negative"]
            ),
        }

        # Destabilization data
        destab_data = {
            "level": self.state.destabilization.level.value,
            "score": self.state.destabilization.score,
            "components": {
                "complexity": self.state.destabilization.position_complexity,
                "eval_variance": self.state.destabilization.evaluation_variance,
                "time_pressure": self.state.destabilization.time_pressure_factor,
                "style_disruption": self.state.destabilization.style_disruption,
            },
            "history": self.state.destabilization.destabilization_history[-50:],
            "color": self._get_level_color(self.state.destabilization.score),
        }

        # Anti-reconstructibility data
        anti_recon_data = {
            "score": self.state.anti_reconstructibility.current_score,
            "components": {
                "ambiguity": self.state.anti_reconstructibility.move_ambiguity,
                "info_hiding": self.state.anti_reconstructibility.information_hiding,
                "depth": self.state.anti_reconstructibility.strategic_depth,
                "style_var": self.state.anti_reconstructibility.style_variation,
            },
            "history": self.state.anti_reconstructibility.history[-50:],
            "color": self._get_level_color(self.state.anti_reconstructibility.current_score),
        }

        return {
            "width": self.width,
            "height": self.height,
            "overall_score": self.state.overall_anti_holographic_score,
            "model_breaking_index": self.state.model_breaking_index,
            "stochasticity": stoch_data,
            "drift": drift_data,
            "destabilization": destab_data,
            "anti_reconstructibility": anti_recon_data,
            "display_options": {
                "show_stochasticity": self.state.show_stochasticity,
                "show_drift": self.state.show_drift,
                "show_destabilization": self.state.show_destabilization,
                "show_anti_reconstructibility": self.state.show_anti_reconstructibility,
                "show_history_graphs": self.state.show_history_graphs,
                "animate_changes": self.state.animate_changes,
            },
            "thresholds": {
                "stochasticity": self.state.stochasticity_threshold,
                "drift_warning": self.state.drift_warning_threshold,
                "destabilization_target": self.state.destabilization_target,
            },
            "colors": self.COLORS,
        }

    def to_json(self) -> str:
        """Serialize render data to JSON."""
        import json

        return json.dumps(self.get_render_data())
