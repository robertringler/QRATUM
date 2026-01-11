"""Elo certification protocol for QRATUM-Chess.

Computes Elo rating using logistic regression over self-play and
engine matches. Implements promotion gates and regression detection.

Promotion gate:
- ELO_QRATUM - ELO_SF17 ≥ +250
- Any regression ≥ 10 Elo triggers rollback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class EloRating:
    """Elo rating with confidence interval."""
    rating: float
    confidence_low: float
    confidence_high: float
    games_played: int
    win_rate: float
    
    @property
    def confidence_interval(self) -> float:
        """Width of 95% confidence interval."""
        return self.confidence_high - self.confidence_low


@dataclass
class EloCertificationResult:
    """Result of Elo certification."""
    generation: str
    elo_rating: EloRating
    baseline_elo: float
    elo_difference: float
    promotion_threshold: float
    is_promoted: bool
    is_regression: bool
    regression_threshold: float
    games_analyzed: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "elo": self.elo_rating.rating,
            "confidence": [self.elo_rating.confidence_low, self.elo_rating.confidence_high],
            "baseline_elo": self.baseline_elo,
            "elo_difference": self.elo_difference,
            "is_promoted": self.is_promoted,
            "is_regression": self.is_regression,
            "games": self.games_analyzed,
        }


class EloCertification:
    """Elo certification protocol for engine generations.
    
    Implements the promotion and regression gates:
    - Generation promoted if Elo ≥ baseline + 250
    - Generation rejected if Elo < previous - 10
    """
    
    # Standard Elo parameters
    K_FACTOR = 32  # Rating adjustment factor
    BASE_RATING = 1500  # Initial rating
    
    # Certification thresholds
    PROMOTION_THRESHOLD = 250  # Elo above baseline for promotion
    REGRESSION_THRESHOLD = 10  # Elo below previous for rejection
    CONFIDENCE_LEVEL = 0.95  # For confidence interval calculation
    
    def __init__(self, baseline_elo: float = 3500):
        """Initialize the certification protocol.
        
        Args:
            baseline_elo: Baseline Elo (e.g., Stockfish 17).
        """
        self.baseline_elo = baseline_elo
        self.generation_history: list[EloRating] = []
    
    def compute_elo_from_games(
        self,
        results: list[tuple[str, float]]  # List of (result, opponent_elo)
    ) -> EloRating:
        """Compute Elo rating from game results.
        
        Uses maximum likelihood estimation for Elo.
        
        Args:
            results: List of (result, opponent_elo) tuples.
                     result is "win", "loss", or "draw".
                     
        Returns:
            Computed Elo rating with confidence interval.
        """
        if not results:
            return EloRating(
                rating=self.BASE_RATING,
                confidence_low=self.BASE_RATING - 100,
                confidence_high=self.BASE_RATING + 100,
                games_played=0,
                win_rate=0.5,
            )
        
        # Convert results to scores
        scores = []
        opponent_elos = []
        for result, opp_elo in results:
            if result == "win":
                scores.append(1.0)
            elif result == "draw":
                scores.append(0.5)
            else:
                scores.append(0.0)
            opponent_elos.append(opp_elo)
        
        # MLE estimation using Newton-Raphson
        elo = self._mle_elo_estimation(scores, opponent_elos)
        
        # Compute confidence interval using Fisher information
        confidence_low, confidence_high = self._compute_confidence_interval(
            elo, scores, opponent_elos
        )
        
        win_rate = sum(scores) / len(scores)
        
        return EloRating(
            rating=elo,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            games_played=len(results),
            win_rate=win_rate,
        )
    
    def _mle_elo_estimation(
        self,
        scores: list[float],
        opponent_elos: list[float]
    ) -> float:
        """Maximum likelihood estimation of Elo rating.
        
        Args:
            scores: Game scores (1=win, 0.5=draw, 0=loss).
            opponent_elos: Opponent Elo ratings.
            
        Returns:
            Estimated Elo rating.
        """
        # Initial estimate from average performance
        avg_score = sum(scores) / len(scores)
        avg_opp = sum(opponent_elos) / len(opponent_elos)
        
        # Performance rating formula
        if avg_score >= 1.0:
            return avg_opp + 400
        elif avg_score <= 0.0:
            return avg_opp - 400
        
        # Initial estimate
        elo = avg_opp + 400 * math.log10(avg_score / (1 - avg_score))
        
        # Newton-Raphson refinement
        for _ in range(20):
            gradient = 0.0
            hessian = 0.0
            
            for score, opp_elo in zip(scores, opponent_elos):
                expected = self._expected_score(elo, opp_elo)
                gradient += score - expected
                hessian -= expected * (1 - expected)
            
            if abs(hessian) < 1e-10:
                break
            
            # Scale gradient by Elo conversion factor
            gradient *= 400 * math.log(10)
            hessian *= (400 * math.log(10)) ** 2
            
            step = -gradient / hessian if abs(hessian) > 1e-10 else 0
            elo += step
            
            if abs(step) < 0.01:
                break
        
        return elo
    
    def _expected_score(self, rating: float, opponent_rating: float) -> float:
        """Calculate expected score using Elo formula.
        
        Args:
            rating: Player's Elo rating.
            opponent_rating: Opponent's Elo rating.
            
        Returns:
            Expected score between 0 and 1.
        """
        return 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400))
    
    def _compute_confidence_interval(
        self,
        elo: float,
        scores: list[float],
        opponent_elos: list[float]
    ) -> tuple[float, float]:
        """Compute 95% confidence interval for Elo estimate.
        
        Uses Fisher information for variance estimation.
        
        Args:
            elo: Estimated Elo rating.
            scores: Game scores.
            opponent_elos: Opponent ratings.
            
        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        # Fisher information
        fisher = 0.0
        for opp_elo in opponent_elos:
            expected = self._expected_score(elo, opp_elo)
            fisher += expected * (1 - expected) * (400 * math.log(10)) ** 2
        
        if fisher < 1e-10:
            return elo - 200, elo + 200
        
        variance = 1.0 / fisher
        std_error = math.sqrt(variance)
        
        # 95% confidence interval (z = 1.96)
        margin = 1.96 * std_error
        
        return elo - margin, elo + margin
    
    def certify_generation(
        self,
        generation_name: str,
        game_results: list[tuple[str, float]]
    ) -> EloCertificationResult:
        """Certify a new engine generation.
        
        Args:
            generation_name: Name of the generation.
            game_results: Game results against rated opponents.
            
        Returns:
            Certification result with promotion/regression status.
        """
        elo_rating = self.compute_elo_from_games(game_results)
        elo_diff = elo_rating.rating - self.baseline_elo
        
        # Check promotion
        is_promoted = elo_diff >= self.PROMOTION_THRESHOLD
        
        # Check regression against previous generation
        is_regression = False
        if self.generation_history:
            prev_elo = self.generation_history[-1].rating
            if elo_rating.rating < prev_elo - self.REGRESSION_THRESHOLD:
                is_regression = True
        
        result = EloCertificationResult(
            generation=generation_name,
            elo_rating=elo_rating,
            baseline_elo=self.baseline_elo,
            elo_difference=elo_diff,
            promotion_threshold=self.PROMOTION_THRESHOLD,
            is_promoted=is_promoted,
            is_regression=is_regression,
            regression_threshold=self.REGRESSION_THRESHOLD,
            games_analyzed=len(game_results),
        )
        
        # Record generation if not regression
        if not is_regression:
            self.generation_history.append(elo_rating)
        
        return result
    
    def compute_elo_from_winrate(
        self,
        win_rate: float,
        opponent_elo: float,
        num_games: int = 1000
    ) -> EloRating:
        """Compute Elo from win rate against a single opponent.
        
        Args:
            win_rate: Win rate (0 to 1, including draws as 0.5).
            opponent_elo: Opponent's Elo rating.
            num_games: Number of games played.
            
        Returns:
            Estimated Elo rating.
        """
        # Clamp win rate to avoid infinity
        win_rate = max(0.001, min(0.999, win_rate))
        
        # Elo formula from win rate
        elo = opponent_elo + 400 * math.log10(win_rate / (1 - win_rate))
        
        # Confidence interval based on sample size
        std_error = 400 / math.sqrt(num_games)
        
        return EloRating(
            rating=elo,
            confidence_low=elo - 1.96 * std_error,
            confidence_high=elo + 1.96 * std_error,
            games_played=num_games,
            win_rate=win_rate,
        )
