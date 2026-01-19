"""Self-play generator for QRATUM-Chess training.

Implements the self-play RL loop:
    for episode in range(N):
        state = initial_board()
        while not terminal(state):
            π = MCTS(state, fθ)
            action = sample(π, τ)
            state = step(state, action)
            store(state, π, z)

Where:
- π = improved policy from MCTS
- z = final game result propagated backward
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class GameSample:
    """Sample from a self-play game.

    Attributes:
        position_fen: Position in FEN format.
        policy: MCTS-improved policy distribution.
        value: Game outcome from this position's perspective.
        move_played: Move that was played.
    """

    position_fen: str
    policy: dict[str, float]  # Move UCI -> probability
    value: float  # -1 (loss), 0 (draw), +1 (win)
    move_played: str


@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation.

    Attributes:
        num_games: Number of games to generate.
        mcts_simulations: MCTS simulations per move.
        temperature: Temperature for move sampling.
        temperature_threshold: Ply after which temperature drops.
        dirichlet_alpha: Dirichlet noise alpha.
        dirichlet_epsilon: Fraction of prior from noise.
        max_moves: Maximum moves per game.
    """

    num_games: int = 1000
    mcts_simulations: int = 800
    temperature: float = 1.0
    temperature_threshold: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_moves: int = 500


class SelfPlayGenerator:
    """Generates training data through self-play.

    Uses MCTS with neural network guidance to generate games,
    then extracts (state, policy, value) tuples for training.
    """

    def __init__(self, config: SelfPlayConfig | None = None):
        """Initialize the self-play generator.

        Args:
            config: Self-play configuration.
        """
        self.config = config or SelfPlayConfig()
        self.samples: list[GameSample] = []

    def generate_games(self, num_games: int | None = None) -> list[list[GameSample]]:
        """Generate self-play games.

        Args:
            num_games: Number of games to generate.

        Returns:
            List of games, each containing a list of samples.
        """
        from qratum_chess.search.mcts import MCTSConfig, MCTSSearch

        num_games = num_games or self.config.num_games
        all_games = []

        # Configure MCTS
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            temperature=self.config.temperature,
            temperature_threshold=self.config.temperature_threshold,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )
        mcts = MCTSSearch(config=mcts_config)

        for game_idx in range(num_games):
            game_samples = self._play_game(mcts)
            all_games.append(game_samples)

            # Add to global samples
            self.samples.extend(game_samples)

        return all_games

    def _play_game(self, mcts: MCTSSearch) -> list[GameSample]:
        """Play a single self-play game.

        Args:
            mcts: MCTS search engine.

        Returns:
            List of samples from the game.
        """
        from qratum_chess.core.position import Position

        position = Position.starting()
        game_samples = []
        move_history = []

        for ply in range(self.config.max_moves):
            # Check for game end
            if position.is_checkmate() or position.is_stalemate() or position.is_draw():
                break

            legal_moves = position.generate_legal_moves()
            if not legal_moves:
                break

            # Run MCTS
            best_move, visit_counts, _ = mcts.search(
                position, add_noise=(ply < 10)  # Add noise in early game
            )

            if not best_move:
                break

            # Create policy from visit counts
            total_visits = sum(visit_counts.values()) if visit_counts else 1
            policy = {m.to_uci(): v / total_visits for m, v in visit_counts.items()}

            # Sample move based on temperature
            if ply < self.config.temperature_threshold:
                # Proportional sampling
                moves = list(policy.keys())
                probs = [policy[m] ** (1.0 / self.config.temperature) for m in moves]
                total = sum(probs)
                probs = [p / total for p in probs]

                move_uci = random.choices(moves, weights=probs, k=1)[0]
                from qratum_chess.core.position import Move

                selected_move = Move.from_uci(move_uci)
            else:
                # Greedy selection
                selected_move = best_move

            # Record sample (value will be filled in later)
            sample = GameSample(
                position_fen=position.to_fen(),
                policy=policy,
                value=0.0,  # Placeholder
                move_played=selected_move.to_uci(),
            )
            game_samples.append(sample)
            move_history.append(selected_move)

            # Make move
            position = position.make_move(selected_move)

        # Determine game result
        result = self._determine_result(position)

        # Propagate result back through samples
        for i, sample in enumerate(game_samples):
            # Value from white's perspective at that position
            # Odd ply = black to move, even ply = white to move
            if i % 2 == 0:  # White to move
                sample.value = result
            else:  # Black to move
                sample.value = -result

        return game_samples

    def _determine_result(self, position: Position) -> float:
        """Determine game result.

        Args:
            position: Final position.

        Returns:
            Result from white's perspective: 1 (white wins), -1 (black wins), 0 (draw).
        """
        from qratum_chess.core import Color

        if position.is_checkmate():
            # Side to move is in checkmate
            if position.side_to_move == Color.WHITE:
                return -1.0  # Black wins
            else:
                return 1.0  # White wins

        # Draw
        return 0.0

    def get_training_batch(self, batch_size: int) -> list[GameSample]:
        """Get a random batch of samples for training.

        Args:
            batch_size: Number of samples to return.

        Returns:
            List of randomly sampled game samples.
        """
        if len(self.samples) < batch_size:
            return self.samples.copy()

        return random.sample(self.samples, batch_size)

    def clear_samples(self) -> None:
        """Clear all stored samples."""
        self.samples = []

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about generated samples.

        Returns:
            Dictionary of statistics.
        """
        if not self.samples:
            return {"total_samples": 0}

        wins = sum(1 for s in self.samples if s.value > 0.5)
        losses = sum(1 for s in self.samples if s.value < -0.5)
        draws = len(self.samples) - wins - losses

        return {
            "total_samples": len(self.samples),
            "white_wins": wins,
            "black_wins": losses,
            "draws": draws,
            "avg_policy_entropy": sum(
                -sum(p * (0 if p <= 0 else __import__("math").log(p)) for p in s.policy.values())
                for s in self.samples
            )
            / len(self.samples),
        }
