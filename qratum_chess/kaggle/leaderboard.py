"""Kaggle leaderboard tracking and rank monitoring for QRATUM-Chess.

Provides real-time leaderboard polling, rank tracking, and performance analysis.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class LeaderboardEntry:
    """Single entry on the leaderboard.

    Attributes:
        team_name: Team/user name
        score: Competition score
        rank: Current rank
        submissions: Number of submissions
        last_submission: Last submission time
    """

    team_name: str
    score: float
    rank: int
    submissions: int = 0
    last_submission: str | None = None


@dataclass
class RankHistory:
    """Historical rank tracking.

    Attributes:
        timestamp: Time of measurement
        rank: Rank at that time
        score: Score at that time
        delta_rank: Change from previous
        delta_score: Change from previous score
    """

    timestamp: float
    rank: int
    score: float
    delta_rank: int = 0
    delta_score: float = 0.0


@dataclass
class LeaderboardStats:
    """Statistics from leaderboard analysis.

    Attributes:
        current_rank: Current rank
        best_rank: Best rank achieved
        worst_rank: Worst rank
        avg_rank: Average rank
        rank_volatility: Rank volatility measure
        score_progression: Score progression rate
        submissions_count: Total submissions
        history: Rank history
    """

    current_rank: int
    best_rank: int
    worst_rank: int
    avg_rank: float
    rank_volatility: float
    score_progression: float
    submissions_count: int
    history: list[RankHistory] = field(default_factory=list)


class LeaderboardTracker:
    """Tracks competition leaderboard position and performance.

    Provides real-time monitoring and historical analysis.
    """

    def __init__(self, competition_id: str, team_name: str):
        """Initialize leaderboard tracker.

        Args:
            competition_id: Competition identifier
            team_name: Team/username to track
        """
        self.competition_id = competition_id
        self.team_name = team_name
        self.history: list[RankHistory] = []
        self.last_rank: int | None = None
        self.last_score: float | None = None

    def poll_leaderboard(self, timeout: int = 30) -> LeaderboardEntry | None:
        """Poll current leaderboard position.

        Args:
            timeout: Timeout in seconds

        Returns:
            Current leaderboard entry or None if not found
        """
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "leaderboard", self.competition_id, "--show"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                print(f"Warning: Failed to fetch leaderboard: {result.stderr}")
                return None

            # Parse leaderboard output
            lines = result.stdout.strip().split("\n")

            # Find our team
            for line in lines[1:]:  # Skip header
                if self.team_name in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return LeaderboardEntry(
                            team_name=self.team_name,
                            rank=int(parts[0]) if parts[0].isdigit() else 0,
                            score=float(parts[1]) if self._is_float(parts[1]) else 0.0,
                            submissions=(
                                int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                            ),
                            last_submission=parts[3] if len(parts) > 3 else None,
                        )

            # Not found on leaderboard yet
            return None

        except subprocess.TimeoutExpired:
            print("Warning: Leaderboard poll timeout")
            return None
        except Exception as e:
            print(f"Warning: Leaderboard poll error: {e}")
            return None

    def wait_for_score(
        self, max_wait: int = 300, poll_interval: int = 10
    ) -> LeaderboardEntry | None:
        """Wait for submission to be scored.

        Args:
            max_wait: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Leaderboard entry once scored, or None if timeout
        """
        print(f"Waiting for submission to be scored (max {max_wait}s)...")

        start_time = time.time()
        attempts = 0

        while (time.time() - start_time) < max_wait:
            attempts += 1
            print(f"  Poll attempt {attempts}...", end="\r")

            entry = self.poll_leaderboard()
            if entry and entry.score > 0:
                print("\n✓ Submission scored!")
                self._record_rank(entry.rank, entry.score)
                return entry

            time.sleep(poll_interval)

        print("\n⚠ Timeout waiting for score")
        return None

    def _record_rank(self, rank: int, score: float) -> None:
        """Record rank in history.

        Args:
            rank: Current rank
            score: Current score
        """
        delta_rank = 0
        delta_score = 0.0

        if self.last_rank is not None:
            delta_rank = self.last_rank - rank  # Positive = improved

        if self.last_score is not None:
            delta_score = score - self.last_score

        history_entry = RankHistory(
            timestamp=time.time(),
            rank=rank,
            score=score,
            delta_rank=delta_rank,
            delta_score=delta_score,
        )

        self.history.append(history_entry)
        self.last_rank = rank
        self.last_score = score

    def get_stats(self) -> LeaderboardStats | None:
        """Get leaderboard statistics.

        Returns:
            Leaderboard statistics or None if no history
        """
        if not self.history:
            return None

        ranks = [h.rank for h in self.history]
        scores = [h.score for h in self.history]

        current_rank = ranks[-1]
        best_rank = min(ranks)
        worst_rank = max(ranks)
        avg_rank = sum(ranks) / len(ranks)

        # Calculate rank volatility (standard deviation)
        rank_variance = sum((r - avg_rank) ** 2 for r in ranks) / len(ranks)
        rank_volatility = rank_variance**0.5

        # Calculate score progression (simple linear trend)
        if len(scores) > 1:
            score_progression = (scores[-1] - scores[0]) / len(scores)
        else:
            score_progression = 0.0

        return LeaderboardStats(
            current_rank=current_rank,
            best_rank=best_rank,
            worst_rank=worst_rank,
            avg_rank=avg_rank,
            rank_volatility=rank_volatility,
            score_progression=score_progression,
            submissions_count=len(self.history),
            history=self.history,
        )

    def print_status(self) -> None:
        """Print current leaderboard status."""
        stats = self.get_stats()

        if not stats:
            print("No leaderboard data yet")
            return

        print("\n" + "=" * 60)
        print(f"🏆 Kaggle Leaderboard Status - {self.competition_id}")
        print("=" * 60)
        print(f"Current Rank: #{stats.current_rank}")
        print(f"Best Rank: #{stats.best_rank}")

        if stats.history:
            last = stats.history[-1]
            if last.delta_rank > 0:
                print(f"Δ Rank: +{last.delta_rank} ⬆")
            elif last.delta_rank < 0:
                print(f"Δ Rank: {last.delta_rank} ⬇")
            else:
                print("Δ Rank: 0 ➡")

            print(f"Current Score: {last.score:.6f}")
            if last.delta_score != 0:
                print(f"Δ Score: {last.delta_score:+.6f}")

        print(f"Total Submissions: {stats.submissions_count}")
        print(f"Rank Volatility: {stats.rank_volatility:.2f}")
        print(f"Score Progression: {stats.score_progression:+.6f}/submission")
        print("=" * 60 + "\n")

    def save_history(self, output_file: str | Path) -> None:
        """Save rank history to file.

        Args:
            output_file: Output JSON file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        history_data = {
            "competition_id": self.competition_id,
            "team_name": self.team_name,
            "history": [
                {
                    "timestamp": h.timestamp,
                    "datetime": datetime.fromtimestamp(h.timestamp).isoformat(),
                    "rank": h.rank,
                    "score": h.score,
                    "delta_rank": h.delta_rank,
                    "delta_score": h.delta_score,
                }
                for h in self.history
            ],
            "stats": self.get_stats().__dict__ if self.get_stats() else None,
        }

        with open(output_path, "w") as f:
            json.dump(history_data, f, indent=2)

        print(f"✓ History saved: {output_path}")

    def load_history(self, input_file: str | Path) -> None:
        """Load rank history from file.

        Args:
            input_file: Input JSON file
        """
        input_path = Path(input_file)

        if not input_path.exists():
            return

        with open(input_path) as f:
            data = json.load(f)

        self.history = [
            RankHistory(
                timestamp=h["timestamp"],
                rank=h["rank"],
                score=h["score"],
                delta_rank=h.get("delta_rank", 0),
                delta_score=h.get("delta_score", 0.0),
            )
            for h in data.get("history", [])
        ]

        if self.history:
            last = self.history[-1]
            self.last_rank = last.rank
            self.last_score = last.score

        print(f"✓ History loaded: {len(self.history)} entries")

    def _is_float(self, value: str) -> bool:
        """Check if string is a valid float.

        Args:
            value: String to check

        Returns:
            True if valid float
        """
        try:
            float(value)
            return True
        except ValueError:
            return False
