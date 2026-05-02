"""Kaggle Competition Integration for QRATUM-Chess.

This module provides full integration with Kaggle chess competitions,
including authentication, dataset management, submission, and leaderboard tracking.

Core modules:
- client: Kaggle API client with authentication
- submission: Competition submission formatting and validation
- leaderboard: Leaderboard polling and rank tracking
- config: Configuration management for Kaggle credentials
"""

from __future__ import annotations

from qratum_chess.kaggle.client import KaggleClient
from qratum_chess.kaggle.config import KaggleConfig
from qratum_chess.kaggle.leaderboard import LeaderboardTracker
from qratum_chess.kaggle.submission import SubmissionFormatter, SubmissionValidator

__all__ = [
    "KaggleClient",
    "SubmissionFormatter",
    "SubmissionValidator",
    "LeaderboardTracker",
    "KaggleConfig",
]
