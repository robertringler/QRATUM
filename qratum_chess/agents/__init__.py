"""Multi-agent orchestration for QRATUM-Chess.

Implements the agent pipeline within QRATUM:
- Board manager agent
- Evaluation agent
- Move proposal agent
- Rule validator agent
- Meta-strategy director agent

Each agent communicates via structured messages with defined protocols.
"""

from __future__ import annotations

from qratum_chess.agents.board_manager import BoardManagerAgent
from qratum_chess.agents.director import MetaStrategyDirector
from qratum_chess.agents.evaluator import EvaluationAgent
from qratum_chess.agents.move_proposer import MoveProposalAgent
from qratum_chess.agents.orchestrator import AgentOrchestrator
from qratum_chess.agents.validator import RuleValidatorAgent

__all__ = [
    "AgentOrchestrator",
    "BoardManagerAgent",
    "EvaluationAgent",
    "MoveProposalAgent",
    "RuleValidatorAgent",
    "MetaStrategyDirector",
]
