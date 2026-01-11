"""Training module for QRATUM-Chess neural networks.

Implements:
- Self-play reinforcement learning
- Expert data training
- Cross-generation knowledge distillation
- Adversarial training against engine ensemble
"""

from __future__ import annotations

from qratum_chess.training.selfplay import SelfPlayGenerator
from qratum_chess.training.trainer import NetworkTrainer
from qratum_chess.training.distillation import KnowledgeDistillation

__all__ = [
    "SelfPlayGenerator",
    "NetworkTrainer",
    "KnowledgeDistillation",
]
