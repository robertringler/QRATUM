"""QRATUM-Chess: A professional-strength chess engine powered by QRATUM AI platform.

This module implements a complete chess engine system including:
- Core chess logic with bitboard representation
- Neural evaluation networks (Tri-Modal Cognitive Core)
- Advanced search algorithms (Alpha-Beta, MCTS, Asymmetric Adaptive Search)
- Multi-agent orchestration for strategic decision making
- Comprehensive benchmarking and load-testing framework
- UCI protocol interface for chess GUI integration

Stage I: Core Engine Components
Stage II: Neural Evaluation Network Architecture
Stage III: Tri-Modal Cognitive Core (Tactical, Strategic, Conceptual)
Stage IV: Benchmarking & Load-Testing Protocol
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "ChessEngine",
    "Position",
    "Move",
    "BitBoard",
    "NeuralEvaluator",
    "TriModalCore",
    "BenchmarkRunner",
]
