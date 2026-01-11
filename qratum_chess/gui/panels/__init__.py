"""Panel components for QRATUM-Chess GUI."""

from __future__ import annotations

from qratum_chess.gui.panels.board import BoardPanel
from qratum_chess.gui.panels.tricortex import TriCortexPanel
from qratum_chess.gui.panels.search_tree import SearchTreePanel
from qratum_chess.gui.panels.motif_tracker import MotifTracker
from qratum_chess.gui.panels.quantum import QuantumPanel
from qratum_chess.gui.panels.anti_holographic import AntiHolographicPanel
from qratum_chess.gui.panels.telemetry import TelemetryPanel
from qratum_chess.gui.panels.control import ControlPanel

__all__ = [
    "BoardPanel",
    "TriCortexPanel",
    "SearchTreePanel",
    "MotifTracker",
    "QuantumPanel",
    "AntiHolographicPanel",
    "TelemetryPanel",
    "ControlPanel",
]
