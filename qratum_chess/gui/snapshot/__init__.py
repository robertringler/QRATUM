"""Snapshot Capture System for QRATUM-Chess.

Automatic high-fidelity snapshot capture of the QRATUM Chess system ("Bob")
during active operation. Captures visual and data records of all system
features including GUI panels, telemetry, cortex outputs, quantum layers,
and benchmarking modes.
"""

from __future__ import annotations

__all__ = [
    "SnapshotCapture",
    "SnapshotManager",
    "SnapshotEvent",
    "SnapshotConfig",
]

from qratum_chess.gui.snapshot.capture import SnapshotCapture, SnapshotConfig
from qratum_chess.gui.snapshot.events import SnapshotEvent
from qratum_chess.gui.snapshot.manager import SnapshotManager
