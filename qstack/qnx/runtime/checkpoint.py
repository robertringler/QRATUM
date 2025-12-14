"""Deterministic checkpoint/restore utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import QNXState


def _copy_state(state: QNXState) -> dict[str, Any]:
    return dict(state.data)


@dataclass(frozen=True)
class Checkpoint:
    tick: int
    snapshot: dict[str, Any]


class CheckpointManager:
    def __init__(self):
        self._checkpoints: dict[int, Checkpoint] = {}

    def create(self, tick: int, state: QNXState) -> Checkpoint:
        cp = Checkpoint(tick=tick, snapshot=_copy_state(state))
        self._checkpoints[tick] = cp
        return cp

    def restore(self, tick: int, state: QNXState) -> QNXState:
        if tick not in self._checkpoints:
            raise KeyError("checkpoint missing")
        snapshot = self._checkpoints[tick].snapshot
        return QNXState(dict(snapshot))

    def available(self) -> dict[int, Checkpoint]:
        return dict(self._checkpoints)
