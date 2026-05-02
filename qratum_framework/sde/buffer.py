"""buffer.py — the rolling window that feeds the SDE scorer.

A :class:`RollingWindowBuffer` wraps a bounded :class:`collections.deque`
and emits a frozen :class:`WindowSnapshot` once it has accumulated at
least ``MIN_WINDOW`` tokens.  Below the minimum it returns ``None`` so
that the scorer does not see an under-populated window.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from qratum_framework.sde.tokens import Token, WindowSnapshot


class RollingWindowBuffer:
    """Bounded deque-backed window with a min-population gate.

    Parameters
    ----------
    depth:
        Maximum number of tokens retained.  Older tokens are dropped
        FIFO once the buffer is full.
    min_window:
        Minimum population before :meth:`push` returns a snapshot;
        below this, ``push`` returns ``None``.

    The buffer carries a monotonic ``tick`` counter that is incremented
    on every successful ``push`` and stamped onto each emitted
    snapshot.  Two runs with identical token sequences therefore emit
    snapshots with identical ``tick`` values.
    """

    def __init__(self, depth: int = 256, min_window: int = 32) -> None:
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if min_window <= 0:
            raise ValueError(f"min_window must be positive, got {min_window}")
        if min_window > depth:
            raise ValueError(
                f"min_window ({min_window}) must not exceed depth ({depth})"
            )
        self._depth = int(depth)
        self._min_window = int(min_window)
        self._buf: Deque[Token] = deque(maxlen=self._depth)
        self._tick: int = 0

    # --- properties -----------------------------------------------------

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def min_window(self) -> int:
        return self._min_window

    @property
    def tick(self) -> int:
        return self._tick

    def __len__(self) -> int:
        return len(self._buf)

    # --- mutation -------------------------------------------------------

    def push(self, token: Token) -> Optional[WindowSnapshot]:
        """Append ``token`` and return a frozen snapshot when ready.

        Returns ``None`` while the buffer is below ``min_window``;
        thereafter returns a fresh :class:`WindowSnapshot` on every
        push.  The returned snapshot is independent of the buffer's
        internal storage and is safe to retain across subsequent
        pushes.
        """
        self._buf.append(token)
        self._tick += 1
        if len(self._buf) < self._min_window:
            return None
        return WindowSnapshot(tokens=tuple(self._buf), tick=self._tick)

    def snapshot(self) -> Optional[WindowSnapshot]:
        """Return a snapshot without ingesting a new token, or ``None``
        if the buffer has not yet reached ``min_window``."""
        if len(self._buf) < self._min_window:
            return None
        return WindowSnapshot(tokens=tuple(self._buf), tick=self._tick)


__all__ = ["RollingWindowBuffer"]
