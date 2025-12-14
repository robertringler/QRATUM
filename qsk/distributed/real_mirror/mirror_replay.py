"""Replay external feed history deterministically."""

from __future__ import annotations

from typing import Iterable

from qsk.distributed.real_mirror.mirror_state import MirrorState


def replay_events(
    state: MirrorState, events: Iterable[tuple[int, str, dict[str, object]]]
) -> MirrorState:
    for tick, domain, payload in sorted(events, key=lambda e: (e[0], e[1])):
        state.apply(domain, payload, tick)
    return state
