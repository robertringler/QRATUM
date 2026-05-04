"""Deterministic message passing primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Channel:
    name: str
    queue: deque[tuple[str, object]] = field(default_factory=deque)

    def send(self, sender: str, message: object) -> None:
        self.queue.append((sender, message))

    def recv(self) -> tuple[str, object]:
        if not self.queue:
            raise IndexError("empty channel")
        return self.queue.popleft()

    def pending(self) -> int:
        return len(self.queue)


class Mailbox:
    def __init__(self) -> None:
        self.channels: dict[str, Channel] = {}

    def channel(self, name: str) -> Channel:
        if name not in self.channels:
            self.channels[name] = Channel(name)
        return self.channels[name]

    def send(self, channel: str, sender: str, message: object) -> None:
        self.channel(channel).send(sender, message)

    def recv(self, channel: str) -> tuple[str, object]:
        return self.channel(channel).recv()

    def stats(self) -> dict[str, int]:
        return {name: ch.pending() for name, ch in sorted(self.channels.items())}
