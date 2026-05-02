"""Deterministic interaction utilities between agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    sender: str
    recipient: str
    payload: dict[str, object]
    tick: int


class InteractionBus:
    """Deterministic message bus with total ordering by tick then sender."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def publish(self, message: Message) -> None:
        self._messages.append(message)
        self._messages.sort(key=lambda m: (m.tick, m.sender, m.recipient))

    def drain(self, recipient: str, up_to_tick: int) -> list[Message]:
        delivered: list[Message] = []
        remaining: list[Message] = []
        for msg in self._messages:
            if msg.recipient == recipient and msg.tick <= up_to_tick:
                delivered.append(msg)
            else:
                remaining.append(msg)
        self._messages = remaining
        return delivered
