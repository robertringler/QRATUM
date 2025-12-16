"""Deterministic network model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Packet:
    source: str
    destination: str
    payload: str
    hop: int = 0


class NetworkLink:
    def __init__(self, latency: int = 1) -> None:
        self.latency = latency
        self.queue: List[Tuple[int, Packet]] = []

    def send(self, packet: Packet) -> None:
        self.queue.append((self.latency, packet))

    def tick(self) -> List[Packet]:
        delivered: List[Packet] = []
        next_queue: List[Tuple[int, Packet]] = []
        for remaining, packet in self.queue:
            remaining -= 1
            packet.hop += 1
            if remaining <= 0:
                delivered.append(packet)
            else:
                next_queue.append((remaining, packet))
        self.queue = next_queue
        return delivered

    def pending(self) -> int:
        return len(self.queue)


def route(network: Dict[Tuple[str, str], NetworkLink], packet: Packet) -> List[Packet]:
    key = (packet.source, packet.destination)
    if key not in network:
        raise KeyError("route not found")
    link = network[key]
    link.send(packet)
    return link.tick()
