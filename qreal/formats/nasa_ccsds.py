"""Minimal deterministic CCSDS packet decoder."""

from __future__ import annotations

from typing import Dict


def decode(packet: bytes) -> Dict[str, object]:
    if len(packet) < 4:
        raise ValueError("CCSDS packet too short")
    version = packet[0] >> 5
    apid = ((packet[0] & 0x1F) << 8) | packet[1]
    sequence = ((packet[2] & 0x3F) << 8) | packet[3]
    payload = packet[4:]
    return {"version": version, "apid": apid, "sequence": sequence, "payload": payload.hex()}
