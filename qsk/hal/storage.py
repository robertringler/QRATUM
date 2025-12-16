"""Logical block device simulation."""

from __future__ import annotations

from typing import Dict


class BlockDevice:
    def __init__(self, blocks: int = 8, block_size: int = 512) -> None:
        self.block_size = block_size
        self.data: Dict[int, bytes] = dict.fromkeys(range(blocks), b"\x00" * block_size)

    def write(self, index: int, payload: bytes) -> None:
        if len(payload) != self.block_size:
            raise ValueError("invalid block size")
        if index not in self.data:
            raise IndexError(index)
        self.data[index] = payload

    def read(self, index: int) -> bytes:
        if index not in self.data:
            raise IndexError(index)
        return self.data[index]

    def checksum(self) -> int:
        return sum(sum(block) for block in self.data.values())
