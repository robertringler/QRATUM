"""Decode simple multiline scientific JSONL."""

from __future__ import annotations

import json
from typing import Iterable


def decode(lines: Iterable[str]) -> list[dict]:
    return [json.loads(line) for line in lines]
