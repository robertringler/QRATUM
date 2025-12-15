"""Decode simple multiline scientific JSONL."""

from __future__ import annotations

import json
from typing import Iterable, List


def decode(lines: Iterable[str]) -> List[dict]:
    return [json.loads(line) for line in lines]
