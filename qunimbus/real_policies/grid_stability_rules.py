"""Grid stability rules."""

from __future__ import annotations

from typing import Dict, List


def stability(snapshot: Dict[str, float], min_freq: float, max_freq: float) -> List[str]:
    violations: List[str] = []
    freq = float(snapshot.get("frequency", 0.0))
    if freq < min_freq or freq > max_freq:
        violations.append("frequency_out_of_bounds")
    return violations
