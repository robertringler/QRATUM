"""Grid stability rules."""
from __future__ import annotations



def stability(snapshot: dict[str, float], min_freq: float, max_freq: float) -> list[str]:
    violations: list[str] = []
    freq = float(snapshot.get("frequency", 0.0))
    if freq < min_freq or freq > max_freq:
        violations.append("frequency_out_of_bounds")
    return violations
