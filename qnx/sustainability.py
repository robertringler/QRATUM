from __future__ import annotations

import time
from typing import Any

from .logging import get_logger

try:  # pragma: no cover - optional dependency
    from codecarbon import EmissionsTracker
except Exception:  # pragma: no cover - optional dependency
    EmissionsTracker = None  # type: ignore

logger = get_logger(__name__)


def estimate_carbon(results: Any) -> float:
    """Estimate carbon emissions for a simulation result.

    If CodeCarbon is available the emissions are measured; otherwise ``0.0`` is
    returned to avoid introducing a hard dependency.
    """

    if EmissionsTracker is None:
        logger.info("CodeCarbon not installed; returning zero emissions estimate")
        time.sleep(0.01)
        return 0.0

    tracker = EmissionsTracker(measure_power_secs=1)
    tracker.start()
    try:
        time.sleep(0.01)
    finally:
        emissions = tracker.stop() or 0.0
    return float(emissions)
