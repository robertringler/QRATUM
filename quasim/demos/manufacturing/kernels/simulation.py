"""Simulation kernel for manufacturing."""

from typing import Any, Dict, Optional

import numpy as np

from quasim.viz.run_capture import RunCapture, create_dummy_frame


def run_simulation(
    scenario: Dict[str, Any],
    steps: int = 200,
    seed: int = 42,
    capture: Optional[RunCapture] = None,
) -> Dict[str, Any]:
    """Run manufacturing simulation."""
    rng = np.random.RandomState(seed)

    state = {"time": 0.0, "value": 0.0}
    trace = []
    values = []

    for step in range(steps):
        state["time"] += 1.0
        state["value"] += rng.randn() * 0.1
        values.append(state["value"])

        trace.append({"step": step, "time": state["time"], "value": state["value"]})

        if capture:
            frame = create_dummy_frame(width=640, height=480, step=step, pattern="gradient")
            capture.record({"frame": frame, "step": step, "value": state["value"]})

    np.array(values)
    np.linspace(0, 10, steps)

    results = {"trace": trace, "final_value": state["value"]}

    # Add KPIs
    results["MTBF"] = float(np.abs(rng.randn() * 10) + 0)
    results["downtime_pct"] = float(np.abs(rng.randn() * 10) + 5)
    results["throughput_units_hr"] = float(np.abs(rng.randn() * 10) + 10)
    results["false_alarm_rate"] = float(np.abs(rng.randn() * 10) + 15)

    return results
