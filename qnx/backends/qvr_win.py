from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import MutableMapping

from ..types import SimulationConfig


class QVRWinBackend:
    """Windows-only backend bridging to the QVR CLI."""

    name = "qvr_win"

    def run(self, config: SimulationConfig) -> MutableMapping[str, object]:
        if _is_windows() is False:  # pragma: no cover - platform-specific
            raise RuntimeError("QVRWinBackend is only usable on Windows hosts.")

        executable = os.environ.get("QVR_EXECUTABLE", "qvr.exe")
        if not Path(executable).exists():
            raise FileNotFoundError(f"QVR executable not found at '{executable}'")

        payload = {
            "scenario_id": config.scenario_id,
            "timesteps": config.timesteps,
            "seed": config.seed if config.seed is not None else 0,
        }

        proc = subprocess.run(
            [executable],
            input=json.dumps(payload).encode(),
            capture_output=True,
            check=False,
        )

        stdout = proc.stdout.decode(errors="replace")
        stderr = proc.stderr.decode(errors="replace")

        try:
            parsed = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            parsed = {"raw_stdout": stdout}

        return {
            "engine": self.name,
            "scenario_id": config.scenario_id,
            "timesteps": config.timesteps,
            "seed": payload["seed"],
            "payload": parsed,
            "stderr": stderr,
            "returncode": proc.returncode,
        }

    def validate(self, config: SimulationConfig) -> bool:
        if _is_windows() is False:  # pragma: no cover - platform-specific
            return False
        executable = os.environ.get("QVR_EXECUTABLE", "qvr.exe")
        return Path(executable).exists()


__all__ = ["QVRWinBackend"]


def _is_windows() -> bool:
    """Return True when running on a Windows host."""

    return os.name == "nt"
