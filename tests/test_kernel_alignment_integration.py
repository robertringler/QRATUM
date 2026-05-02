from __future__ import annotations

import pytest

from qstack.config import QStackConfig
from qstack.events import EventBus, EventType
from qstack.kernel import QStackKernel
from qstack.system import QStackSystem
from qstack.telemetry import Telemetry


def test_kernel_emits_alignment_event_on_fatal_violation():
    config = QStackConfig()
    config.qnx.timesteps = 0
    event_bus = EventBus(timestamp_seed="test-seed")
    telemetry = Telemetry()
    system = QStackSystem(config)
    kernel = QStackKernel(config=config, system=system, event_bus=event_bus, telemetry=telemetry)

    with pytest.raises(ValueError):
        kernel.run_qnx_cycles(steps=1)

    assert any(event.type == EventType.ALIGNMENT_PRE_CHECK_FAILED for event in event_bus.events)
