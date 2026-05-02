"""Real-world control bridge over the MVRI substrate.

This package exposes a typed, safety-gated, sensor → estimator →
controller → actuator pipeline. The pipeline is closed by
:class:`~qagents.realworld_bridge.control_loop.ControlLoop` whose
:meth:`~qagents.realworld_bridge.control_loop.ControlLoop.step` runs the
canonical 9-step sequence with no reordering and no shortcuts.
"""

from qagents.realworld_bridge.actuators import (Actuator,
                                                ExternalActuatorAdapter,
                                                MockActuator)
from qagents.realworld_bridge.control_loop import ControlLoop
from qagents.realworld_bridge.observer import (ObserverStateInvalid,
                                               StateObserver)
from qagents.realworld_bridge.safety_gate import (SafetyConfig, check_mag,
                                                  check_mvr, check_roc,
                                                  check_tgt, validate_action)
from qagents.realworld_bridge.sensors import (MockSensor, RealSensorAdapter,
                                              Sensor, SensorExhausted,
                                              SensorReadError)
from qagents.realworld_bridge.types import (ActionResult,
                                            ActionValidationResult,
                                            ControlRunResult, LoopTrace,
                                            Observation)
from qagents.realworld_bridge.world_model_sync import (DEFAULT_MAX_CORRECTION,
                                                       drift_magnitude,
                                                       sync_model)

__all__ = [
    "DEFAULT_MAX_CORRECTION",
    "ActionResult",
    "ActionValidationResult",
    "Actuator",
    "ControlLoop",
    "ControlRunResult",
    "ExternalActuatorAdapter",
    "LoopTrace",
    "MockActuator",
    "MockSensor",
    "Observation",
    "ObserverStateInvalid",
    "RealSensorAdapter",
    "SafetyConfig",
    "Sensor",
    "SensorExhausted",
    "SensorReadError",
    "StateObserver",
    "check_mag",
    "check_mvr",
    "check_roc",
    "check_tgt",
    "drift_magnitude",
    "sync_model",
    "validate_action",
]
