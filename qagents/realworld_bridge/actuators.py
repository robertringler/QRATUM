"""Actuator abstractions for the real-world bridge.

* :class:`MockActuator` — append-only log, always succeeds, zero latency.
* :class:`ExternalActuatorAdapter` — wraps a writer callable; promotes
  any writer exception into an ``ActionResult(success=False, ...)``.

Actuators **never** re-validate actions; the safety check is enforced
by :class:`~qagents.realworld_bridge.control_loop.ControlLoop` *before*
``execute()`` is called.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from qagents.mvri.action_space import Action
from qagents.realworld_bridge.types import ActionResult

# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class Actuator(ABC):
    """Abstract actuator interface."""

    @abstractmethod
    def execute(self, action: Action) -> ActionResult: ...

    @abstractmethod
    def source_id(self) -> str: ...


# --------------------------------------------------------------------------- #
# MockActuator
# --------------------------------------------------------------------------- #


class MockActuator(Actuator):
    """In-memory deterministic actuator.

    Records every executed action as an :class:`ActionResult` in an
    internal append-only log.  ``execute()`` always reports success
    with zero latency. ``get_log()`` returns the full history as an
    immutable tuple.
    """

    def __init__(self, source_id: str = "mock") -> None:
        self._source_id: str = str(source_id)
        self._log: tuple[ActionResult, ...] = ()

    def execute(self, action: Action) -> ActionResult:
        result = ActionResult(
            success=True,
            action=action,
            latency_ms=0.0,
            failure_reason=None,
        )
        self._log = (*self._log, result)
        return result

    def source_id(self) -> str:
        return self._source_id

    def get_log(self) -> tuple[ActionResult, ...]:
        return self._log


# --------------------------------------------------------------------------- #
# ExternalActuatorAdapter
# --------------------------------------------------------------------------- #


class ExternalActuatorAdapter(Actuator):
    """Adapter for an external writer callable.

    The writer must return ``True`` on success or ``False`` on failure.
    Exceptions raised by the writer are caught and reported as
    ``ActionResult(success=False, failure_reason=str(e))``.

    ``latency_ms`` is supplied at construction time — never measured at
    runtime — so the adapter remains deterministic.
    """

    def __init__(
        self,
        writer: Callable[[Action], bool],
        source_id: str,
        latency_ms: float = 0.0,
    ) -> None:
        if float(latency_ms) < 0.0:
            raise ValueError("latency_ms must be non-negative")
        self._writer = writer
        self._source_id = str(source_id)
        self._latency_ms = float(latency_ms)

    def execute(self, action: Action) -> ActionResult:
        try:
            ok = bool(self._writer(action))
        except Exception as exc:  # noqa: BLE001 - explicit promotion
            return ActionResult(
                success=False,
                action=action,
                latency_ms=self._latency_ms,
                failure_reason=str(exc),
            )
        if not ok:
            return ActionResult(
                success=False,
                action=action,
                latency_ms=self._latency_ms,
                failure_reason="writer returned False",
            )
        return ActionResult(
            success=True,
            action=action,
            latency_ms=self._latency_ms,
            failure_reason=None,
        )

    def source_id(self) -> str:
        return self._source_id


__all__ = ["Actuator", "ExternalActuatorAdapter", "MockActuator"]
