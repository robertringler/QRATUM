"""Sensor abstractions for the real-world bridge.

Two concrete implementations are provided:

* :class:`MockSensor` — replays a fixed sequence of :class:`Observation`
  records. Deterministic by construction; raises
  :class:`SensorExhausted` once the sequence is consumed.

* :class:`RealSensorAdapter` — wraps an arbitrary
  ``Callable[[], tuple[float, ...]]`` reader. Every reader failure is
  promoted to :class:`SensorReadError` so the loop can never silently
  swallow a hardware fault.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from qagents.realworld_bridge.types import Observation

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class SensorExhausted(RuntimeError):  # noqa: N818 - name fixed by spec
    """Raised by :class:`MockSensor` when its replay sequence is empty."""


class SensorReadError(RuntimeError):
    """Raised by :class:`RealSensorAdapter` on any reader fault."""


# --------------------------------------------------------------------------- #
# Sensor base
# --------------------------------------------------------------------------- #


class Sensor(ABC):
    """Abstract sensor interface."""

    @abstractmethod
    def read(self) -> Observation: ...

    @abstractmethod
    def source_id(self) -> str: ...


# --------------------------------------------------------------------------- #
# MockSensor
# --------------------------------------------------------------------------- #


class MockSensor(Sensor):
    """Deterministic replay sensor.

    The constructor takes an ordered tuple of :class:`Observation`
    records and an explicit step counter starting at zero.  ``read()``
    returns observations in order; once the sequence is exhausted any
    further call raises :class:`SensorExhausted`.

    Identical construction → identical read sequence; no global state
    is consulted.
    """

    def __init__(
        self,
        observations: tuple[Observation, ...],
        source_id: str,
    ) -> None:
        # Defensive copy into a tuple so external mutation cannot
        # change replay order.
        self._observations: tuple[Observation, ...] = tuple(observations)
        self._source_id: str = str(source_id)
        self._cursor: int = 0

    def read(self) -> Observation:
        if self._cursor >= len(self._observations):
            raise SensorExhausted(
                f"MockSensor[{self._source_id}] exhausted after " f"{len(self._observations)} reads"
            )
        obs = self._observations[self._cursor]
        self._cursor += 1
        return obs

    def source_id(self) -> str:
        return self._source_id

    @property
    def cursor(self) -> int:
        """Number of reads performed so far (test introspection)."""
        return self._cursor

    @property
    def remaining(self) -> int:
        return len(self._observations) - self._cursor


# --------------------------------------------------------------------------- #
# RealSensorAdapter
# --------------------------------------------------------------------------- #


class RealSensorAdapter(Sensor):
    """Adapter that turns an arbitrary reader callable into a Sensor.

    The reader must return a ``tuple[float, ...]`` whose length matches
    the supplied ``expected_length``. Any other outcome — wrong length,
    raised exception — is promoted to :class:`SensorReadError`.

    ``timestamp_provider`` is an explicit callable returning the float
    timestamp for the next observation. This avoids hidden coupling to
    wall-clock time and makes the adapter deterministic given a
    deterministic timestamp source.

    ``missing_mask_provider`` is an explicit callable returning the
    per-dimension missing mask. The mask is **never** silently defaulted
    to all-False; the caller must supply a real provider, even one that
    returns a fixed all-False tuple.
    """

    def __init__(
        self,
        reader: Callable[[], tuple[float, ...]],
        source_id: str,
        expected_length: int,
        timestamp_provider: Callable[[], float],
        missing_mask_provider: Callable[[tuple[float, ...]], tuple[bool, ...]],
    ) -> None:
        if expected_length < 0:
            raise ValueError("expected_length must be non-negative")
        self._reader = reader
        self._source_id = str(source_id)
        self._expected_length = int(expected_length)
        self._timestamp_provider = timestamp_provider
        self._missing_mask_provider = missing_mask_provider

    def read(self) -> Observation:
        try:
            measurements = self._reader()
        except Exception as exc:  # noqa: BLE001 - explicit promotion
            raise SensorReadError(f"reader for sensor {self._source_id!r} raised: {exc!r}") from exc

        if not isinstance(measurements, tuple):
            try:
                measurements = tuple(measurements)
            except TypeError as exc:
                raise SensorReadError(
                    f"reader for sensor {self._source_id!r} returned "
                    f"non-iterable: {measurements!r}"
                ) from exc

        if len(measurements) != self._expected_length:
            raise SensorReadError(
                f"reader for sensor {self._source_id!r} returned "
                f"{len(measurements)} values; expected "
                f"{self._expected_length}"
            )

        try:
            measurements = tuple(float(m) for m in measurements)
        except (TypeError, ValueError) as exc:
            raise SensorReadError(
                f"reader for sensor {self._source_id!r} returned "
                f"non-numeric values: {measurements!r}"
            ) from exc

        mask = self._missing_mask_provider(measurements)
        if len(mask) != len(measurements):
            raise SensorReadError(
                f"missing_mask_provider returned wrong length "
                f"({len(mask)}) for {len(measurements)} measurements"
            )

        return Observation(
            timestamp=float(self._timestamp_provider()),
            source_id=self._source_id,
            measurements=measurements,
            missing_mask=tuple(bool(b) for b in mask),
        )

    def source_id(self) -> str:
        return self._source_id


__all__ = [
    "MockSensor",
    "RealSensorAdapter",
    "Sensor",
    "SensorExhausted",
    "SensorReadError",
]
