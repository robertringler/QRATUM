"""Base classes for deterministic reality adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from qreal.normalizers import NormalizationChain
from qreal.provenance import ProvenanceRecord, compute_provenance


@dataclass
class AdapterOutput:
    """Normalized output with provenance annotation."""

    normalized: dict[str, object]
    percept: dict[str, object]
    provenance: ProvenanceRecord


@dataclass
class BaseAdapter:
    """Abstract deterministic adapter for external feeds.

    Subclasses implement ``_normalize`` and ``_to_percept`` to translate raw
    external payloads into Q-Stack structured percepts. All transformations are
    deterministic and keyed by tick to avoid wall-clock dependence.
    """

    source: str
    chain: NormalizationChain = field(default_factory=NormalizationChain)
    validators: list[Callable[[dict[str, object]], None]] = field(default_factory=list)

    def process(self, raw: object, tick: int) -> AdapterOutput:
        """Process raw external data deterministically.

        Args:
            raw: Raw payload from an external feed (pre-sandboxed).
            tick: Logical tick used for ordering and provenance.
        """

        normalized = self._normalize(raw, tick)
        normalized = self.chain.apply(normalized)
        for validator in self.validators:
            validator(normalized)
        percept = self._to_percept(normalized)
        provenance = compute_provenance(self.source, normalized, tick)
        return AdapterOutput(normalized=normalized, percept=percept, provenance=provenance)

    # Subclass hooks -----------------------------------------------------------------
    def _normalize(
        self, raw: object, tick: int
    ) -> dict[str, object]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _to_percept(
        self, normalized: dict[str, object]
    ) -> dict[str, object]:  # pragma: no cover - abstract
        raise NotImplementedError

    # Helpers ------------------------------------------------------------------------
    def add_normalizer(self, func: Callable[[dict[str, object]], dict[str, object]]) -> None:
        self.chain.steps.append(func)

    def add_validator(self, func: Callable[[dict[str, object]], None]) -> None:
        self.validators.append(func)

    def serialize(self) -> str:
        """Serialize adapter configuration for deterministic replay."""

        return json.dumps({"source": self.source, "chain": self.chain.describe()}, sort_keys=True)
