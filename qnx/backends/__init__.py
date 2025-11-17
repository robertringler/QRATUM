from __future__ import annotations

from typing import Dict

from ..logging import get_logger
from ..types import SimulationBackend
from .qvr_win import QVRWinBackend
from .quasim_legacy_v120 import QuasimLegacyV120Backend
from .quasim_modern import QuasimModernBackend

logger = get_logger(__name__)


def _safe_init(backend_cls: type[SimulationBackend]) -> SimulationBackend | None:
    try:
        return backend_cls()
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("Skipping backend during initialisation", extra={"backend": backend_cls.__name__, "error": str(exc)})
        return None


def get_backend_registry() -> Dict[str, SimulationBackend]:
    """Initialise all available backends and return them keyed by name."""

    registry: Dict[str, SimulationBackend] = {}
    for backend_cls in (QuasimModernBackend, QuasimLegacyV120Backend, QVRWinBackend):
        backend = _safe_init(backend_cls)
        if backend is not None:
            registry[backend.name] = backend
    return registry

__all__ = ["get_backend_registry"]
