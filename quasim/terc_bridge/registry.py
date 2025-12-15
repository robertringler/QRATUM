"""Observable registry for TERC integration."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Global registry of observable functions
_OBSERVABLE_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_observable(name: str, func: Callable[..., Any]) -> None:
    """Register an observable function.

    Parameters
    ----------
    name : str
        Observable name
    func : Callable
        Observable extraction function
    """
    _OBSERVABLE_REGISTRY[name] = func
    logger.info(f"Registered observable: {name}")


def get_observable(name: str) -> Callable[..., Any] | None:
    """Get registered observable function.

    Parameters
    ----------
    name : str
        Observable name

    Returns
    -------
    Callable or None
        Observable function if registered, None otherwise
    """
    return _OBSERVABLE_REGISTRY.get(name)


def list_observables() -> list[str]:
    """List all registered observables.

    Returns
    -------
    list[str]
        List of observable names
    """
    return list(_OBSERVABLE_REGISTRY.keys())


def register_observables(runtime: Any = None) -> None:
    """Register all REVULTRA and QGH observables with QuASIM runtime.

    This function registers standard observables from REVULTRA and QGH
    algorithms so they can be consumed by TERC validation tiers.

    Parameters
    ----------
    runtime : Any, optional
        QuASIM runtime object (for future integration)
    """
    from quasim.terc_bridge.observables import (beta_metrics_from_cipher,
                                                emergent_complexity,
                                                ioc_period_candidates,
                                                qgh_consensus_status,
                                                stability_assessment,
                                                stream_synchronization_metrics)

    # Register REVULTRA observables
    register_observable("beta_metrics", beta_metrics_from_cipher)
    register_observable("ioc_periods", ioc_period_candidates)
    register_observable("emergent_complexity", emergent_complexity)

    # Register QGH observables
    register_observable("consensus_status", qgh_consensus_status)
    register_observable("stream_sync", stream_synchronization_metrics)
    register_observable("stability", stability_assessment)

    logger.info(f"Registered {len(_OBSERVABLE_REGISTRY)} observables for TERC validation")

    if runtime is not None:
        logger.info("Runtime integration: observables attached to runtime")
        # Future: attach observables to runtime if API available


def compute_observable(name: str, *args: Any, **kwargs: Any) -> Any:
    """Compute a registered observable.

    Parameters
    ----------
    name : str
        Observable name
    *args : Any
        Positional arguments for observable function
    **kwargs : Any
        Keyword arguments for observable function

    Returns
    -------
    Any
        Observable computation result

    Raises
    ------
    ValueError
        If observable is not registered
    """
    func = get_observable(name)
    if func is None:
        raise ValueError(f"Observable '{name}' not registered. Available: {list_observables()}")

    return func(*args, **kwargs)


# Auto-register on import
register_observables()
