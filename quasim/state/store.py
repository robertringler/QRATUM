"""State storage utilities for QuASIM.

This module provides interfaces for storing and retrieving simulation state
using HDF5 and Zarr backends.
"""

from typing import Any, Dict

import numpy as np

from quasim.io.hdf5 import read_snapshot, write_snapshot


def save_state(path: str, state: Dict[str, Any]) -> None:
    """Save simulation state to storage.

    Parameters
    ----------
    path : str
        Output file path
    state : Dict[str, Any]
        State dictionary with 'meta' and array data

    Examples
    --------
    >>> state = {"meta": {"version": "1.0"}, "data": np.array([1, 2, 3])}
    >>> save_state("state.hdf5", state)
    """

    meta = state.get("meta", {})
    arrays = {k: v for k, v in state.items() if k != "meta" and isinstance(v, np.ndarray)}
    write_snapshot(path, meta, arrays)


def load_state(path: str) -> Dict[str, Any]:
    """Load simulation state from storage.

    Parameters
    ----------
    path : str
        Input file path

    Returns
    -------
    Dict[str, Any]
        State dictionary

    Examples
    --------
    >>> state = load_state("state.hdf5")
    >>> print(state['meta']['version'])
    1.0
    """

    return read_snapshot(path)
