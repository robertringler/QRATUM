"""Adapters for QuASIM state integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def from_quasim_state(agent_or_path: str | Path | dict[str, Any]) -> str | np.ndarray:
    """Adapt QuASIM agent state to REVULTRA/QGH input format.

    Converts QuASIM agent trajectories or state snapshots into formats
    suitable for REVULTRA and QGH algorithm processing.

    Parameters
    ----------
    agent_or_path : str, Path, or dict
        QuASIM agent state, path to state file, or state dictionary

    Returns
    -------
    str or np.ndarray
        Adapted state as text (for REVULTRA) or array (for QGH)

    Examples
    --------
    >>> state = {"trajectory": "ABCDEF", "type": "text"}
    >>> result = from_quasim_state(state)
    >>> isinstance(result, str)
    True
    """

    # Handle file path
    if isinstance(agent_or_path, (str, Path)):
        path = Path(agent_or_path)
        if path.exists():
            with open(path) as f:
                agent_or_path = json.load(f)
        else:
            # Treat as raw text
            return str(agent_or_path)

    # Handle dictionary state
    if isinstance(agent_or_path, dict):
        # Check for trajectory field (text-based)
        if "trajectory" in agent_or_path:
            return str(agent_or_path["trajectory"])

        # Check for state vector (array-based)
        if "state_vector" in agent_or_path:
            data = agent_or_path["state_vector"]
            if isinstance(data, list):
                return np.array(data, dtype=np.float64)
            return data

        # Check for tensor data
        if "tensor" in agent_or_path:
            data = agent_or_path["tensor"]
            if isinstance(data, list):
                return np.array(data, dtype=np.float64)
            return data

    # Fallback: convert to string
    return str(agent_or_path)


def to_terc_observable_format(results: dict[str, Any]) -> dict[str, Any]:
    """Convert algorithm results to TERC observable format.

    Parameters
    ----------
    results : dict[str, Any]
        Algorithm results from REVULTRA or QGH

    Returns
    -------
    dict[str, Any]
        TERC-formatted observables with metadata
    """

    # Serialize numpy arrays to lists
    def serialize(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        return obj

    serialized = serialize(results)

    return {
        "observables": serialized,
        "format_version": "1.0",
        "source": "quasim-revultra-qgh",
    }


def extract_trajectory_text(state: dict[str, Any]) -> str:
    """Extract text trajectory from QuASIM state for REVULTRA analysis.

    Parameters
    ----------
    state : dict[str, Any]
        QuASIM state dictionary

    Returns
    -------
    str
        Extracted text trajectory
    """

    if "trajectory" in state:
        return str(state["trajectory"])
    if "sequence" in state:
        return str(state["sequence"])
    if "text" in state:
        return str(state["text"])

    # Fallback: concatenate all string values
    text_parts = []
    for value in state.values():
        if isinstance(value, str):
            text_parts.append(value)

    return "".join(text_parts) if text_parts else "UNKNOWN"


def extract_node_states(state: dict[str, Any]) -> np.ndarray:
    """Extract node states from QuASIM state for QGH analysis.

    Parameters
    ----------
    state : dict[str, Any]
        QuASIM state dictionary

    Returns
    -------
    np.ndarray
        Extracted node states, shape (num_nodes, state_dim)
    """

    if "node_states" in state:
        data = state["node_states"]
        if isinstance(data, list):
            return np.array(data, dtype=np.float64)
        return data

    if "nodes" in state:
        data = state["nodes"]
        if isinstance(data, list):
            return np.array(data, dtype=np.float64)
        return data

    # Fallback: create dummy states
    return np.array([[0.0]], dtype=np.float64)
