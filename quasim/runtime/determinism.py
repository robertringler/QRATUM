"""Determinism utilities for reproducible simulations.

This module provides seed management and deterministic replay capabilities
for QuASIM simulations, ensuring reproducibility across runs.
"""

import hashlib
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global random seed for reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    - JAX (if available)

    Parameters
    ----------
    seed : int
        Random seed to use (default: 42)

    Examples
    --------
    >>> from quasim.runtime.determinism import set_seed
    >>> set_seed(42)
    >>> import numpy as np
    >>> np.random.randint(0, 100)
    51
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed if available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Enable deterministic operations
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Set JAX seed if available
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ImportError:
        pass

    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def hash_input(data: Any) -> str:
    """Compute SHA256 hash of input data for replay verification.

    Parameters
    ----------
    data : Any
        Data to hash (string, bytes, numpy array, etc.)

    Returns
    -------
    str
        Hexadecimal hash string

    Examples
    --------
    >>> hash_input("test query")
    'ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb'
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    elif not isinstance(data, bytes):
        data = str(data).encode("utf-8")

    return hashlib.sha256(data).hexdigest()
