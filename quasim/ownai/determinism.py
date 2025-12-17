"""Determinism utilities for reproducible AI training and inference."""

import hashlib
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int = 1337) -> None:
    """Set global random seed for reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available)

    Parameters
    ----------
    seed : int
        Random seed to use (default: 1337)

    Examples
    --------
    >>> from quasim.ownai.determinism import set_seed
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

    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def hash_array(arr: np.ndarray) -> str:
    """Compute SHA256 hash of numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Array to hash

    Returns
    -------
    str
        Hexadecimal hash string

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> h = hash_array(arr)
    >>> len(h)
    64
    """

    return hashlib.sha256(arr.tobytes()).hexdigest()


def hash_preds(predictions: Any) -> str:
    """Compute hash of model predictions for consensus checking.

    Parameters
    ----------
    predictions : Any
        Model predictions (numpy array, torch tensor, or convertible)

    Returns
    -------
    str
        Hexadecimal hash string

    Examples
    --------
    >>> import numpy as np
    >>> preds = np.array([0.1, 0.9, 0.3])
    >>> h = hash_preds(preds)
    >>> len(h)
    64
    """

    # Convert to numpy if needed
    if hasattr(predictions, "numpy"):
        predictions = predictions.detach().cpu().numpy()
    elif hasattr(predictions, "cpu"):
        predictions = predictions.cpu().numpy()

    # Ensure it's a numpy array
    predictions = np.asarray(predictions)

    return hash_array(predictions)


def verify_determinism(func, *args, seed: int = 1337, n_runs: int = 2, **kwargs) -> bool:
    """Verify that a function produces deterministic outputs.

    Parameters
    ----------
    func : callable
        Function to test
    *args
        Positional arguments for func
    seed : int
        Random seed to use (default: 1337)
    n_runs : int
        Number of runs to compare (default: 2)
    **kwargs
        Keyword arguments for func

    Returns
    -------
    bool
        True if all runs produce identical output hashes

    Examples
    --------
    >>> import numpy as np
    >>> def random_func():
    ...     return np.random.randn(10)
    >>> verify_determinism(random_func, seed=42, n_runs=3)
    True
    """

    hashes = []

    for _ in range(n_runs):
        set_seed(seed)
        result = func(*args, **kwargs)

        # Convert result to hashable format
        if isinstance(result, (list, tuple)):
            result = np.array(result)

        h = hash_preds(result)
        hashes.append(h)

    # All hashes should be identical
    return len(set(hashes)) == 1
