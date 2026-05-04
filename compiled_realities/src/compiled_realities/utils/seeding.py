"""Seeding utilities for reproducible simulations."""

from typing import List

import numpy as np


def generate_seed_sequence(base_seed: int, n_seeds: int) -> List[int]:
    """
    Generate a sequence of seeds from a base seed.

    Args:
        base_seed: Base random seed
        n_seeds: Number of seeds to generate

    Returns:
        List of seeds
    """
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(n_seeds)]


def get_rng(seed: int) -> np.random.RandomState:
    """
    Get a random number generator with the given seed.

    Args:
        seed: Random seed

    Returns:
        Random number generator
    """
    return np.random.RandomState(seed)
