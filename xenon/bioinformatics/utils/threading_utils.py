"""Thread-safe execution utilities for XENON Bioinformatics.

Provides:
- Deterministic thread-level seed derivation
- Thread-safe wrappers for engines
- Concurrent execution with ordering guarantees
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple

try:
    from qratum.common.seeding import SeedManager
except ImportError:
    pass


def derive_thread_seed(base_seed: int, thread_id: int) -> int:
    """Derive deterministic seed for thread.

    Args:
        base_seed: Base seed
        thread_id: Thread identifier

    Returns:
        Derived seed for thread
    """

    combined = f"{base_seed}:thread_{thread_id}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big")


class ThreadSafeEngine:
    """Thread-safe wrapper for XENON engines.

    Provides lock-based synchronization for engine operations.
    """

    def __init__(self, engine: Any, base_seed: int = 42):
        """Initialize thread-safe wrapper.

        Args:
            engine: Engine to wrap
            base_seed: Base seed for thread derivation
        """

        self.engine = engine
        self.base_seed = base_seed
        self._lock = threading.Lock()
        self._thread_engines: Dict[int, Any] = {}

    def execute(self, method_name: str, *args, **kwargs) -> Any:
        """Execute engine method in thread-safe manner.

        Args:
            method_name: Name of method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Method result
        """

        with self._lock:
            method = getattr(self.engine, method_name)
            return method(*args, **kwargs)

    def execute_concurrent(
        self,
        method_name: str,
        args_list: List[Tuple],
        max_workers: int = 4,
    ) -> List[Any]:
        """Execute method concurrently with multiple argument sets.

        Args:
            method_name: Name of method to call
            args_list: List of (args, kwargs) tuples
            max_workers: Maximum number of worker threads

        Returns:
            List of results in input order
        """

        results = [None] * len(args_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}

            for idx, (args, kwargs) in enumerate(args_list):
                # Derive thread seed
                thread_seed = derive_thread_seed(self.base_seed, idx)
                kwargs["seed"] = thread_seed

                future = executor.submit(self._execute_with_lock, method_name, args, kwargs)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    def _execute_with_lock(
        self,
        method_name: str,
        args: Tuple,
        kwargs: Dict,
    ) -> Any:
        """Execute with lock."""

        with self._lock:
            method = getattr(self.engine, method_name)
            return method(*args, **kwargs)


def thread_safe(func: Callable) -> Callable:
    """Decorator to make function thread-safe.

    Args:
        func: Function to wrap

    Returns:
        Thread-safe wrapper
    """

    lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return wrapper
