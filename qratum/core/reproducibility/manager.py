"""
Reproducibility Manager for QRATUM

Enforces deterministic behavior across Python, NumPy, PyTorch, TensorFlow, and Qiskit.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import os
import random
from typing import Optional

from .global_seed import get_global_seed


class ReproducibilityManager:
    """
    Manages reproducibility across multiple frameworks.

    Ensures deterministic execution for:
    - Python random
    - NumPy random
    - PyTorch (CPU and CUDA)
    - TensorFlow
    - Qiskit quantum simulators
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize reproducibility manager.

        Args:
            seed: Optional seed override. If None, uses global seed.
        """
        self.seed = seed if seed is not None else get_global_seed()
        self._initialized = False

    def setup_deterministic_mode(self) -> None:
        """
        Configure all frameworks for deterministic execution.

        Sets up:
        - Python random state
        - NumPy random state
        - PyTorch deterministic mode
        - TensorFlow deterministic ops
        - Qiskit seed
        """
        if self._initialized:
            return

        # Python random
        random.seed(self.seed)

        # NumPy
        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:
            pass

        # PyTorch
        try:
            import torch

            torch.manual_seed(self.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        # TensorFlow
        try:
            import tensorflow as tf

            tf.random.set_seed(self.seed)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
        except ImportError:
            pass

        # Environment variables for additional determinism
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        self._initialized = True

    def get_qiskit_seed(self) -> int:
        """
        Get seed for Qiskit quantum simulators.

        Returns:
            int: Seed value for Qiskit
        """
        return self.seed

    def verify_determinism(self) -> dict:
        """
        Verify deterministic configuration across frameworks.

        Returns:
            dict: Status of deterministic settings
        """
        status = {
            "seed": self.seed,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "initialized": self._initialized,
        }

        # Check PyTorch
        try:
            import torch

            status["pytorch"] = {
                "deterministic": torch.backends.cudnn.deterministic,
                "benchmark": torch.backends.cudnn.benchmark,
            }
        except ImportError:
            status["pytorch"] = "not_available"

        # Check TensorFlow
        try:
            status["tensorflow"] = {"deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS")}
        except ImportError:
            status["tensorflow"] = "not_available"

        return status
