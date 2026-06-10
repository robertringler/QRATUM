"""Tiny CNN for vision classification tasks."""

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.ensemble import RandomForestClassifier

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - optional dependency
    RandomForestClassifier = None  # type: ignore[assignment,misc]
    _HAS_SKLEARN = False

from quasim.ownai.determinism import set_seed


class TinyCNN:
    """Tiny CNN for vision tasks (using simplified sklearn implementation).

    Parameters
    ----------
    n_classes : int
        Number of output classes (default: 10)
    seed : int
        Random seed for determinism (default: 42)
    """

    def __init__(self, n_classes: int = 10, seed: int = 42):
        self.n_classes = n_classes
        self.seed = seed

        set_seed(self.seed)

        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for TinyCNN; install it with "
                "'pip install quasim[ownai]'"
            )

        # Use RandomForest as a simple baseline
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=seed,
            n_jobs=1,
        )

    def _flatten_images(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        """Flatten image tensors to vectors.

        Parameters
        ----------
        images : NDArray
            Image tensor of shape (N, C, H, W)

        Returns
        -------
        NDArray
            Flattened images of shape (N, C*H*W)
        """

        batch_size = images.shape[0]
        return images.reshape(batch_size, -1)

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> "TinyCNN":
        """Train the CNN.

        Parameters
        ----------
        X : NDArray
            Training images of shape (N, C, H, W)
        y : NDArray
            Training labels

        Returns
        -------
        self
        """

        set_seed(self.seed)

        # Flatten images for sklearn
        X_flat = self._flatten_images(X)

        self.model.fit(X_flat, y)
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Make predictions.

        Parameters
        ----------
        X : NDArray
            Input images

        Returns
        -------
        NDArray
            Predicted class labels
        """

        X_flat = self._flatten_images(X)
        return self.model.predict(X_flat)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities.

        Parameters
        ----------
        X : NDArray
            Input images

        Returns
        -------
        NDArray
            Class probabilities
        """

        X_flat = self._flatten_images(X)
        return self.model.predict_proba(X_flat)
