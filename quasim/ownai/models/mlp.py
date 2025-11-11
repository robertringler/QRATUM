"""Deterministic MLP model for classification and regression."""

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier, MLPRegressor

from quasim.ownai.determinism import set_seed


class DeterministicMLP:
    """Deterministic Multi-Layer Perceptron using sklearn.

    Parameters
    ----------
    task : str
        Task type: 'classification' or 'regression'
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions (default: (128, 64))
    max_iter : int
        Maximum training iterations (default: 200)
    seed : int
        Random seed for determinism (default: 42)
    """

    def __init__(
        self,
        task: str = "classification",
        hidden_dims: tuple[int, ...] = (128, 64),
        max_iter: int = 200,
        seed: int = 42,
    ):
        self.task = task
        self.hidden_dims = hidden_dims
        self.max_iter = max_iter
        self.seed = seed

        set_seed(self.seed)

        if task == "classification":
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_dims,
                max_iter=max_iter,
                random_state=seed,
                early_stopping=False,
                learning_rate_init=0.001,
                solver="adam",
            )
        elif task == "regression":
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_dims,
                max_iter=max_iter,
                random_state=seed,
                early_stopping=False,
                learning_rate_init=0.001,
                solver="adam",
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def fit(self, X: NDArray[np.float32], y: NDArray) -> "DeterministicMLP":
        """Train the MLP.

        Parameters
        ----------
        X : NDArray
            Training features
        y : NDArray
            Training targets

        Returns
        -------
        self
        """
        set_seed(self.seed)
        self.model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray:
        """Make predictions.

        Parameters
        ----------
        X : NDArray
            Input features

        Returns
        -------
        NDArray
            Predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : NDArray
            Input features

        Returns
        -------
        NDArray
            Class probabilities
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)
