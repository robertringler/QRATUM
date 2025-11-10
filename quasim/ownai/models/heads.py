"""Task-specific heads for classification, regression, and sequence tasks."""

import numpy as np
from numpy.typing import NDArray


class ClassificationHead:
    """Simple classification head using softmax.

    Attributes
    ----------
    n_classes : int
        Number of output classes
    weights : NDArray | None
        Weight matrix
    bias : NDArray | None
        Bias vector
    """

    def __init__(self, input_dim: int, n_classes: int):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.weights: NDArray[np.float32] | None = None
        self.bias: NDArray[np.float32] | None = None

    def initialize(self, seed: int = 42) -> None:
        """Initialize weights and biases.

        Parameters
        ----------
        seed : int
            Random seed for initialization
        """
        np.random.seed(seed)
        # Xavier initialization
        scale = np.sqrt(2.0 / (self.input_dim + self.n_classes))
        self.weights = np.random.randn(self.input_dim, self.n_classes).astype(np.float32) * scale
        self.bias = np.zeros(self.n_classes, dtype=np.float32)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass.

        Parameters
        ----------
        x : NDArray
            Input features of shape (batch_size, input_dim)

        Returns
        -------
        NDArray
            Logits of shape (batch_size, n_classes)
        """
        if self.weights is None or self.bias is None:
            self.initialize()

        logits = x @ self.weights + self.bias
        return logits

    def predict(self, x: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict class labels.

        Parameters
        ----------
        x : NDArray
            Input features

        Returns
        -------
        NDArray
            Predicted class labels
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=1).astype(np.int64)


class RegressionHead:
    """Simple regression head for continuous outputs.

    Attributes
    ----------
    weights : NDArray | None
        Weight matrix
    bias : float | None
        Bias term
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.weights: NDArray[np.float32] | None = None
        self.bias: float | None = None

    def initialize(self, seed: int = 42) -> None:
        """Initialize weights and bias.

        Parameters
        ----------
        seed : int
            Random seed for initialization
        """
        np.random.seed(seed)
        scale = np.sqrt(2.0 / self.input_dim)
        self.weights = np.random.randn(self.input_dim, 1).astype(np.float32) * scale
        self.bias = 0.0

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass.

        Parameters
        ----------
        x : NDArray
            Input features of shape (batch_size, input_dim)

        Returns
        -------
        NDArray
            Predictions of shape (batch_size,)
        """
        if self.weights is None or self.bias is None:
            self.initialize()

        predictions = (x @ self.weights).squeeze() + self.bias
        return predictions.astype(np.float32)

    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict continuous values.

        Parameters
        ----------
        x : NDArray
            Input features

        Returns
        -------
        NDArray
            Predicted values
        """
        return self.forward(x)
