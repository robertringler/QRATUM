"""Data preprocessing utilities."""

import numpy as np
from numpy.typing import NDArray


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance.

    Attributes
    ----------
    mean_ : NDArray | None
        Mean of training data
    std_ : NDArray | None
        Standard deviation of training data
    """

    def __init__(self):
        self.mean_: NDArray[np.float32] | None = None
        self.std_: NDArray[np.float32] | None = None

    def fit(self, X: NDArray[np.float32]) -> "StandardScaler":
        """Compute mean and std for later scaling.

        Parameters
        ----------
        X : NDArray
            Training data

        Returns
        -------
        self
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Scale data using computed mean and std.

        Parameters
        ----------
        X : NDArray
            Data to scale

        Returns
        -------
        NDArray
            Scaled data
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit and transform in one step.

        Parameters
        ----------
        X : NDArray
            Data to fit and transform

        Returns
        -------
        NDArray
            Scaled data
        """
        return self.fit(X).transform(X)


class SimpleTokenizer:
    """Simple fixed-vocabulary tokenizer for text.

    Attributes
    ----------
    vocab : dict[str, int]
        Vocabulary mapping words to IDs
    max_length : int
        Maximum sequence length
    """

    def __init__(self, max_length: int = 128):
        self.vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.max_length = max_length
        self.next_id = 2

    def fit(self, texts: list[str]) -> "SimpleTokenizer":
        """Build vocabulary from texts.

        Parameters
        ----------
        texts : list[str]
            Training texts

        Returns
        -------
        self
        """
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = self.next_id
                    self.next_id += 1
        return self

    def transform(self, texts: list[str]) -> NDArray[np.int64]:
        """Convert texts to token IDs.

        Parameters
        ----------
        texts : list[str]
            Texts to tokenize

        Returns
        -------
        NDArray
            Token IDs of shape (len(texts), max_length)
        """
        result = np.zeros((len(texts), self.max_length), dtype=np.int64)

        for i, text in enumerate(texts):
            words = text.lower().split()[: self.max_length]
            for j, word in enumerate(words):
                result[i, j] = self.vocab.get(word, 1)  # 1 is <UNK>

        return result

    def fit_transform(self, texts: list[str]) -> NDArray[np.int64]:
        """Fit vocabulary and transform in one step.

        Parameters
        ----------
        texts : list[str]
            Texts to fit and transform

        Returns
        -------
        NDArray
            Token IDs
        """
        return self.fit(texts).transform(texts)


def normalize_images(images: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize images to [0, 1] range.

    Parameters
    ----------
    images : NDArray
        Image tensor

    Returns
    -------
    NDArray
        Normalized images
    """
    images = images.astype(np.float32)
    min_val = images.min()
    max_val = images.max()

    if max_val - min_val < 1e-10:
        return images

    return (images - min_val) / (max_val - min_val)


def create_sliding_windows(
    data: NDArray[np.float32], window_size: int, stride: int = 1
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Create sliding windows from time series data.

    Parameters
    ----------
    data : NDArray
        Time series data of shape (n_timesteps,) or (n_timesteps, n_features)
    window_size : int
        Size of each window
    stride : int
        Stride between windows (default: 1)

    Returns
    -------
    windows : NDArray
        Windows of shape (n_windows, window_size, n_features)
    targets : NDArray
        Target values (next value after each window)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_timesteps, n_features = data.shape
    windows = []
    targets = []

    for i in range(0, n_timesteps - window_size, stride):
        window = data[i : i + window_size]
        target = data[i + window_size, 0]  # Predict first feature

        windows.append(window)
        targets.append(target)

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)
