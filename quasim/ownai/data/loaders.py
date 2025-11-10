"""Data loaders for various datasets (tabular, text, vision, time-series)."""

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

from quasim.ownai.configs import DEFAULT_CONFIG


def get_cache_dir() -> Path:
    """Get the cache directory for datasets."""
    return DEFAULT_CONFIG.cache_dir


def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def load_tabular(
    name: Literal["adult", "wine", "higgs-mini"],
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tabular dataset.

    Parameters
    ----------
    name : str
        Dataset name: 'adult', 'wine', or 'higgs-mini'
    cache_dir : Path, optional
        Cache directory (default: from config)

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels

    Examples
    --------
    >>> X, y = load_tabular("wine")
    >>> X.shape[0] == y.shape[0]
    True
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    cache_dir.mkdir(parents=True, exist_ok=True)

    if name == "wine":
        # Generate synthetic wine-like dataset
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=13,
            n_informative=10,
            n_redundant=3,
            n_classes=3,
            random_state=42,
        )
        return X.astype(np.float32), y.astype(np.int64)

    elif name == "adult":
        # Generate synthetic adult-like dataset
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=10000,
            n_features=14,
            n_informative=10,
            n_redundant=4,
            n_classes=2,
            random_state=42,
        )
        return X.astype(np.float32), y.astype(np.int64)

    elif name == "higgs-mini":
        # Generate synthetic higgs-like dataset
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=50000,
            n_features=28,
            n_informative=20,
            n_redundant=8,
            n_classes=2,
            random_state=42,
        )
        return X.astype(np.float32), y.astype(np.int64)

    else:
        raise ValueError(f"Unknown tabular dataset: {name}")


def load_text(
    name: Literal["imdb-mini", "agnews-mini"],
    cache_dir: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load text classification dataset.

    Parameters
    ----------
    name : str
        Dataset name: 'imdb-mini' or 'agnews-mini'
    cache_dir : Path, optional
        Cache directory (default: from config)

    Returns
    -------
    texts : list[str]
        List of text samples
    y : np.ndarray
        Target labels

    Examples
    --------
    >>> texts, y = load_text("imdb-mini")
    >>> len(texts) == len(y)
    True
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    cache_dir.mkdir(parents=True, exist_ok=True)

    if name == "imdb-mini":
        # Generate synthetic text data
        texts = []
        labels = []

        positive_words = ["excellent", "amazing", "wonderful", "great", "fantastic", "love"]
        negative_words = ["terrible", "awful", "horrible", "bad", "worst", "hate"]

        np.random.seed(42)
        for i in range(1000):
            label = i % 2
            if label == 1:
                words = np.random.choice(positive_words, size=np.random.randint(10, 20))
            else:
                words = np.random.choice(negative_words, size=np.random.randint(10, 20))

            text = " ".join(words) + " movie."
            texts.append(text)
            labels.append(label)

        return texts, np.array(labels, dtype=np.int64)

    elif name == "agnews-mini":
        # Generate synthetic news data
        texts = []
        labels = []

        topics = [
            ["sports", "game", "team", "player", "score", "win"],
            ["business", "market", "stock", "company", "profit", "trade"],
            ["technology", "computer", "software", "internet", "digital", "tech"],
            ["politics", "government", "election", "policy", "president", "vote"],
        ]

        np.random.seed(42)
        for i in range(1000):
            label = i % 4
            words = np.random.choice(topics[label], size=np.random.randint(15, 25))
            text = " ".join(words) + " news."
            texts.append(text)
            labels.append(label)

        return texts, np.array(labels, dtype=np.int64)

    else:
        raise ValueError(f"Unknown text dataset: {name}")


def load_vision(
    name: Literal["cifar10-subset", "mnist-1k"],
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load vision dataset.

    Parameters
    ----------
    name : str
        Dataset name: 'cifar10-subset' or 'mnist-1k'
    cache_dir : Path, optional
        Cache directory (default: from config)

    Returns
    -------
    images : np.ndarray
        Image tensor of shape (N, C, H, W)
    y : np.ndarray
        Target labels

    Examples
    --------
    >>> images, y = load_vision("mnist-1k")
    >>> images.shape[0] == y.shape[0]
    True
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    cache_dir.mkdir(parents=True, exist_ok=True)

    if name == "cifar10-subset":
        # Generate synthetic CIFAR-10-like data
        np.random.seed(42)
        n_samples = 1000
        images = np.random.rand(n_samples, 3, 32, 32).astype(np.float32)
        labels = np.random.randint(0, 10, size=n_samples, dtype=np.int64)
        return images, labels

    elif name == "mnist-1k":
        # Generate synthetic MNIST-like data
        np.random.seed(42)
        n_samples = 1000
        images = np.random.rand(n_samples, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, size=n_samples, dtype=np.int64)
        return images, labels

    else:
        raise ValueError(f"Unknown vision dataset: {name}")


def load_timeseries(
    name: Literal["etth1-mini", "synthetic-arma"],
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load time series dataset.

    Parameters
    ----------
    name : str
        Dataset name: 'etth1-mini' or 'synthetic-arma'
    cache_dir : Path, optional
        Cache directory (default: from config)

    Returns
    -------
    sequences : np.ndarray
        Time series sequences of shape (N, seq_len, n_features)
    y : np.ndarray
        Target values

    Examples
    --------
    >>> seqs, y = load_timeseries("synthetic-arma")
    >>> seqs.shape[0] == y.shape[0]
    True
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    cache_dir.mkdir(parents=True, exist_ok=True)

    if name == "etth1-mini":
        # Generate synthetic time series data
        np.random.seed(42)
        n_samples = 1000
        seq_len = 96
        n_features = 7

        sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        # Target is next value
        targets = np.random.randn(n_samples).astype(np.float32)

        return sequences, targets

    elif name == "synthetic-arma":
        # Generate ARMA-like synthetic data
        np.random.seed(42)
        n_samples = 1000
        seq_len = 50

        sequences = []
        targets = []

        for _ in range(n_samples):
            # Simple AR(1) process
            x = np.zeros(seq_len + 1)
            for t in range(1, seq_len + 1):
                x[t] = 0.7 * x[t - 1] + np.random.randn() * 0.3

            sequences.append(x[:seq_len].reshape(-1, 1))
            targets.append(x[seq_len])

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return sequences, targets

    else:
        raise ValueError(f"Unknown time series dataset: {name}")
