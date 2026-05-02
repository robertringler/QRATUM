"""REVULTRA feature extraction for symbolic latents.

This module extracts REVULTRA features (IoC tensor, spectral autocorrelation,
entropy motifs) and converts them into symbolic latent representations that
can be fused with learned embeddings in the Symbolic-Latent Transformer.
"""

import numpy as np
from numpy.typing import NDArray

from quasim.revultra.algorithms import REVULTRAAlgorithms
from quasim.revultra.metrics import compute_entropy, normalize_array


def extract_ioc_features(text: str, max_period: int = 20) -> NDArray[np.float32]:
    """Extract Index of Coincidence tensor as features.

    Parameters
    ----------
    text : str
        Input text
    max_period : int
        Maximum period to analyze (default: 20)

    Returns
    -------
    NDArray
        IoC tensor features of shape (max_period,)
    """

    rev = REVULTRAAlgorithms()
    ioc_tensor = rev.index_of_coincidence_tensor(text, max_period=max_period)

    # Normalize to [0, 1]
    ioc_features = normalize_array(ioc_tensor, method="minmax")

    return ioc_features.astype(np.float32)


def extract_spectral_features(text: str, max_lag: int = 30) -> NDArray[np.float32]:
    """Extract spectral autocorrelation features.

    Parameters
    ----------
    text : str
        Input text
    max_lag : int
        Maximum lag for autocorrelation (default: 30)

    Returns
    -------
    NDArray
        Spectral autocorrelation features of shape (max_lag,)
    """

    rev = REVULTRAAlgorithms()
    autocorr = rev.spectral_autocorrelation(text, max_lag=max_lag)

    # Normalize to [0, 1]
    spectral_features = normalize_array(autocorr, method="minmax")

    return spectral_features.astype(np.float32)


def extract_entropy_motifs(text: str, window_size: int = 10) -> NDArray[np.float32]:
    """Extract entropy motifs from text using sliding windows.

    Parameters
    ----------
    text : str
        Input text
    window_size : int
        Size of sliding window (default: 10)

    Returns
    -------
    NDArray
        Entropy motif features
    """

    if len(text) < window_size:
        # Pad text if too short
        text = text * (window_size // len(text) + 1)

    entropies = []
    for i in range(0, len(text) - window_size + 1, max(1, window_size // 2)):
        window = text[i : i + window_size]

        # Compute character frequency distribution
        chars = list(window.lower())
        unique_chars = set(chars)

        if len(unique_chars) == 0:
            entropies.append(0.0)
            continue

        probs = np.array([chars.count(c) / len(chars) for c in unique_chars])
        entropy = compute_entropy(probs)
        entropies.append(entropy)

    # Pad or truncate to fixed size
    target_size = 20
    if len(entropies) < target_size:
        entropies.extend([0.0] * (target_size - len(entropies)))
    else:
        entropies = entropies[:target_size]

    entropy_features = np.array(entropies, dtype=np.float32)

    # Normalize
    entropy_features = normalize_array(entropy_features, method="minmax")

    return entropy_features


def extract_symbolic_latents(
    text: str,
    ioc_periods: int = 20,
    spectral_lags: int = 30,
    entropy_window: int = 10,
) -> NDArray[np.float32]:
    """Extract all REVULTRA symbolic latent features from text.

    This combines IoC tensor, spectral autocorrelation, and entropy motifs
    into a single feature vector that represents symbolic properties of the input.

    Parameters
    ----------
    text : str
        Input text
    ioc_periods : int
        Number of IoC periods to extract (default: 20)
    spectral_lags : int
        Number of spectral lags to extract (default: 30)
    entropy_window : int
        Window size for entropy motifs (default: 10)

    Returns
    -------
    NDArray
        Symbolic latent features of shape (ioc_periods + spectral_lags + 20,)

    Examples
    --------
    >>> latents = extract_symbolic_latents("HELLO WORLD" * 10)
    >>> latents.shape
    (70,)
    """

    ioc_feats = extract_ioc_features(text, max_period=ioc_periods)
    spectral_feats = extract_spectral_features(text, max_lag=spectral_lags)
    entropy_feats = extract_entropy_motifs(text, window_size=entropy_window)

    # Concatenate all features
    symbolic_latents = np.concatenate([ioc_feats, spectral_feats, entropy_feats])

    return symbolic_latents.astype(np.float32)


def extract_symbolic_latents_batch(
    texts: list[str],
    ioc_periods: int = 20,
    spectral_lags: int = 30,
    entropy_window: int = 10,
) -> NDArray[np.float32]:
    """Extract symbolic latents for a batch of texts.

    Parameters
    ----------
    texts : list[str]
        List of input texts
    ioc_periods : int
        Number of IoC periods to extract (default: 20)
    spectral_lags : int
        Number of spectral lags to extract (default: 30)
    entropy_window : int
        Window size for entropy motifs (default: 10)

    Returns
    -------
    NDArray
        Symbolic latents of shape (len(texts), feature_dim)
    """

    latents = []
    for text in texts:
        latent = extract_symbolic_latents(text, ioc_periods, spectral_lags, entropy_window)
        latents.append(latent)

    return np.array(latents, dtype=np.float32)


def extract_symbolic_latents_numeric(
    X: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Extract symbolic latents from numeric/tabular data.

    For numeric data, we compute statistical features that serve as
    symbolic representations.

    Parameters
    ----------
    X : NDArray
        Numeric feature matrix of shape (n_samples, n_features)

    Returns
    -------
    NDArray
        Symbolic latents of shape (n_samples, latent_dim)
    """

    n_samples, n_features = X.shape

    latents = []
    for i in range(n_samples):
        row = X[i]

        # Compute statistical features
        mean_val = np.mean(row)
        std_val = np.std(row)
        min_val = np.min(row)
        max_val = np.max(row)
        median_val = np.median(row)

        # Compute autocorrelation-like features
        if n_features > 1:
            autocorr = np.corrcoef(row[:-1], row[1:])[0, 1] if n_features > 1 else 0.0
        else:
            autocorr = 0.0

        # Compute entropy from discretized values
        bins = 10
        hist, _ = np.histogram(row, bins=bins)
        probs = hist / (np.sum(hist) + 1e-10)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Create feature vector
        features = np.array(
            [mean_val, std_val, min_val, max_val, median_val, autocorr, entropy],
            dtype=np.float32,
        )

        latents.append(features)

    latents_array = np.array(latents, dtype=np.float32)

    # Normalize
    latents_array = normalize_array(latents_array, method="zscore")

    return latents_array
