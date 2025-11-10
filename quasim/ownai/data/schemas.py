"""Data schema definitions for different task types."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

TaskType = Literal["tabular-cls", "tabular-reg", "text-cls", "vision-cls", "ts-reg"]


@dataclass
class TabularBatch:
    """Batch specification for tabular data.

    Attributes
    ----------
    X : NDArray
        Feature matrix of shape (batch_size, n_features)
    y : NDArray
        Target array of shape (batch_size,)
    feature_names : list[str], optional
        Names of features
    """

    X: NDArray[np.float32]
    y: NDArray[Any]
    feature_names: list[str] | None = None


@dataclass
class TextBatch:
    """Batch specification for text data.

    Attributes
    ----------
    input_ids : NDArray
        Token IDs of shape (batch_size, seq_len)
    attention_mask : NDArray, optional
        Attention mask of shape (batch_size, seq_len)
    y : NDArray
        Target labels of shape (batch_size,)
    """

    input_ids: NDArray[np.int64]
    attention_mask: NDArray[np.int64] | None = None
    y: NDArray[Any] = None


@dataclass
class VisionBatch:
    """Batch specification for vision data.

    Attributes
    ----------
    images : NDArray
        Image tensor of shape (batch_size, channels, height, width)
    y : NDArray
        Target labels of shape (batch_size,)
    """

    images: NDArray[np.float32]
    y: NDArray[np.int64]


@dataclass
class TimeSeriesBatch:
    """Batch specification for time series data.

    Attributes
    ----------
    sequences : NDArray
        Time series sequences of shape (batch_size, seq_len, n_features)
    y : NDArray
        Target values of shape (batch_size,) or (batch_size, forecast_horizon)
    """

    sequences: NDArray[np.float32]
    y: NDArray[np.float32]
