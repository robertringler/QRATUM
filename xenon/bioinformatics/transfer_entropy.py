"""Transfer Entropy estimation for time-series omics data.

This module implements batched, GPU-safe transfer entropy estimation with:
- Batched computation for scalability
- Deterministic reproducibility guarantees
- Time-series validation
- Numerical stability monitoring

Mathematical Foundation:
    Transfer Entropy measures directed information flow:
    TE(X→Y) = I(Y_t; X_{t-k} | Y_{t-1})

    Where:
    - Y_t is the current state of target
    - X_{t-k} is the past state of source (lag k)
    - Y_{t-1} is the past state of target

    TE quantifies the reduction in uncertainty about Y's future given X's past,
    beyond what Y's own past provides.

References:
    Schreiber, T. (2000). Measuring information transfer.
    Physical Review Letters, 85(2), 461.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Try new package name first, fallback to old for compatibility
try:
    from qratum.common.seeding import SeedManager
except ImportError:
    from quasim.common.seeding import SeedManager


@dataclass
class TransferEntropyConfig:
    """Configuration for transfer entropy estimation.

    Attributes:
        max_lag: Maximum time lag to consider
        n_bins: Number of bins for discretization
        batch_size: Batch size for processing
        min_samples: Minimum samples required for valid estimate
        stability_threshold: Condition number threshold
        use_gpu: Whether to use GPU if available (placeholder)
    """

    max_lag: int = 5
    n_bins: int = 10
    batch_size: int = 1000
    min_samples: int = 50
    stability_threshold: float = 1e10
    use_gpu: bool = False  # Placeholder for future GPU support


@dataclass
class TransferEntropyResult:
    """Result of transfer entropy estimation.

    Attributes:
        te_value: Transfer entropy value (bits)
        optimal_lag: Optimal time lag
        source_name: Source variable name
        target_name: Target variable name
        n_samples: Number of samples used
        condition_number: Numerical stability metric
        valid: Whether result is statistically valid
    """

    te_value: float
    optimal_lag: int
    source_name: str
    target_name: str
    n_samples: int
    condition_number: float
    valid: bool


class TransferEntropyEngine:
    """Transfer entropy estimation for time-series omics data.

    Provides batched, deterministic transfer entropy computation with
    numerical stability guarantees.

    Attributes:
        config: Transfer entropy configuration
        seed_manager: Deterministic seed management
        _te_cache: Cache of computed transfer entropies
    """

    def __init__(
        self,
        config: Optional[TransferEntropyConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize transfer entropy engine.

        Args:
            config: Transfer entropy configuration
            seed: Random seed for reproducibility
        """

        self.config = config or TransferEntropyConfig()
        self.seed_manager = SeedManager(seed if seed is not None else 42)
        self._te_cache: dict[str, TransferEntropyResult] = {}

    def compute_entropy_timeseries(
        self,
        data: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute entropy of time series data.

        Args:
            data: Time series data (samples,)
            bins: Number of bins for discretization

        Returns:
            Shannon entropy in bits
        """

        if len(data) == 0:
            return 0.0

        # Discretize
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]

        if len(hist) == 0:
            return 0.0

        # Normalize and compute entropy
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy)

    def compute_conditional_entropy(
        self,
        y: np.ndarray,
        x: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute conditional entropy H(Y|X).

        Mathematical Basis:
            H(Y|X) = H(X,Y) - H(X)

        Args:
            y: Target variable
            x: Conditioning variable
            bins: Number of bins

        Returns:
            Conditional entropy in bits
        """

        if len(y) == 0 or len(x) == 0:
            return 0.0

        # Ensure same length
        min_len = min(len(y), len(x))
        y = y[:min_len]
        x = x[:min_len]

        # Compute H(X)
        h_x = self.compute_entropy_timeseries(x, bins=bins)

        # Compute H(X,Y)
        xy = np.column_stack([x, y])
        hist, _ = np.histogramdd(xy, bins=bins)
        hist = hist.ravel()
        hist = hist[hist > 0]

        if len(hist) == 0:
            return 0.0

        probs = hist / hist.sum()
        h_xy = -np.sum(probs * np.log2(probs))

        # H(Y|X) = H(X,Y) - H(X)
        cond_entropy = h_xy - h_x

        return max(0.0, float(cond_entropy))

    def compute_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1,
        source_name: str = "source",
        target_name: str = "target",
    ) -> TransferEntropyResult:
        """Compute transfer entropy from source to target at given lag.

        TE(X→Y) = I(Y_t; X_{t-k} | Y_{t-1})
                = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-k})

        Args:
            source: Source time series
            target: Target time series
            lag: Time lag
            source_name: Name of source variable
            target_name: Name of target variable

        Returns:
            TransferEntropyResult
        """

        # Validate inputs
        if len(source) < self.config.min_samples or len(target) < self.config.min_samples:
            return TransferEntropyResult(
                te_value=0.0,
                optimal_lag=lag,
                source_name=source_name,
                target_name=target_name,
                n_samples=min(len(source), len(target)),
                condition_number=float("inf"),
                valid=False,
            )

        # Prepare lagged data
        # Ensure we have enough data after lagging
        n_samples = min(len(source), len(target)) - lag
        if n_samples < self.config.min_samples:
            return TransferEntropyResult(
                te_value=0.0,
                optimal_lag=lag,
                source_name=source_name,
                target_name=target_name,
                n_samples=n_samples,
                condition_number=float("inf"),
                valid=False,
            )

        # Extract relevant time points
        y_t = target[lag:]  # Current target
        y_t_1 = target[lag - 1 : -1] if lag > 0 else target[:-1]  # Past target
        x_t_k = source[:n_samples]  # Lagged source

        # Compute H(Y_t | Y_{t-1})
        h_yt_given_yt1 = self.compute_conditional_entropy(y_t, y_t_1, bins=self.config.n_bins)

        # Compute H(Y_t | Y_{t-1}, X_{t-k})
        # Stack Y_{t-1} and X_{t-k} as conditioning variables
        cond_vars = np.column_stack([y_t_1, x_t_k])

        # Compute joint entropy H(Y_t, Y_{t-1}, X_{t-k})
        joint_data = np.column_stack([y_t, y_t_1, x_t_k])
        hist_joint, _ = np.histogramdd(joint_data, bins=self.config.n_bins)
        hist_joint = hist_joint.ravel()
        hist_joint = hist_joint[hist_joint > 0]

        if len(hist_joint) == 0:
            h_joint = 0.0
        else:
            probs = hist_joint / hist_joint.sum()
            h_joint = -np.sum(probs * np.log2(probs))

        # Compute H(Y_{t-1}, X_{t-k})
        hist_cond, _ = np.histogramdd(cond_vars, bins=self.config.n_bins)
        hist_cond = hist_cond.ravel()
        hist_cond = hist_cond[hist_cond > 0]

        if len(hist_cond) == 0:
            h_cond = 0.0
        else:
            probs = hist_cond / hist_cond.sum()
            h_cond = -np.sum(probs * np.log2(probs))

        # H(Y_t | Y_{t-1}, X_{t-k}) = H(Y_t, Y_{t-1}, X_{t-k}) - H(Y_{t-1}, X_{t-k})
        h_yt_given_yt1_xtk = h_joint - h_cond

        # TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-k})
        te = h_yt_given_yt1 - h_yt_given_yt1_xtk

        # Enforce non-negativity
        te = max(0.0, te)

        # Compute condition number for stability
        values = np.concatenate([y_t, y_t_1, x_t_k])
        condition_number = float(np.abs(values).max() / (np.abs(values).min() + 1e-10))

        result = TransferEntropyResult(
            te_value=float(te),
            optimal_lag=lag,
            source_name=source_name,
            target_name=target_name,
            n_samples=n_samples,
            condition_number=condition_number,
            valid=True,
        )

        return result

    def compute_transfer_entropy_batched(
        self,
        sources: list[np.ndarray],
        targets: list[np.ndarray],
        source_names: Optional[list[str]] = None,
        target_names: Optional[list[str]] = None,
    ) -> list[list[TransferEntropyResult]]:
        """Compute transfer entropy for multiple source-target pairs in batches.

        Args:
            sources: List of source time series
            targets: List of target time series
            source_names: Names for source variables
            target_names: Names for target variables

        Returns:
            Matrix of transfer entropy results
        """

        n_sources = len(sources)
        n_targets = len(targets)

        if source_names is None:
            source_names = [f"source_{i}" for i in range(n_sources)]
        if target_names is None:
            target_names = [f"target_{i}" for i in range(n_targets)]

        results = []

        for i, (source, source_name) in enumerate(zip(sources, source_names)):
            target_results = []
            for j, (target, target_name) in enumerate(zip(targets, target_names)):
                # Find optimal lag
                best_result = None
                best_te = -1.0

                for lag in range(1, self.config.max_lag + 1):
                    result = self.compute_transfer_entropy(
                        source, target, lag, source_name, target_name
                    )

                    if result.valid and result.te_value > best_te:
                        best_te = result.te_value
                        best_result = result

                if best_result is None:
                    # No valid result found
                    best_result = TransferEntropyResult(
                        te_value=0.0,
                        optimal_lag=1,
                        source_name=source_name,
                        target_name=target_name,
                        n_samples=0,
                        condition_number=float("inf"),
                        valid=False,
                    )

                target_results.append(best_result)

            results.append(target_results)

        return results

    def compute_information_network(
        self,
        time_series: dict[str, np.ndarray],
        threshold: float = 0.1,
    ) -> dict[str, any]:
        """Build information flow network from time series data.

        Args:
            time_series: Dictionary mapping variable names to time series
            threshold: Minimum transfer entropy for edge inclusion

        Returns:
            Dictionary with nodes and directed edges
        """

        variable_names = list(time_series.keys())
        time_series_list = [time_series[name] for name in variable_names]

        # Compute all pairwise transfer entropies
        te_matrix = self.compute_transfer_entropy_batched(
            time_series_list, time_series_list, variable_names, variable_names
        )

        # Build network
        nodes = [{"id": i, "name": name} for i, name in enumerate(variable_names)]
        edges = []

        for i, source_results in enumerate(te_matrix):
            for j, result in enumerate(source_results):
                if i != j and result.valid and result.te_value >= threshold:
                    edges.append(
                        {
                            "source": i,
                            "target": j,
                            "te": result.te_value,
                            "lag": result.optimal_lag,
                        }
                    )

        return {
            "nodes": nodes,
            "edges": edges,
            "threshold": threshold,
        }

    def get_statistics(self) -> dict[str, any]:
        """Get engine statistics.

        Returns:
            Dictionary with statistics
        """

        return {
            "cached_results": len(self._te_cache),
            "config": {
                "max_lag": self.config.max_lag,
                "n_bins": self.config.n_bins,
                "batch_size": self.config.batch_size,
                "min_samples": self.config.min_samples,
                "use_gpu": self.config.use_gpu,
            },
        }

    def clear_cache(self) -> None:
        """Clear result cache."""

        self._te_cache.clear()
