"""Symbolic-Latent Transformer (SLT) - QuASIM-Own Model 'Modo'.

This is the core model that fuses REVULTRA symbolic features with learned embeddings.
For the initial implementation, we use a simplified architecture based on sklearn components
combined with symbolic latent features.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from quasim.ownai.determinism import set_seed
from quasim.ownai.revultra.feats import (
    extract_symbolic_latents_batch,
    extract_symbolic_latents_numeric,
)


class SymbolicLatentTransformer:
    """Symbolic-Latent Transformer (SLT) - Modo.

    Fuses REVULTRA symbolic features with learned representations for
    deterministic, auditable AI predictions.

    Parameters
    ----------
    task : str
        Task type: 'tabular-cls', 'tabular-reg', 'text-cls', 'vision-cls', 'ts-reg'
    n_estimators : int
        Number of trees in the ensemble (default: 100)
    max_depth : int
        Maximum tree depth (default: 10)
    seed : int
        Random seed for determinism (default: 42)
    use_symbolic : bool
        Whether to use symbolic REVULTRA features (default: True)
    """

    def __init__(
        self,
        task: str = "tabular-cls",
        n_estimators: int = 100,
        max_depth: int = 10,
        seed: int = 42,
        use_symbolic: bool = True,
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.use_symbolic = use_symbolic

        set_seed(self.seed)

        # Determine if classification or regression
        if "cls" in task:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=1,  # Ensure determinism
            )
            self.is_classification = True
        elif "reg" in task:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=1,  # Ensure determinism
            )
            self.is_classification = False
        else:
            raise ValueError(f"Unknown task: {task}")

        self.tokenizer = None
        self.scaler = None

    def _extract_features(self, X: any) -> NDArray[np.float32]:
        """Extract features including symbolic latents.

        Parameters
        ----------
        X : any
            Input data (can be numeric array or list of texts)

        Returns
        -------
        NDArray
            Feature matrix
        """
        # Handle text input
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], str):
            # Text data - extract symbolic latents
            if self.use_symbolic:
                symbolic_feats = extract_symbolic_latents_batch(X)

                # For text, we'll use symbolic features directly
                # In a full implementation, we'd combine with embeddings
                return symbolic_feats
            else:
                # Simple bag-of-words representation
                from quasim.ownai.data.preprocess import SimpleTokenizer

                if self.tokenizer is None:
                    self.tokenizer = SimpleTokenizer(max_length=64)
                    self.tokenizer.fit(X)

                token_ids = self.tokenizer.transform(X)
                # Simple aggregation: mean of token IDs
                features = np.mean(token_ids, axis=1, keepdims=True).astype(np.float32)
                return features

        # Handle numeric input
        else:
            X = np.asarray(X, dtype=np.float32)

            if self.use_symbolic:
                # Extract symbolic latents from numeric data
                symbolic_feats = extract_symbolic_latents_numeric(X)

                # Concatenate with original features
                features = np.concatenate([X, symbolic_feats], axis=1)
                return features
            else:
                return X

    def fit(self, X: any, y: NDArray) -> "SymbolicLatentTransformer":
        """Train the Symbolic-Latent Transformer.

        Parameters
        ----------
        X : any
            Training features (numeric array or list of texts)
        y : NDArray
            Training targets

        Returns
        -------
        self
        """
        set_seed(self.seed)

        # Extract features including symbolic latents
        features = self._extract_features(X)

        # Train the model
        self.model.fit(features, y)

        return self

    def predict(self, X: any) -> NDArray:
        """Make predictions.

        Parameters
        ----------
        X : any
            Input features

        Returns
        -------
        NDArray
            Predictions
        """
        features = self._extract_features(X)
        return self.model.predict(features)

    def predict_proba(self, X: any) -> NDArray[np.float32]:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : any
            Input features

        Returns
        -------
        NDArray
            Class probabilities
        """
        if not self.is_classification:
            raise ValueError("predict_proba only available for classification")

        features = self._extract_features(X)
        return self.model.predict_proba(features)


def build_slt(
    task: str = "tabular-cls",
    seed: int = 42,
    use_symbolic: bool = True,
) -> SymbolicLatentTransformer:
    """Build a Symbolic-Latent Transformer model.

    Parameters
    ----------
    task : str
        Task type
    seed : int
        Random seed
    use_symbolic : bool
        Whether to use symbolic features

    Returns
    -------
    SymbolicLatentTransformer
        Initialized SLT model

    Examples
    --------
    >>> model = build_slt(task="text-cls", seed=1337)
    >>> model.task
    'text-cls'
    """
    return SymbolicLatentTransformer(task=task, seed=seed, use_symbolic=use_symbolic)
