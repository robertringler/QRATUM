"""Tests for SLT model forward pass and basic functionality."""

import numpy as np

from quasim.ownai.data import loaders
from quasim.ownai.determinism import hash_preds, set_seed
from quasim.ownai.models.slt import build_slt


def test_slt_build():
    """Test building SLT model."""

    model = build_slt(task="text-cls", seed=42)
    assert model is not None
    assert model.task == "text-cls"


def test_slt_text_classification():
    """Test SLT on text classification task."""

    set_seed(42)

    # Load small dataset
    texts, y = loaders.load_text("imdb-mini")
    X_train = texts[:100]
    y_train = y[:100]
    X_test = texts[100:120]
    y_test = y[100:120]

    # Build and train model
    model = build_slt(task="text-cls", seed=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert y_pred.dtype == np.int64


def test_slt_tabular_classification():
    """Test SLT on tabular classification task."""

    set_seed(42)

    # Load small dataset
    X, y = loaders.load_tabular("wine")
    X_train = X[:100]
    y_train = y[:100]
    X_test = X[100:120]
    y_test = y[100:120]

    # Build and train model
    model = build_slt(task="tabular-cls", seed=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert y_pred.dtype == np.int64


def test_slt_determinism():
    """Test that SLT produces deterministic predictions."""

    texts, y = loaders.load_text("imdb-mini")
    X_train = texts[:100]
    y_train = y[:100]
    X_test = texts[100:110]

    # Train and predict with seed 1337
    set_seed(1337)
    model1 = build_slt(task="text-cls", seed=1337)
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_test)
    hash1 = hash_preds(pred1)

    # Train and predict again with same seed
    set_seed(1337)
    model2 = build_slt(task="text-cls", seed=1337)
    model2.fit(X_train, y_train)
    pred2 = model2.predict(X_test)
    hash2 = hash_preds(pred2)

    # Hashes should match for determinism
    assert hash1 == hash2


def test_slt_with_symbolic_features():
    """Test SLT with symbolic features enabled."""

    texts, y = loaders.load_text("imdb-mini")
    X_train = texts[:50]
    y_train = y[:50]
    X_test = texts[50:60]

    model = build_slt(task="text-cls", seed=42, use_symbolic=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_slt_without_symbolic_features():
    """Test SLT without symbolic features."""

    texts, y = loaders.load_text("imdb-mini")
    X_train = texts[:50]
    y_train = y[:50]
    X_test = texts[50:60]

    model = build_slt(task="text-cls", seed=42, use_symbolic=False)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
