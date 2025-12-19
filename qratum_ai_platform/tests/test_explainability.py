"""Tests for SHAP explainer."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.explainability import SHAPExplainer, explain_prediction


def test_shap_explainer_init():
    """Test SHAP explainer initialization."""
    explainer = SHAPExplainer()
    assert explainer.model_type == "text"

    explainer_tabular = SHAPExplainer("tabular")
    assert explainer_tabular.model_type == "tabular"


def test_explain_text_input():
    """Test explanation for text input."""
    explainer = SHAPExplainer("text")
    input_text = "This is a sample input text for testing"
    prediction = "positive"

    result = explainer.explain(input_text, prediction, top_k=5)

    assert result["model_type"] == "text"
    assert result["prediction"] == "positive"
    assert "explanations" in result
    assert len(result["explanations"]) <= 5
    assert all("token" in e for e in result["explanations"])
    assert all("importance" in e for e in result["explanations"])


def test_explain_generic_input():
    """Test explanation for generic input."""
    explainer = SHAPExplainer("tabular")
    input_data = [1.0, 2.0, 3.0]
    prediction = 0.85

    result = explainer.explain(input_data, prediction, top_k=10)

    assert result["model_type"] == "tabular"
    assert "explanations" in result
    assert all("feature" in e for e in result["explanations"])
    assert all("importance" in e for e in result["explanations"])


def test_importance_values():
    """Test that importance values are numeric and within reasonable range."""
    explainer = SHAPExplainer()
    result = explainer.explain("test input", "test pred", top_k=5)

    for explanation in result["explanations"]:
        importance = explanation["importance"]
        assert isinstance(importance, float)
        assert -1.0 <= importance <= 1.0


def test_importance_ordering():
    """Test that explanations are ordered by importance."""
    explainer = SHAPExplainer()
    result = explainer.explain("one two three four five", "pred", top_k=5)

    importances = [abs(e["importance"]) for e in result["explanations"]]
    # Should be in descending order
    assert importances == sorted(importances, reverse=True)


def test_top_k_limiting():
    """Test that top_k limits number of explanations."""
    explainer = SHAPExplainer()

    result = explainer.explain("a b c d e f g h", "pred", top_k=3)
    assert len(result["explanations"]) <= 3

    result = explainer.explain("a b c d e f g h", "pred", top_k=10)
    assert len(result["explanations"]) <= 10


def test_batch_explain():
    """Test batch explanation."""
    explainer = SHAPExplainer()
    inputs = ["first input", "second input", "third input"]
    predictions = ["pos", "neg", "neutral"]

    results = explainer.batch_explain(inputs, predictions, top_k=5)

    assert len(results) == 3
    assert all("explanations" in r for r in results)
    assert [r["prediction"] for r in results] == ["pos", "neg", "neutral"]


def test_explain_prediction_convenience():
    """Test convenience function."""
    result = explain_prediction("test input text", "prediction")

    assert "explanations" in result
    assert result["prediction"] == "prediction"


def test_total_importance():
    """Test that total importance is calculated."""
    explainer = SHAPExplainer()
    result = explainer.explain("sample text", "pred", top_k=5)

    assert "total_importance" in result
    assert result["total_importance"] > 0


def test_generate_importance_vector():
    """Test importance vector generation."""
    explainer = SHAPExplainer()
    vec = explainer._generate_importance_vector(10)

    assert len(vec) == 10
    assert all(isinstance(x, float) for x in vec)
    # Values should be in reasonable range
    assert all(-1.0 <= x <= 1.0 for x in vec)
