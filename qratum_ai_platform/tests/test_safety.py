"""Tests for safety policy engine."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.safety import PolicyEngine, PolicyVerdict, evaluate_safety


def test_policy_engine_init():
    """Test policy engine initialization."""
    engine = PolicyEngine()
    assert len(engine.blocked_patterns) > 0
    assert len(engine.flagged_patterns) > 0


def test_evaluate_clean_text():
    """Test evaluation of clean text."""
    engine = PolicyEngine()
    result = engine.evaluate("This is safe text without any issues")

    assert result["verdict"] == PolicyVerdict.APPROVED.value
    assert result["violation_count"] == 0
    assert result["sanitized_text"] == "This is safe text without any issues"


def test_evaluate_blocked_content():
    """Test evaluation of blocked content."""
    engine = PolicyEngine()

    # Test secret detection
    result = engine.evaluate("Here is my api_key: ABC123")
    assert result["verdict"] == PolicyVerdict.BLOCKED.value
    assert result["violation_count"] > 0
    assert "[REDACTED]" in result["sanitized_text"]

    # Test SSN-like pattern
    result = engine.evaluate("SSN: 123-45-6789")
    assert result["verdict"] == PolicyVerdict.BLOCKED.value
    assert "[REDACTED]" in result["sanitized_text"]


def test_evaluate_flagged_content():
    """Test evaluation of flagged content."""
    engine = PolicyEngine()

    result = engine.evaluate("This is confidential information")
    assert result["verdict"] == PolicyVerdict.FLAGGED.value
    assert result["violation_count"] > 0

    result = engine.evaluate("TODO: fix this later")
    assert result["verdict"] == PolicyVerdict.FLAGGED.value


def test_evaluate_with_metadata():
    """Test evaluation with metadata."""
    engine = PolicyEngine()
    metadata = {"user": "test_user", "context": "internal"}

    result = engine.evaluate("Some text", metadata=metadata)
    assert "verdict" in result


def test_add_blocked_pattern():
    """Test adding custom blocked patterns."""
    engine = PolicyEngine()
    initial_count = len(engine.blocked_patterns)

    engine.add_blocked_pattern(r"\bcustom[_-]?pattern\b")
    assert len(engine.blocked_patterns) == initial_count + 1

    # Test new pattern
    result = engine.evaluate("This has custom_pattern in it")
    assert result["verdict"] == PolicyVerdict.BLOCKED.value


def test_add_flagged_pattern():
    """Test adding custom flagged patterns."""
    engine = PolicyEngine()
    initial_count = len(engine.flagged_patterns)

    engine.add_flagged_pattern(r"\bwatchword\b")
    assert len(engine.flagged_patterns) == initial_count + 1

    # Test new pattern
    result = engine.evaluate("This has watchword in it")
    assert result["verdict"] == PolicyVerdict.FLAGGED.value


def test_evaluate_safety_convenience():
    """Test convenience function."""
    result = evaluate_safety("Clean text here")
    assert result["verdict"] == PolicyVerdict.APPROVED.value

    result = evaluate_safety("Secret password detected")
    assert result["verdict"] == PolicyVerdict.BLOCKED.value


def test_multiple_violations():
    """Test text with multiple violations."""
    engine = PolicyEngine()
    text = "My password is secret123 and api_key is ABC"

    result = engine.evaluate(text)
    assert result["verdict"] == PolicyVerdict.BLOCKED.value
    assert result["violation_count"] >= 2


def test_sanitization_preserves_length():
    """Test that sanitization maintains reasonable text structure."""
    engine = PolicyEngine()
    result = engine.evaluate("Short password text")

    # Sanitized text should exist
    assert "sanitized_text" in result
    assert result["sanitized_length"] > 0
