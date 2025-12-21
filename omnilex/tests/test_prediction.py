"""Tests for litigation prediction engine.

Version: 1.0.0
Status: Production
"""

import pytest

from omnilex.prediction import LitigationPredictionEngine


class TestLitigationPredictionEngine:
    """Test litigation prediction engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = LitigationPredictionEngine(seed=42)
        assert engine.seed == 42

    def test_predict_outcome_contract(self):
        """Test outcome prediction for contract case."""
        engine = LitigationPredictionEngine(seed=42)

        result = engine.predict_outcome(
            case_type="contract_breach",
            jurisdiction="US",
            key_facts={"claimed_damages": 100000}
        )

        assert "plaintiff_win_probability" in result
        assert "defendant_win_probability" in result
        assert "expected_trial_outcome" in result
        assert "damages_estimate" in result
        assert "settlement_range" in result
        assert "caveats" in result

        # Probabilities should sum to 1
        assert abs(
            result["plaintiff_win_probability"] +
            result["defendant_win_probability"] - 1.0
        ) < 0.01

    def test_predict_outcome_tort(self):
        """Test outcome prediction for tort case."""
        engine = LitigationPredictionEngine(seed=42)

        result = engine.predict_outcome(
            case_type="negligence",
            jurisdiction="US-CA",
            key_facts={"claimed_damages": 500000}
        )

        assert result["case_type"] == "negligence"
        assert result["damages_estimate"]["claimed_amount"] == 500000

    def test_get_historical_rates_contract(self):
        """Test getting historical rates for contract cases."""
        engine = LitigationPredictionEngine()

        rates = engine._get_historical_rates("contract_breach", "US")

        assert "plaintiff_win_rate" in rates
        assert "median_damages" in rates
        assert "trial_rate" in rates
        assert 0.0 <= rates["plaintiff_win_rate"] <= 1.0

    def test_get_historical_rates_unknown(self):
        """Test getting historical rates for unknown case type."""
        engine = LitigationPredictionEngine()

        rates = engine._get_historical_rates("unknown_case_type", "US")

        # Should return defaults
        assert rates["plaintiff_win_rate"] == 0.50

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        engine = LitigationPredictionEngine(seed=42)

        rates = {"plaintiff_win_rate": 0.65}
        result = engine._monte_carlo_simulation(rates, n_simulations=1000)

        assert "plaintiff_win_prob" in result
        assert "confidence_interval" in result
        assert "expected_outcome" in result

        # With seed, result should be deterministic
        result2 = LitigationPredictionEngine(seed=42)._monte_carlo_simulation(
            rates, n_simulations=1000
        )
        assert result["plaintiff_win_prob"] == result2["plaintiff_win_prob"]

    def test_monte_carlo_confidence_interval(self):
        """Test confidence interval calculation."""
        engine = LitigationPredictionEngine(seed=42)

        rates = {"plaintiff_win_rate": 0.50}
        result = engine._monte_carlo_simulation(rates, n_simulations=10000)

        ci_lower, ci_upper = result["confidence_interval"]

        # CI should bracket the probability
        assert ci_lower <= result["plaintiff_win_prob"] <= ci_upper

        # CI should be reasonable width
        assert (ci_upper - ci_lower) < 0.1  # Less than 10% width

    def test_estimate_damages_contract(self):
        """Test damages estimation for contract case."""
        engine = LitigationPredictionEngine()

        facts = {"claimed_damages": 100000}
        damages = engine._estimate_damages(facts, "contract_breach")

        assert "low_estimate" in damages
        assert "mid_estimate" in damages
        assert "high_estimate" in damages
        assert damages["low_estimate"] < damages["mid_estimate"] < damages["high_estimate"]
        assert damages["claimed_amount"] == 100000

    def test_estimate_damages_tort(self):
        """Test damages estimation for tort case."""
        engine = LitigationPredictionEngine()

        facts = {"claimed_damages": 200000, "punitive_eligible": True}
        damages = engine._estimate_damages(facts, "negligence")

        assert damages["components"]["punitive"] > 0

    def test_calculate_settlement_value(self):
        """Test settlement value calculation."""
        engine = LitigationPredictionEngine()

        simulation = {
            "plaintiff_win_prob": 0.65,
            "defendant_win_prob": 0.35
        }
        damages = {
            "mid_estimate": 100000
        }

        settlement = engine._calculate_settlement_value(simulation, damages)

        assert "minimum" in settlement
        assert "reasonable" in settlement
        assert "maximum" in settlement
        assert "expected_value" in settlement

        # Settlement range should be reasonable
        assert settlement["minimum"] < settlement["reasonable"] < settlement["maximum"]

        # Settlement should be less than full damages (discounted for risk)
        assert settlement["reasonable"] < damages["mid_estimate"]

    def test_deterministic_with_seed(self):
        """Test that predictions are deterministic with same seed."""
        facts = {"claimed_damages": 100000}

        engine1 = LitigationPredictionEngine(seed=42)
        result1 = engine1.predict_outcome("contract_breach", "US", facts)

        engine2 = LitigationPredictionEngine(seed=42)
        result2 = engine2.predict_outcome("contract_breach", "US", facts)

        # Results should be identical
        assert result1["plaintiff_win_probability"] == result2["plaintiff_win_probability"]
        assert result1["settlement_range"]["reasonable"] == result2["settlement_range"]["reasonable"]

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        facts = {"claimed_damages": 100000}

        engine1 = LitigationPredictionEngine(seed=42)
        result1 = engine1.predict_outcome("contract_breach", "US", facts)

        engine2 = LitigationPredictionEngine(seed=123)
        result2 = engine2.predict_outcome("contract_breach", "US", facts)

        # Results should differ (Monte Carlo variation)
        # Note: There's a tiny chance they could be the same, but extremely unlikely
        assert result1["plaintiff_win_probability"] != result2["plaintiff_win_probability"] or \
               result1["settlement_range"]["reasonable"] != result2["settlement_range"]["reasonable"]
