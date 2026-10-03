"""Monte Carlo campaign: reproducibility and honest percentile reporting."""

from __future__ import annotations

import math

import pytest

from fluxa.montecarlo import (
    PERCENTILE_MIN_SAMPLES,
    CampaignConfig,
    draw_samples,
    percentiles,
    run_campaign,
)


def test_samples_are_reproducible_from_the_campaign_seed():
    a = draw_samples(CampaignConfig(campaign_seed=4242, n_samples=12))
    b = draw_samples(CampaignConfig(campaign_seed=4242, n_samples=12))
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_a_different_campaign_seed_gives_different_samples():
    a = draw_samples(CampaignConfig(campaign_seed=1, n_samples=8))
    b = draw_samples(CampaignConfig(campaign_seed=2, n_samples=8))
    assert [s.to_dict() for s in a] != [s.to_dict() for s in b]


def test_sample_k_is_independent_of_the_campaign_size():
    """Required for the scaling benchmark: the 1-, 10- and 100-sample points
    must be nested subsets of one experiment, not three different workloads."""
    small = draw_samples(CampaignConfig(n_samples=5))
    large = draw_samples(CampaignConfig(n_samples=100))
    assert [s.to_dict() for s in small] == [s.to_dict() for s in large[:5]]


def test_drawn_parameters_stay_inside_their_declared_support():
    config = CampaignConfig(n_samples=200)
    for sample in draw_samples(config):
        assert config.load_factor_min <= sample.load_factor <= config.load_factor_max
        assert config.solar_factor_min <= sample.solar_factor <= config.solar_factor_max
        assert config.wind_factor_min <= sample.wind_factor <= config.wind_factor_max
        assert config.disturbance_onset_min_h <= sample.onset_h <= config.disturbance_onset_max_h
        assert sample.vre_shock_target in ("SOLAR_PV_1", "WIND_1")
        assert (
            config.outage_duration_min_minutes
            <= sample.outage_duration_min
            <= config.outage_duration_max_minutes
        )


def test_both_outage_and_shock_branches_are_exercised():
    samples = draw_samples(CampaignConfig(n_samples=100))
    assert any(s.outage for s in samples) and any(not s.outage for s in samples)
    assert any(s.vre_shock for s in samples) and any(not s.vre_shock for s in samples)


def test_percentiles_are_withheld_when_the_sample_size_is_too_small():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = percentiles(values, len(values))
    assert out["P50"] is not None
    assert out["P90"] is None and out["P95"] is None and out["P99"] is None


def test_percentiles_are_reported_once_the_tail_is_populated():
    out = percentiles([float(i) for i in range(100)], 100)
    for p in PERCENTILE_MIN_SAMPLES:
        assert out[f"P{p}"] is not None


def test_percentile_gate_uses_finite_observations_not_the_campaign_size():
    """A metric undefined for most samples must not get a P99 backed by a
    handful of observations just because the campaign drew 100 samples."""
    values = [float(i) for i in range(15)] + [None] * 85
    out = percentiles(values, 100)
    assert out["n"] == 15 and out["n_campaign_samples"] == 100
    assert out["P50"] is not None and out["P90"] is not None
    assert out["P95"] is None, "P95 needs 20 observations, only 15 are finite"
    assert out["P99"] is None


def test_non_finite_values_are_excluded_and_counted():
    out = percentiles([1.0, 2.0, float("inf"), float("nan"), None], 5)
    assert out["n"] == 2
    assert out["n_excluded_non_finite"] == 3


def test_empty_input_yields_no_statistics():
    out = percentiles([None, float("nan")], 2)
    assert out["n"] == 0
    assert out["mean"] is None
    for p in PERCENTILE_MIN_SAMPLES:
        assert out[f"P{p}"] is None


def test_small_campaign_runs_and_reports_every_metric():
    from fluxa.montecarlo import DISTRIBUTION_METRICS

    campaign = run_campaign(CampaignConfig(n_samples=4, n_steps=48))
    assert len(campaign.rows) == 4
    assert not campaign.failures
    assert set(campaign.distributions) == set(DISTRIBUTION_METRICS)
    assert [r["sample_index"] for r in campaign.rows] == [0, 1, 2, 3]


def test_parallel_and_serial_campaigns_agree_exactly():
    """A speed-up that changes the answer is not a speed-up."""
    config = CampaignConfig(n_samples=4, n_steps=48)
    serial = run_campaign(config, parallel=False)
    parallel = run_campaign(config, parallel=True)
    assert [r["identity_hash"] for r in serial.rows] == [
        r["identity_hash"] for r in parallel.rows
    ]
    assert [r["sample_index"] for r in parallel.rows] == [0, 1, 2, 3]


def test_every_sample_is_authorized_and_records_its_authority():
    campaign = run_campaign(CampaignConfig(n_samples=2, n_steps=24))
    for row in campaign.rows:
        assert row["required_safety_level"] in ("ROUTINE", "ELEVATED", "SENSITIVE")
        assert math.isfinite(row["total_operating_cost_usd"])
