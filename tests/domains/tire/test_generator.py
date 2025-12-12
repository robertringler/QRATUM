"""Tests for tire scenario generator."""

import pytest

from quasim.domains.tire.generator import TireScenarioGenerator


def test_generator_creation():
    """Test TireScenarioGenerator initialization."""
    gen = TireScenarioGenerator(random_seed=42)
    assert gen.random_seed == 42


def test_generate_compound_variants():
    """Test compound variant generation."""
    gen = TireScenarioGenerator(random_seed=42)
    compounds = gen.generate_compound_variants(count=8)

    assert len(compounds) == 8
    assert all(c.compound_id for c in compounds)
    assert all(c.compound_type for c in compounds)


def test_generate_geometry_variants():
    """Test geometry variant generation."""
    gen = TireScenarioGenerator(random_seed=42)
    geometries = gen.generate_geometry_variants(count=10)

    assert len(geometries) == 10
    assert all(g.tire_id for g in geometries)
    assert all(g.tire_type for g in geometries)
    assert all(g.width > 0 for g in geometries)


def test_generate_environment_variants():
    """Test environment variant generation."""
    gen = TireScenarioGenerator(random_seed=42)
    environments = gen.generate_environment_variants(count=10)

    assert len(environments) == 10
    assert all(e.surface_type for e in environments)
    assert all(e.weather for e in environments)
    # Check temperature range coverage
    temps = [e.ambient_temperature for e in environments]
    assert min(temps) < 0  # Should have cold temperatures
    assert max(temps) > 40  # Should have hot temperatures


def test_generate_operating_conditions():
    """Test operating condition generation."""
    gen = TireScenarioGenerator(random_seed=42)
    conditions = gen.generate_operating_conditions(count=25)

    assert len(conditions) == 25
    for load, pressure, speed in conditions:
        assert 200 <= load <= 1200
        assert 180 <= pressure <= 350
        assert 30 <= speed <= 250


def test_generate_scenarios():
    """Test scenario generation."""
    gen = TireScenarioGenerator(random_seed=42)
    scenarios = gen.generate_scenarios(target_count=100)

    assert len(scenarios) == 100
    assert all("simulation_id" in s for s in scenarios)
    assert all("compound" in s for s in scenarios)
    assert all("geometry" in s for s in scenarios)
    assert all("environment" in s for s in scenarios)


def test_scenario_diversity():
    """Test that generated scenarios are diverse."""
    gen = TireScenarioGenerator(random_seed=42)
    scenarios = gen.generate_scenarios(target_count=500)  # Increase count for diversity

    # Check diversity of compounds
    compound_types = set(s["compound"].compound_type for s in scenarios)
    assert len(compound_types) > 1

    # Check diversity of tire types
    tire_types = set(s["geometry"].tire_type for s in scenarios)
    assert len(tire_types) > 1

    # Check diversity of surfaces
    surfaces = set(s["environment"].surface_type for s in scenarios)
    assert len(surfaces) > 1


def test_deterministic_generation():
    """Test that generation is deterministic with same seed."""
    gen1 = TireScenarioGenerator(random_seed=42)
    scenarios1 = gen1.generate_scenarios(target_count=50)

    gen2 = TireScenarioGenerator(random_seed=42)
    scenarios2 = gen2.generate_scenarios(target_count=50)

    # Should generate same scenarios
    assert len(scenarios1) == len(scenarios2)
    for s1, s2 in zip(scenarios1, scenarios2):
        assert s1["simulation_id"] == s2["simulation_id"]
        assert s1["compound"].compound_type == s2["compound"].compound_type
        assert s1["geometry"].tire_type == s2["geometry"].tire_type
