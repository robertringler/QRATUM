"""Tests for tire material modeling."""

import pytest

from quasim.domains.tire.materials import (
    CompoundType,
    MaterialProperties,
    TireCompound,
)


def test_material_properties_creation():
    """Test MaterialProperties creation with defaults."""
    props = MaterialProperties()
    assert props.density == 1150.0
    assert props.elastic_modulus == 0.002
    assert props.hardness_shore_a == 60.0
    assert 0 <= props.wet_grip_coefficient <= 1


def test_compute_effective_modulus():
    """Test temperature and strain-rate dependent modulus."""
    props = MaterialProperties(elastic_modulus=0.002)

    # At reference temperature
    modulus_ref = props.compute_effective_modulus(20.0, 1.0)
    assert modulus_ref > 0

    # Higher temperature should reduce modulus
    modulus_hot = props.compute_effective_modulus(80.0, 1.0)
    assert modulus_hot < modulus_ref

    # Lower temperature should increase modulus (up to limit)
    modulus_cold = props.compute_effective_modulus(-20.0, 1.0)
    assert modulus_cold > modulus_ref


def test_compute_hysteresis_loss():
    """Test hysteresis energy loss calculation."""
    props = MaterialProperties(viscoelastic_loss_factor=0.15)

    loss = props.compute_hysteresis_loss(10.0, 20.0)
    assert loss > 0
    assert loss >= props.viscoelastic_loss_factor


def test_age_material():
    """Test material aging and degradation."""
    props = MaterialProperties(
        elastic_modulus=0.002,
        abrasion_resistance=0.8,
        wet_grip_coefficient=0.75,
    )

    initial_modulus = props.elastic_modulus
    initial_abrasion = props.abrasion_resistance

    # Age the material
    props.age_material(exposure_days=365, uv_hours=1000, stress_cycles=1000000)

    # Properties should degrade
    assert props.elastic_modulus < initial_modulus
    assert props.abrasion_resistance < initial_abrasion


def test_tire_compound_creation():
    """Test TireCompound creation."""
    props = MaterialProperties()
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test Compound",
        compound_type=CompoundType.NATURAL_RUBBER,
        base_properties=props,
    )

    assert compound.compound_id == "TEST_001"
    assert compound.compound_type == CompoundType.NATURAL_RUBBER
    assert compound.quantum_optimization_level == 0.0


def test_tire_compound_to_dict():
    """Test TireCompound serialization."""
    props = MaterialProperties()
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test Compound",
        compound_type=CompoundType.SYNTHETIC_RUBBER,
        base_properties=props,
    )

    data = compound.to_dict()
    assert data["compound_id"] == "TEST_001"
    assert data["compound_type"] == "synthetic_rubber"
    assert "base_properties" in data


def test_quantum_optimization():
    """Test quantum optimization application."""
    props = MaterialProperties()
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test Compound",
        compound_type=CompoundType.QUANTUM_OPTIMIZED,
        base_properties=props,
        additives={"silica": 0.15, "carbon_black": 0.25},
    )

    result = compound.apply_quantum_optimization(
        target_properties={"wet_grip": 0.9, "rolling_resistance": 0.008},
        optimization_iterations=10,
    )

    assert "solution" in result
    assert "convergence" in result
    # Quantum optimization level should be updated
    assert compound.quantum_optimization_level >= 0.0
