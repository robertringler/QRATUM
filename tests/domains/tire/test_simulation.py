"""Tests for tire simulation engine."""

from quasim.domains.tire.environment import (EnvironmentalConditions,
                                             RoadSurface)
from quasim.domains.tire.geometry import (TireGeometry, TireStructure,
                                          TireType, TreadDesign)
from quasim.domains.tire.materials import (CompoundType, MaterialProperties,
                                           TireCompound)
from quasim.domains.tire.simulation import TireSimulation


def test_tire_simulation_creation():
    """Test TireSimulation initialization."""

    sim = TireSimulation(use_quantum_acceleration=True, random_seed=42)
    assert sim.use_quantum_acceleration is True
    assert sim.random_seed == 42


def test_complete_simulation():
    """Test complete tire simulation workflow."""

    # Create tire compound
    props = MaterialProperties()
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test Compound",
        compound_type=CompoundType.NATURAL_RUBBER,
        base_properties=props,
    )

    # Create tire geometry
    tread = TreadDesign()
    structure = TireStructure()
    geometry = TireGeometry(
        tire_id="TIRE_001",
        tire_type=TireType.PASSENGER,
        width=225,
        aspect_ratio=45,
        diameter=17,
        tread_design=tread,
        structure=structure,
    )

    # Create environmental conditions
    environment = EnvironmentalConditions(
        ambient_temperature=20.0,
        road_temperature=25.0,
        surface_type=RoadSurface.DRY_ASPHALT,
    )

    # Run simulation
    sim = TireSimulation(use_quantum_acceleration=True, random_seed=42)
    result = sim.simulate(
        simulation_id="SIM_001",
        tire_geometry=geometry,
        compound=compound,
        environment=environment,
        load_kg=500.0,
        pressure_kpa=250.0,
        speed_kmh=100.0,
    )

    # Verify results
    assert result.simulation_id == "SIM_001"
    assert result.quantum_enhanced is True
    assert result.performance_metrics.grip_coefficient > 0
    assert result.performance_metrics.rolling_resistance > 0
    assert result.performance_metrics.predicted_lifetime_km > 0


def test_performance_metrics():
    """Test performance metrics computation."""

    props = MaterialProperties(
        wet_grip_coefficient=0.75,
        rolling_resistance_coeff=0.010,
        abrasion_resistance=0.8,
    )
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test",
        compound_type=CompoundType.NATURAL_RUBBER,
        base_properties=props,
    )

    tread = TreadDesign()
    structure = TireStructure()
    geometry = TireGeometry(
        tire_id="TIRE_001",
        tire_type=TireType.PASSENGER,
        width=225,
        aspect_ratio=45,
        diameter=17,
        tread_design=tread,
        structure=structure,
    )

    environment = EnvironmentalConditions()

    sim = TireSimulation(random_seed=42)
    result = sim.simulate(
        simulation_id="SIM_001",
        tire_geometry=geometry,
        compound=compound,
        environment=environment,
        load_kg=500.0,
        pressure_kpa=250.0,
        speed_kmh=100.0,
    )

    metrics = result.performance_metrics

    # Check all metrics are present
    assert metrics.grip_coefficient >= 0
    assert metrics.dry_grip >= 0
    assert metrics.wet_grip >= 0
    assert metrics.rolling_resistance > 0
    assert metrics.wear_rate > 0
    assert metrics.thermal_performance >= 0
    assert metrics.noise_level > 0
    assert metrics.predicted_lifetime_km > 0
    assert metrics.optimization_score >= 0


def test_thermal_distribution():
    """Test thermal distribution computation."""

    props = MaterialProperties()
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test",
        compound_type=CompoundType.NATURAL_RUBBER,
        base_properties=props,
    )

    tread = TreadDesign()
    structure = TireStructure()
    geometry = TireGeometry(
        tire_id="TIRE_001",
        tire_type=TireType.PASSENGER,
        width=225,
        aspect_ratio=45,
        diameter=17,
        tread_design=tread,
        structure=structure,
    )

    environment = EnvironmentalConditions(
        ambient_temperature=30.0,
        road_temperature=40.0,
    )

    sim = TireSimulation(random_seed=42)
    result = sim.simulate(
        simulation_id="SIM_001",
        tire_geometry=geometry,
        compound=compound,
        environment=environment,
        load_kg=500.0,
        pressure_kpa=250.0,
        speed_kmh=120.0,
    )

    thermal = result.thermal_map
    assert "tread_center" in thermal
    assert "sidewall" in thermal
    assert "max_temperature" in thermal
    assert thermal["max_temperature"] >= environment.ambient_temperature


def test_optimization_suggestions():
    """Test optimization suggestions generation."""

    props = MaterialProperties(
        wet_grip_coefficient=0.5,  # Low wet grip
        rolling_resistance_coeff=0.015,  # High rolling resistance
    )
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test",
        compound_type=CompoundType.NATURAL_RUBBER,
        base_properties=props,
        quantum_optimization_level=0.0,  # Not optimized
    )

    tread = TreadDesign(groove_depth=5.0)  # Shallow grooves
    structure = TireStructure()
    geometry = TireGeometry(
        tire_id="TIRE_001",
        tire_type=TireType.PASSENGER,
        width=225,
        aspect_ratio=45,
        diameter=17,
        tread_design=tread,
        structure=structure,
    )

    environment = EnvironmentalConditions(surface_wetness=0.8)

    sim = TireSimulation(random_seed=42)
    result = sim.simulate(
        simulation_id="SIM_001",
        tire_geometry=geometry,
        compound=compound,
        environment=environment,
        load_kg=500.0,
        pressure_kpa=250.0,
        speed_kmh=100.0,
    )

    # Should have suggestions due to poor performance
    assert len(result.optimization_suggestions) > 0


def test_result_serialization():
    """Test simulation result serialization."""

    props = MaterialProperties()
    compound = TireCompound(
        compound_id="TEST_001",
        name="Test",
        compound_type=CompoundType.NATURAL_RUBBER,
        base_properties=props,
    )

    tread = TreadDesign()
    structure = TireStructure()
    geometry = TireGeometry(
        tire_id="TIRE_001",
        tire_type=TireType.PASSENGER,
        width=225,
        aspect_ratio=45,
        diameter=17,
        tread_design=tread,
        structure=structure,
    )

    environment = EnvironmentalConditions()

    sim = TireSimulation(random_seed=42)
    result = sim.simulate(
        simulation_id="SIM_001",
        tire_geometry=geometry,
        compound=compound,
        environment=environment,
        load_kg=500.0,
        pressure_kpa=250.0,
        speed_kmh=100.0,
    )

    # Serialize to dict
    data = result.to_dict()

    assert data["simulation_id"] == "SIM_001"
    assert "input_parameters" in data
    assert "performance_metrics" in data
    assert "thermal_analysis" in data
    assert "optimization_suggestions" in data
