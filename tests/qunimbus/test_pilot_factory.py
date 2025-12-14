"""Tests for QuNimbus Pilot Factory."""

from quasim.qunimbus.pilot_factory import PilotFactory, PilotSpec


def test_pilot_factory_initialization():
    """Test PilotFactory initialization."""
    factory = PilotFactory(target_per_day=1000, veto_rate=0.008)

    assert factory.target_per_day == 1000
    assert factory.veto_rate == 0.008
    assert factory.pilots_generated == 0
    assert factory.vetoes == 0


def test_pilot_factory_custom_params():
    """Test PilotFactory with custom parameters."""
    factory = PilotFactory(target_per_day=500, veto_rate=0.01)

    assert factory.target_per_day == 500
    assert factory.veto_rate == 0.01


def test_generate_pilot():
    """Test single pilot generation."""
    factory = PilotFactory()

    pilot = factory.generate_pilot(1)

    assert isinstance(pilot, PilotSpec)
    assert pilot.pilot_id == "001"
    assert pilot.vertical in PilotFactory.VERTICALS
    assert 0.1 <= pilot.runtime_s <= 1.0
    assert 0.995 <= pilot.fidelity <= 0.999
    assert pilot.backend in ["PsiQuantum", "QuEra", "cuQuantum"]
    assert factory.pilots_generated == 1


def test_generate_batch():
    """Test batch pilot generation."""
    factory = PilotFactory()

    pilots = factory.generate_batch(count=10)

    assert len(pilots) == 10
    assert factory.pilots_generated == 10
    assert all(isinstance(p, PilotSpec) for p in pilots)


def test_pilot_verticals_distribution():
    """Test that pilots are distributed across verticals."""
    factory = PilotFactory()

    pilots = factory.generate_batch(count=100)
    verticals = {p.vertical for p in pilots}

    # Should cover all or most verticals with 100 pilots
    assert len(verticals) >= 8


def test_get_first_10_snapshot():
    """Test first 10 pilots snapshot."""
    factory = PilotFactory()

    snapshot = factory.get_first_10_snapshot()

    assert len(snapshot) == 10
    assert snapshot[0]["id"] == "001"
    assert snapshot[0]["vertical"] == "Aerospace"
    assert snapshot[9]["id"] == "010"
    assert snapshot[9]["vertical"] == "Retail"


def test_get_stats():
    """Test factory statistics."""
    factory = PilotFactory()

    # Generate some pilots
    factory.generate_batch(count=100)

    stats = factory.get_stats()

    assert stats["pilots_generated"] == 100
    assert stats["target_per_day"] == 1000
    assert 0 <= stats["veto_rate"] <= 0.02  # Should be close to 0.008


def test_pilot_fidelity_range():
    """Test that pilot fidelities are in expected range."""
    factory = PilotFactory()

    pilots = factory.generate_batch(count=50)

    for pilot in pilots:
        assert 0.995 <= pilot.fidelity <= 0.999


def test_pilot_runtime_range():
    """Test that pilot runtimes are in expected range."""
    factory = PilotFactory()

    pilots = factory.generate_batch(count=50)

    for pilot in pilots:
        assert 0.1 <= pilot.runtime_s <= 1.0


def test_pilot_has_all_fields():
    """Test that generated pilots have all required fields."""
    factory = PilotFactory()

    pilot = factory.generate_pilot(1)

    assert hasattr(pilot, "pilot_id")
    assert hasattr(pilot, "vertical")
    assert hasattr(pilot, "workload")
    assert hasattr(pilot, "runtime_s")
    assert hasattr(pilot, "fidelity")
    assert hasattr(pilot, "impact")
    assert hasattr(pilot, "backend")
    assert hasattr(pilot, "timestamp")
