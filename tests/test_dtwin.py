"""Digital twin module tests.

This module tests the QuASIM digital twin functionality including:
- Digital twin creation and validation
- State management
- Forward simulation
- State evolution
"""

from __future__ import annotations

import pytest

from quasim.dtwin import DigitalTwin, StateManager


class TestStateManager:
    """Test state management functionality."""

    def test_state_manager_initialization(self):
        """Test state manager initialization."""
        sm = StateManager()
        assert sm.get_current_state() == {}

    def test_state_update(self):
        """Test updating state."""
        sm = StateManager()
        state = {"temperature": 300.0, "pressure": 101325.0}
        sm.update(state)
        assert sm.get_current_state() == state

    def test_multiple_state_updates(self):
        """Test multiple state updates."""
        sm = StateManager()
        sm.update({"x": 1.0})
        sm.update({"x": 2.0, "y": 3.0})
        current = sm.get_current_state()
        assert current["x"] == 2.0
        assert current["y"] == 3.0


class TestDigitalTwin:
    """Test digital twin functionality."""

    def test_digital_twin_creation_aerospace(self):
        """Test creating aerospace digital twin."""
        twin = DigitalTwin(twin_id="aero-001", system_type="aerospace")
        assert twin.twin_id == "aero-001"
        assert twin.system_type == "aerospace"
        assert isinstance(twin.state_manager, StateManager)

    def test_digital_twin_creation_pharma(self):
        """Test creating pharmaceutical digital twin."""
        twin = DigitalTwin(twin_id="pharma-001", system_type="pharma")
        assert twin.twin_id == "pharma-001"
        assert twin.system_type == "pharma"

    def test_digital_twin_creation_finance(self):
        """Test creating finance digital twin."""
        twin = DigitalTwin(twin_id="fin-001", system_type="finance")
        assert twin.twin_id == "fin-001"
        assert twin.system_type == "finance"

    def test_digital_twin_creation_manufacturing(self):
        """Test creating manufacturing digital twin."""
        twin = DigitalTwin(twin_id="mfg-001", system_type="manufacturing")
        assert twin.twin_id == "mfg-001"
        assert twin.system_type == "manufacturing"

    def test_digital_twin_invalid_type(self):
        """Test that invalid system type raises error."""
        with pytest.raises(ValueError, match="System type must be one of"):
            DigitalTwin(twin_id="invalid-001", system_type="invalid")

    def test_digital_twin_with_parameters(self):
        """Test digital twin with custom parameters."""
        params = {"model": "aircraft", "max_altitude": 40000}
        twin = DigitalTwin(twin_id="aero-002", system_type="aerospace", parameters=params)
        assert twin.parameters == params

    def test_update_state(self):
        """Test updating digital twin state."""
        twin = DigitalTwin(twin_id="test-001", system_type="aerospace")
        state = {"altitude": 10000, "velocity": 250.0}
        twin.update_state(state)
        assert twin.state_manager.get_current_state() == state

    def test_simulate_forward(self):
        """Test forward simulation."""
        twin = DigitalTwin(twin_id="test-002", system_type="aerospace")
        initial_state = {"altitude": 0, "velocity": 0}
        twin.update_state(initial_state)

        trajectory = twin.simulate_forward(time_steps=5, delta_t=1.0)
        assert len(trajectory) == 5
        assert all(isinstance(state, dict) for state in trajectory)

    def test_simulate_forward_zero_steps(self):
        """Test forward simulation with zero steps."""
        twin = DigitalTwin(twin_id="test-003", system_type="aerospace")
        trajectory = twin.simulate_forward(time_steps=0)
        assert len(trajectory) == 0

    def test_simulate_forward_different_delta_t(self):
        """Test forward simulation with different time steps."""
        twin = DigitalTwin(twin_id="test-004", system_type="pharma")
        twin.update_state({"concentration": 1.0})

        trajectory1 = twin.simulate_forward(time_steps=3, delta_t=0.5)
        trajectory2 = twin.simulate_forward(time_steps=3, delta_t=2.0)

        assert len(trajectory1) == 3
        assert len(trajectory2) == 3
        # Different delta_t can produce different results

    @pytest.mark.parametrize("system_type", ["aerospace", "pharma", "finance", "manufacturing"])
    def test_all_system_types_can_simulate(self, system_type):
        """Test that all system types can perform simulation."""
        twin = DigitalTwin(twin_id=f"test-{system_type}", system_type=system_type)
        twin.update_state({"value": 1.0})
        trajectory = twin.simulate_forward(time_steps=2)
        assert len(trajectory) == 2


class TestDigitalTwinIntegration:
    """Test integrated digital twin scenarios."""

    def test_aerospace_flight_simulation(self):
        """Test aerospace flight simulation scenario."""
        twin = DigitalTwin(
            twin_id="falcon-9", system_type="aerospace", parameters={"vehicle": "Falcon 9"}
        )
        initial_state = {"altitude_m": 0, "velocity_ms": 0, "fuel_kg": 400000}
        twin.update_state(initial_state)

        trajectory = twin.simulate_forward(time_steps=10, delta_t=1.0)
        assert len(trajectory) == 10
        # Verify trajectory is a valid sequence
        for state in trajectory:
            assert isinstance(state, dict)

    def test_pharma_drug_simulation(self):
        """Test pharmaceutical drug concentration simulation."""
        twin = DigitalTwin(
            twin_id="drug-001", system_type="pharma", parameters={"compound": "test-drug"}
        )
        twin.update_state({"concentration_mM": 10.0, "temperature_K": 310.0})

        trajectory = twin.simulate_forward(time_steps=5, delta_t=0.1)
        assert len(trajectory) == 5

    def test_finance_portfolio_simulation(self):
        """Test financial portfolio simulation."""
        twin = DigitalTwin(
            twin_id="portfolio-001",
            system_type="finance",
            parameters={"assets": ["AAPL", "GOOGL", "MSFT"]},
        )
        twin.update_state({"total_value": 1000000.0, "risk": 0.15})

        trajectory = twin.simulate_forward(time_steps=252, delta_t=1.0)  # 1 trading year
        assert len(trajectory) == 252

    def test_manufacturing_production_simulation(self):
        """Test manufacturing production simulation."""
        twin = DigitalTwin(
            twin_id="line-001", system_type="manufacturing", parameters={"product": "widget"}
        )
        twin.update_state({"units_produced": 0, "defect_rate": 0.01})

        trajectory = twin.simulate_forward(time_steps=24, delta_t=1.0)  # 24 hours
        assert len(trajectory) == 24
