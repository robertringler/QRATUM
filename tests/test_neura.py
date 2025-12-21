"""Tests for NEURA Neuroscience & BCI module."""

from qratum_platform.core import PlatformContract, PlatformIntent, VerticalModule, ComputeSubstrate
from verticals.neura import NEURAModule


class TestNEURAModule:
    """Test NEURA Neuroscience & BCI module."""

    def test_module_metadata(self):
        """Test module has required metadata."""
        module = NEURAModule()
        assert module.MODULE_NAME == "NEURA"
        assert module.MODULE_VERSION == "2.0.0"

    def test_neural_simulation(self):
        """Test spiking neural network simulation."""
        module = NEURAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.NEURA,
            operation="neural_simulation",
            parameters={"num_neurons": 100, "duration_ms": 1000},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_001",
            substrate=ComputeSubstrate.CEREBRAS,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["simulation_type"] == "spiking_neural_network"
        assert "spike_trains" in result

    def test_eeg_analysis(self):
        """Test EEG signal analysis."""
        module = NEURAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.NEURA,
            operation="eeg_analysis",
            parameters={"sampling_rate_hz": 256, "duration_s": 60},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_002",
            substrate=ComputeSubstrate.MI300X,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["analysis_type"] == "eeg_spectral"
        assert "band_power" in result

    def test_bci_processing(self):
        """Test BCI signal processing."""
        module = NEURAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.NEURA,
            operation="bci_processing",
            parameters={"task_type": "motor_imagery", "channels": 8},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_003",
            substrate=ComputeSubstrate.IPU,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["processing_type"] == "bci_classification"
        assert "detected_intent" in result
