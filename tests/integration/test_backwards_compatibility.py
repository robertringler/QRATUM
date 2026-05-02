"""Backwards compatibility tests for QRATUM platform integration.

Verifies that all existing quasim.* and qratum.* imports continue to work
after platform integration layer is added.
"""

import pytest


class TestBackwardsCompatibility:
    """Test suite for backwards compatibility."""

    def test_existing_qratum_imports_work(self):
        """Verify all existing qratum.* imports still function."""
        # Core imports that existed before platform integration
        from qratum import (Circuit, DensityMatrix, Measurement, QRATUMConfig,
                            Result, Simulator, StateVector, __version__, gates,
                            get_config, reset_config, set_config)

        # Verify no import errors
        assert Circuit is not None
        assert Simulator is not None
        assert StateVector is not None
        assert DensityMatrix is not None
        assert Measurement is not None
        assert Result is not None
        assert gates is not None

        # Verify config functions work
        assert QRATUMConfig is not None
        assert get_config is not None
        assert set_config is not None
        assert reset_config is not None
        assert __version__ is not None

    def test_existing_qratum_core_imports_work(self):
        """Verify qratum.core imports still work."""
        from qratum.core.circuit import Circuit
        from qratum.core.simulator import Simulator
        from qratum.core.statevector import StateVector

        assert Circuit is not None
        assert Simulator is not None
        assert StateVector is not None

    def test_existing_qratum_config_works(self):
        """Verify existing QRATUMConfig still works."""
        from qratum import QRATUMConfig, get_config, reset_config, set_config

        # Create config
        config = QRATUMConfig(backend="cpu", precision="fp32", seed=42)
        assert config.backend == "cpu"
        assert config.precision == "fp32"
        assert config.seed == 42

        # Set global config
        set_config(config)
        global_config = get_config()
        assert global_config.seed == 42

        # Reset config
        reset_config()
        new_config = get_config()
        assert new_config is not None

    def test_existing_quasim_quantum_imports_work(self):
        """Verify quasim.quantum imports still work."""
        try:
            from quasim.quantum.core import QuantumBackend, QuantumConfig

            assert QuantumBackend is not None
            assert QuantumConfig is not None
        except ImportError:
            # Optional dependencies may not be installed
            pytest.skip("quasim.quantum requires optional dependencies")

    def test_existing_quasim_opt_imports_work(self):
        """Verify quasim.opt imports still work."""
        try:
            from quasim.opt.optimizer import Optimizer

            assert Optimizer is not None
        except (ImportError, AttributeError):
            # Module may have different structure
            pytest.skip("quasim.opt structure may vary")

    def test_new_platform_imports_work(self):
        """Verify new platform imports work alongside existing imports."""
        # Old imports
        # New imports
        from qratum import (PlatformConfig, QRATUMConfig, QRATUMPlatform,
                            Simulator, create_platform)

        # Both should work
        assert QRATUMConfig is not None
        assert Simulator is not None
        assert PlatformConfig is not None
        assert QRATUMPlatform is not None
        assert create_platform is not None

    def test_old_and_new_configs_coexist(self):
        """Verify old and new config classes coexist."""
        from qratum import PlatformConfig, QRATUMConfig

        # Create old config
        old_config = QRATUMConfig(backend="cpu", seed=42)
        assert old_config.backend == "cpu"
        assert old_config.seed == 42

        # Create new config
        new_config = PlatformConfig(quantum_backend="simulator", seed=42)
        assert new_config.quantum_backend == "simulator"
        assert new_config.seed == 42

        # Both should be independent
        assert old_config is not new_config

    def test_existing_simulator_still_works(self):
        """Verify existing Simulator class still works."""
        from qratum import Simulator

        # Should be able to create simulator
        sim = Simulator()
        assert sim is not None

    def test_no_breaking_changes_in_exports(self):
        """Verify __all__ exports include old and new symbols."""
        import qratum

        # Old exports should still be present
        old_exports = [
            "QRATUMConfig",
            "get_config",
            "set_config",
            "reset_config",
            "Simulator",
            "Circuit",
            "StateVector",
            "Measurement",
            "Result",
            "DensityMatrix",
            "gates",
        ]

        for export in old_exports:
            assert export in qratum.__all__, f"Missing old export: {export}"

        # New exports should be present
        new_exports = [
            "PlatformConfig",
            "QRATUMPlatform",
            "create_platform",
        ]

        for export in new_exports:
            assert export in qratum.__all__, f"Missing new export: {export}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
