"""Integration tests for AHTC with platform modules.

Tests the integration of AHTC compression with:
- TensorNetworkEngine
- MultiQubitSimulator  
- StateVector
- Simulator
"""

import os
import tempfile
import numpy as np
import pytest


class TestTensorNetworkEngineIntegration:
    """Tests for TensorNetworkEngine AHTC integration."""

    def test_compress_state_basic(self):
        """Test basic compress_state functionality."""
        from quasim.qc.quasim_tn import TensorNetworkEngine

        engine = TensorNetworkEngine(num_qubits=8, backend='numpy')
        engine.initialize_state('zero')
        
        # Apply some gates
        engine.apply_gate('H', [0])
        engine.apply_gate('CNOT', [0, 1])
        
        # Compress state
        compressed = engine.compress_state(fidelity=0.995)
        
        assert 'compressed_state' in compressed
        assert 'fidelity' in compressed
        assert 'metadata' in compressed
        assert compressed['fidelity'] >= 0.995

    def test_compress_load_roundtrip(self):
        """Test full round-trip: compress_state -> load_compressed_state."""
        from quasim.qc.quasim_tn import TensorNetworkEngine

        engine = TensorNetworkEngine(num_qubits=6, backend='numpy')
        engine.initialize_state('zero')
        
        # Apply gates to create non-trivial state
        engine.apply_gate('H', [0])
        engine.apply_gate('CNOT', [0, 1])
        engine.apply_gate('CNOT', [1, 2])
        
        # Get original state
        original_state = engine.get_state_vector().copy()
        
        # Compress
        compressed = engine.compress_state(fidelity=0.99)
        
        # Load compressed
        engine.load_compressed_state(compressed)
        
        # Get recovered state
        recovered_state = engine.get_state_vector()
        
        # Check fidelity
        overlap = np.abs(np.vdot(original_state, recovered_state))
        fidelity = overlap ** 2
        
        assert fidelity >= 0.99

    def test_compress_state_different_fidelities(self):
        """Test compress_state with different fidelity targets."""
        from quasim.qc.quasim_tn import TensorNetworkEngine

        engine = TensorNetworkEngine(num_qubits=6, backend='numpy')
        engine.initialize_state('zero')
        engine.apply_gate('H', [0])
        
        fidelities = [0.95, 0.99, 0.995]
        
        for target_fidelity in fidelities:
            compressed = engine.compress_state(fidelity=target_fidelity)
            assert compressed['fidelity'] >= target_fidelity


class TestMultiQubitSimulatorIntegration:
    """Tests for MultiQubitSimulator AHTC integration."""

    def test_checkpoint_state_compressed(self):
        """Test checkpoint_state with compression enabled."""
        from quasim.qc.quasim_multi import MultiQubitSimulator

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.npz')
            
            sim = MultiQubitSimulator(num_qubits=8, seed=42)
            sim.initialize_state('zero')
            sim.apply_gate('H', [0])
            sim.apply_gate('CNOT', [0, 1])
            
            # Checkpoint with compression
            metadata = sim.checkpoint_state(checkpoint_path, compress=True)
            
            assert os.path.exists(checkpoint_path)
            assert metadata['compressed'] is True
            assert 'fidelity' in metadata

    def test_checkpoint_state_uncompressed(self):
        """Test checkpoint_state without compression."""
        from quasim.qc.quasim_multi import MultiQubitSimulator

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.npz')
            
            sim = MultiQubitSimulator(num_qubits=6, seed=42)
            sim.initialize_state('zero')
            sim.apply_gate('H', [0])
            
            # Checkpoint without compression
            metadata = sim.checkpoint_state(checkpoint_path, compress=False)
            
            assert os.path.exists(checkpoint_path)
            assert metadata['compressed'] is False
            assert 'state' in metadata

    def test_restore_checkpoint_compressed(self):
        """Test restoring from compressed checkpoint."""
        from quasim.qc.quasim_multi import MultiQubitSimulator

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.npz')
            
            # Create and save state
            sim1 = MultiQubitSimulator(num_qubits=8, seed=42)
            sim1.initialize_state('zero')
            sim1.apply_gate('H', [0])
            sim1.apply_gate('CNOT', [0, 1])
            original_state = sim1.state.copy()
            
            sim1.checkpoint_state(checkpoint_path, compress=True)
            
            # Restore in new simulator
            sim2 = MultiQubitSimulator(num_qubits=8, seed=42)
            sim2.restore_checkpoint(checkpoint_path)
            
            # Check fidelity
            overlap = np.abs(np.vdot(original_state, sim2.state))
            fidelity = overlap ** 2
            
            assert fidelity >= 0.99

    def test_restore_checkpoint_uncompressed(self):
        """Test restoring from uncompressed checkpoint."""
        from quasim.qc.quasim_multi import MultiQubitSimulator

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.npz')
            
            # Create and save state
            sim1 = MultiQubitSimulator(num_qubits=6, seed=42)
            sim1.initialize_state('zero')
            sim1.apply_gate('H', [0])
            original_state = sim1.state.copy()
            
            sim1.checkpoint_state(checkpoint_path, compress=False)
            
            # Restore in new simulator
            sim2 = MultiQubitSimulator(num_qubits=6, seed=42)
            sim2.restore_checkpoint(checkpoint_path)
            
            # Check exact match (no compression)
            assert np.allclose(original_state, sim2.state)

    def test_checkpoint_json_metadata(self):
        """Test that JSON metadata is correctly written."""
        from quasim.qc.quasim_multi import MultiQubitSimulator

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.npz')
            json_path = checkpoint_path.replace('.npz', '_metadata.json')
            
            sim = MultiQubitSimulator(num_qubits=6, seed=42)
            sim.initialize_state('zero')
            sim.checkpoint_state(checkpoint_path, compress=True)
            
            # Check JSON metadata exists and is readable
            assert os.path.exists(json_path)
            
            import json
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            assert 'num_qubits' in metadata
            assert 'compressed' in metadata
            assert metadata['compressed'] is True


class TestStateVectorIntegration:
    """Tests for StateVector AHTC integration."""

    def test_statevector_compress_basic(self):
        """Test StateVector.compress() basic functionality."""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "statevector", 
            "/home/runner/work/QRATUM/QRATUM/qratum_ai_platform/core/statevector.py"
        )
        statevector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(statevector)
        
        StateVector = statevector.StateVector
        
        sv = StateVector.random_state(8, seed=42)
        compressed = sv.compress(fidelity=0.995)
        
        assert compressed.num_qubits == 8
        assert compressed.fidelity >= 0.995
        assert 'compression_ratio' in compressed.metadata

    def test_statevector_compress_decompress_roundtrip(self):
        """Test StateVector compress -> from_compressed roundtrip."""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "statevector", 
            "/home/runner/work/QRATUM/QRATUM/qratum_ai_platform/core/statevector.py"
        )
        statevector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(statevector)
        
        StateVector = statevector.StateVector
        
        sv = StateVector.random_state(10, seed=123)
        original_data = sv.data.copy()
        
        # Compress
        compressed = sv.compress(fidelity=0.995)
        
        # Decompress
        recovered = StateVector.from_compressed(compressed)
        
        # Check fidelity
        overlap = np.abs(np.vdot(original_data, recovered.data))
        fidelity = overlap ** 2
        
        assert fidelity >= 0.995

    def test_statevector_different_fidelities(self):
        """Test StateVector compression with different fidelity targets."""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "statevector", 
            "/home/runner/work/QRATUM/QRATUM/qratum_ai_platform/core/statevector.py"
        )
        statevector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(statevector)
        
        StateVector = statevector.StateVector
        
        sv = StateVector.random_state(8, seed=456)
        
        for target in [0.95, 0.99, 0.995]:
            compressed = sv.compress(fidelity=target)
            assert compressed.fidelity >= target

    def test_compressed_statevector_decompress_method(self):
        """Test CompressedStateVector.decompress() method."""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "statevector", 
            "/home/runner/work/QRATUM/QRATUM/qratum_ai_platform/core/statevector.py"
        )
        statevector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(statevector)
        
        StateVector = statevector.StateVector
        
        sv = StateVector.random_state(8, seed=789)
        compressed = sv.compress(fidelity=0.99)
        
        # Use decompress method
        recovered = compressed.decompress()
        
        assert isinstance(recovered, StateVector)
        assert recovered.num_qubits == 8


class TestSimulatorIntegration:
    """Tests for Simulator AHTC integration."""

    @pytest.mark.skip(reason="Requires Circuit class which may not be available")
    def test_run_with_compression_basic(self):
        """Test Simulator.run_with_compression() basic functionality."""
        # This test is skipped because it requires the Circuit class
        # which may have dependencies that aren't available
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
