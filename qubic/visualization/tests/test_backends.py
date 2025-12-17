"""Tests for visualization backends."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qubic.visualization.adapters.mesh import MeshAdapter
from qubic.visualization.backends.gpu_backend import GPUBackend
from qubic.visualization.backends.headless_backend import HeadlessBackend
from qubic.visualization.backends.matplotlib_backend import MatplotlibBackend


@pytest.fixture
def sample_data():
    """Create sample visualization data for testing."""

    adapter = MeshAdapter()
    return adapter.create_test_mesh("sphere", resolution=10)


class TestMatplotlibBackend:
    """Tests for MatplotlibBackend."""

    def test_initialization(self):
        """Test backend initialization."""

        backend = MatplotlibBackend(figsize=(10, 8), dpi=100)
        assert backend.figsize == (10, 8)
        assert backend.dpi == 100

    def test_render(self, sample_data):
        """Test rendering visualization data."""

        backend = MatplotlibBackend()
        fig = backend.render(sample_data)

        assert fig is not None
        assert backend.fig is not None
        assert backend.ax is not None

        backend.close()

    def test_render_with_scalar_field(self, sample_data):
        """Test rendering with scalar field color mapping."""

        backend = MatplotlibBackend()

        # Add a scalar field
        sample_data.add_scalar_field("test_field", np.random.rand(len(sample_data.vertices)))

        fig = backend.render(sample_data, scalar_field="test_field")
        assert fig is not None

        backend.close()

    def test_save(self, sample_data):
        """Test saving rendered figure."""

        backend = MatplotlibBackend()
        backend.render(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.png"
            backend.save(output_path)

            assert output_path.exists()

    def test_save_without_render(self):
        """Test save raises error without prior render."""

        backend = MatplotlibBackend()

        with pytest.raises(RuntimeError):
            backend.save(Path("output.png"))


class TestHeadlessBackend:
    """Tests for HeadlessBackend."""

    def test_initialization(self):
        """Test headless backend initialization."""

        backend = HeadlessBackend(figsize=(10, 8), dpi=100)
        assert backend.figsize == (10, 8)
        assert backend.dpi == 100

    def test_render(self, sample_data):
        """Test rendering in headless mode."""

        backend = HeadlessBackend()
        fig = backend.render(sample_data)

        assert fig is not None
        backend.close()

    def test_show_raises_error(self, sample_data):
        """Test that show() raises error in headless mode."""

        backend = HeadlessBackend()
        backend.render(sample_data)

        with pytest.raises(RuntimeError):
            backend.show()

        backend.close()

    def test_save_headless(self, sample_data):
        """Test saving in headless mode."""

        backend = HeadlessBackend()
        backend.render(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "headless_output.png"
            backend.save(output_path)

            assert output_path.exists()


class TestGPUBackend:
    """Tests for GPUBackend."""

    def test_initialization(self):
        """Test GPU backend initialization."""

        backend = GPUBackend(figsize=(10, 8), dpi=100)
        assert backend.figsize == (10, 8)
        assert backend.dpi == 100

    def test_force_cpu_fallback(self):
        """Test forcing CPU fallback."""

        backend = GPUBackend(force_cpu=True)
        assert backend.use_gpu is False

    def test_render(self, sample_data):
        """Test rendering with GPU backend (or CPU fallback)."""

        backend = GPUBackend()
        fig = backend.render(sample_data)

        assert fig is not None
        backend.close()

    def test_render_with_scalar_field(self, sample_data):
        """Test rendering with scalar field."""

        backend = GPUBackend()

        sample_data.add_scalar_field("test_field", np.random.rand(len(sample_data.vertices)))

        fig = backend.render(sample_data, scalar_field="test_field")
        assert fig is not None

        backend.close()

    def test_save(self, sample_data):
        """Test saving with GPU backend."""

        backend = GPUBackend()
        backend.render(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gpu_output.png"
            backend.save(output_path)

            assert output_path.exists()


class TestBackendComparison:
    """Tests comparing different backends."""

    def test_all_backends_produce_output(self, sample_data):
        """Test that all backends can produce output."""

        backends = [
            MatplotlibBackend(),
            HeadlessBackend(),
            GPUBackend(force_cpu=True),
        ]

        for backend in backends:
            fig = backend.render(sample_data)
            assert fig is not None

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.png"
                backend.save(output_path)
                assert output_path.exists()

            backend.close()
