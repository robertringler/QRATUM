"""Tests for visualization pipelines."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qubic.visualization.adapters.mesh import MeshAdapter
from qubic.visualization.adapters.timeseries import TimeSeriesAdapter
from qubic.visualization.pipelines.static import StaticPipeline
from qubic.visualization.pipelines.timeseries import TimeSeriesPipeline


@pytest.fixture
def sample_data():
    """Create sample visualization data."""

    adapter = MeshAdapter()
    return adapter.create_test_mesh("sphere", resolution=10)


@pytest.fixture
def timeseries_adapter():
    """Create time-series adapter with sample data."""

    adapter = TimeSeriesAdapter()
    adapter.create_synthetic_timeseries(n_steps=3)
    return adapter


class TestStaticPipeline:
    """Tests for StaticPipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""

        pipeline = StaticPipeline(backend="matplotlib")
        assert pipeline.backend_name == "matplotlib"

    def test_initialization_with_backends(self):
        """Test initialization with different backends."""

        for backend in ["matplotlib", "headless", "gpu"]:
            pipeline = StaticPipeline(backend=backend)
            assert pipeline.backend_name == backend

    def test_invalid_backend(self):
        """Test that invalid backend raises error."""

        with pytest.raises(ValueError):
            StaticPipeline(backend="invalid")

    def test_render(self, sample_data):
        """Test rendering."""

        pipeline = StaticPipeline(backend="headless")
        fig = pipeline.render(sample_data)

        assert fig is not None
        pipeline.close()

    def test_render_with_scalar_field(self, sample_data):
        """Test rendering with scalar field."""

        # Add scalar field
        sample_data.add_scalar_field("test_field", np.random.rand(len(sample_data.vertices)))

        pipeline = StaticPipeline(backend="headless")
        fig = pipeline.render(sample_data, scalar_field="test_field")

        assert fig is not None
        pipeline.close()

    def test_render_and_save(self, sample_data):
        """Test render and save workflow."""

        pipeline = StaticPipeline(backend="headless")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.png"
            pipeline.render_and_save(sample_data, output_path)

            assert output_path.exists()


class TestTimeSeriesPipeline:
    """Tests for TimeSeriesPipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""

        pipeline = TimeSeriesPipeline()
        assert pipeline.figsize == (10, 8)
        assert pipeline.dpi == 100

    def test_render_frames(self, timeseries_adapter):
        """Test rendering individual frames."""

        pipeline = TimeSeriesPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            frame_paths = pipeline.render_frames(adapter=timeseries_adapter, output_dir=output_dir)

            assert len(frame_paths) == 3
            for path in frame_paths:
                assert path.exists()

    @pytest.mark.slow
    def test_render_animation_gif(self, timeseries_adapter):
        """Test rendering GIF animation."""

        try:
            import imageio  # noqa: F401
        except ImportError:
            pytest.skip("imageio not installed")

        pipeline = TimeSeriesPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "animation.gif"
            pipeline.render_animation(
                adapter=timeseries_adapter,
                output_path=output_path,
                fps=5,
                format="gif",
            )

            assert output_path.exists()

    def test_render_animation_invalid_format(self, timeseries_adapter):
        """Test that invalid format raises error."""

        pipeline = TimeSeriesPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "animation.avi"

            with pytest.raises(ValueError):
                pipeline.render_animation(
                    adapter=timeseries_adapter,
                    output_path=output_path,
                    format="avi",
                )

    def test_render_animation_no_timesteps(self):
        """Test that rendering with no timesteps raises error."""

        adapter = TimeSeriesAdapter()  # Empty adapter
        pipeline = TimeSeriesPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "animation.gif"

            with pytest.raises(RuntimeError):
                pipeline.render_animation(adapter=adapter, output_path=output_path, format="gif")
