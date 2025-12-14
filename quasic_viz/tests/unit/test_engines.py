"""Unit tests for multi-GPU renderer."""


class TestMultiGPURenderer:
    """Tests for MultiGPURenderer class."""

    def test_init_no_device_ids(self):
        """Test initialization without explicit device IDs."""
        from quasic_viz.engines.multi_gpu_renderer import MultiGPURenderer

        renderer = MultiGPURenderer()
        assert renderer.device_ids is not None
        assert renderer.num_devices >= 0

    def test_init_with_device_ids(self):
        """Test initialization with explicit device IDs."""
        from quasic_viz.engines.multi_gpu_renderer import MultiGPURenderer

        renderer = MultiGPURenderer(device_ids=[0])
        assert renderer.device_ids == [0]
        assert renderer.num_devices == 1

    def test_is_available(self):
        """Test GPU availability check."""
        from quasic_viz.engines.multi_gpu_renderer import MultiGPURenderer

        renderer = MultiGPURenderer()
        # Returns bool regardless of GPU availability
        assert isinstance(renderer.is_available(), bool)

    def test_get_device_info(self):
        """Test device info retrieval."""
        from quasic_viz.engines.multi_gpu_renderer import MultiGPURenderer

        renderer = MultiGPURenderer()
        info = renderer.get_device_info()
        assert isinstance(info, list)
        assert len(info) >= 1


class TestARAdapter:
    """Tests for ARAdapter class."""

    def test_init(self):
        """Test AR adapter initialization."""
        from quasic_viz.engines.ar_adapter import ARAdapter

        adapter = ARAdapter(ws_url="ws://localhost:8000/ws")
        assert adapter.ws_url == "ws://localhost:8000/ws"
        assert adapter.is_connected is False


class TestTireMeshGenerator:
    """Tests for TireMeshGenerator class."""

    def test_init(self):
        """Test mesh generator initialization."""
        from quasic_viz.engines.tire_mesh_generator import TireMeshGenerator

        gen = TireMeshGenerator(resolution=16)
        assert gen.resolution == 16

    def test_generate_tire_mesh(self):
        """Test tire mesh generation."""
        from quasic_viz.engines.tire_mesh_generator import TireMeshGenerator

        gen = TireMeshGenerator(resolution=16)
        mesh = gen.generate_tire_mesh()

        assert "vertices" in mesh
        assert "faces" in mesh
        assert "normals" in mesh
        assert mesh["num_vertices"] > 0
        assert mesh["num_faces"] > 0
