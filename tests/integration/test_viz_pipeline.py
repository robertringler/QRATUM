"""Integration tests for QuASIM to qubic-viz pipeline."""

import sys
from pathlib import Path

import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qubic-viz"))


def test_viz_pipeline_basic():
    """Test basic visualization pipeline."""

    from core.renderer import RenderConfig, SceneRenderer
    from engines.mesh_generator import TireMeshGenerator

    # Create components
    config = RenderConfig(width=100, height=100)
    renderer = SceneRenderer(config)
    mesh_gen = TireMeshGenerator(resolution=8)

    assert renderer is not None
    assert mesh_gen is not None


def test_tire_mesh_generation():
    """Test tire mesh generation."""

    from engines.mesh_generator import TireMeshGenerator

    class MockTireGeometry:
        outer_diameter_mm = 700.0
        width_mm = 225.0
        rim_diameter_inch = 17.0

    gen = TireMeshGenerator(resolution=8)
    geometry = MockTireGeometry()
    mesh = gen.generate_tire_mesh(geometry)

    assert mesh.num_vertices > 0
    assert mesh.num_faces > 0


@pytest.mark.skip(reason="Requires full QuASIM integration")
def test_quasim_to_viz_integration():
    """Test full QuASIM to visualization integration."""

    pass
