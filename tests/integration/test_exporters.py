"""Integration tests for CAD exporters."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qubic-viz"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qubic-design-studio"))


def test_obj_exporter():
    """Test OBJ exporter."""

    from engines.mesh_generator import TireMeshGenerator
    from exporters.obj_exporter import OBJExporter

    class MockTireGeometry:
        outer_diameter_mm = 700.0
        width_mm = 225.0
        rim_diameter_inch = 17.0

    # Generate mesh
    gen = TireMeshGenerator(resolution=8)
    mesh = gen.generate_tire_mesh(MockTireGeometry())

    # Export
    exporter = OBJExporter()
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.obj"
        exporter.export_mesh(mesh, output_path)
        assert output_path.exists()


def test_gltf_exporter():
    """Test glTF exporter."""

    from engines.mesh_generator import TireMeshGenerator
    from exporters.gltf_exporter import GLTFExporter

    class MockTireGeometry:
        outer_diameter_mm = 700.0
        width_mm = 225.0
        rim_diameter_inch = 17.0

    # Generate mesh
    gen = TireMeshGenerator(resolution=8)
    mesh = gen.generate_tire_mesh(MockTireGeometry())

    # Export
    exporter = GLTFExporter()
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.gltf"
        exporter.export_mesh(mesh, output_path)
        assert output_path.exists()
