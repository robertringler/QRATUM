"""Unit tests for CAD exporter."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


class TestCADExporter:
    """Tests for CADExporter class."""

    def test_supported_formats(self):
        """Test that supported formats are available."""
        from quasic_viz.exporters.cad_exporter import SUPPORTED_FORMATS

        assert "obj" in SUPPORTED_FORMATS
        assert "glb" in SUPPORTED_FORMATS
        assert "fbx" in SUPPORTED_FORMATS
        assert "step" in SUPPORTED_FORMATS

    def test_get_supported_formats(self):
        """Test get_supported_formats method."""
        from quasic_viz.exporters.cad_exporter import CADExporter

        formats = CADExporter.get_supported_formats()
        assert isinstance(formats, list)
        assert "obj" in formats

    def test_export_obj_fallback(self):
        """Test OBJ export without trimesh."""
        from quasic_viz.engines.tire_mesh_generator import TireMeshGenerator
        from quasic_viz.exporters.cad_exporter import CADExporter

        # Generate test mesh
        gen = TireMeshGenerator(resolution=8)
        mesh = gen.generate_tire_mesh()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.obj"
            CADExporter.export_mesh(mesh, output_path, format="obj")
            assert output_path.exists()

            # Verify file content
            content = output_path.read_text()
            assert "v " in content  # Vertices
            assert "f " in content  # Faces

    def test_export_invalid_format(self):
        """Test export with invalid format raises error."""
        from quasic_viz.exporters.cad_exporter import CADExporter

        with pytest.raises(ValueError):
            CADExporter.export_mesh({"vertices": [], "faces": []}, "test.xyz", format="xyz")
