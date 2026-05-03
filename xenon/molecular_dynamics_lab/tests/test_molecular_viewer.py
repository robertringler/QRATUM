"""Tests for the Molecular Viewer."""

from __future__ import annotations

from xenon.molecular_dynamics_lab.core.molecular_viewer import (
    Label,
    MolecularViewer,
    Selection,
    StyleSpec,
    ViewerConfig,
)


class TestViewerConfig:
    """Test ViewerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ViewerConfig()

        assert config.width == 800
        assert config.height == 600
        assert config.background_color == "#000000"
        assert config.ambient_light == 0.4
        assert config.webxr_enabled is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = ViewerConfig(
            width=1024,
            height=768,
            background_color="#ffffff",
            webxr_enabled=True,
        )

        assert config.width == 1024
        assert config.height == 768
        assert config.background_color == "#ffffff"
        assert config.webxr_enabled is True


class TestStyleSpec:
    """Test StyleSpec dataclass."""

    def test_default_style(self):
        """Test default style values."""
        style = StyleSpec()

        assert style.style == "cartoon"
        assert style.color == "spectrum"
        assert style.opacity == 1.0

    def test_custom_style(self):
        """Test custom style specification."""
        style = StyleSpec(
            style="stick",
            color="chainHetatm",
            opacity=0.8,
        )

        assert style.style == "stick"
        assert style.color == "chainHetatm"
        assert style.opacity == 0.8

    def test_to_3dmol(self):
        """Test conversion to 3Dmol.js format."""
        style = StyleSpec(style="sphere", color="Jmol", opacity=0.5)
        mol_style = style.to_3dmol()

        assert "sphere" in mol_style
        assert mol_style["sphere"]["colorscheme"] == "Jmol"
        assert mol_style["sphere"]["opacity"] == 0.5


class TestSelection:
    """Test Selection dataclass."""

    def test_empty_selection(self):
        """Test empty selection (all atoms)."""
        sel = Selection()
        mol_sel = sel.to_3dmol()

        assert mol_sel == {}

    def test_chain_selection(self):
        """Test chain selection."""
        sel = Selection(chain="A")
        mol_sel = sel.to_3dmol()

        assert mol_sel["chain"] == "A"

    def test_residue_selection(self):
        """Test residue selection."""
        sel = Selection(residue_name="ALA")
        mol_sel = sel.to_3dmol()

        assert mol_sel["resn"] == "ALA"

    def test_element_selection(self):
        """Test element selection."""
        sel = Selection(element="C")
        mol_sel = sel.to_3dmol()

        assert mol_sel["elem"] == "C"

    def test_combined_selection(self):
        """Test combined selection criteria."""
        sel = Selection(chain="A", residue_range=(10, 50), element="CA")
        mol_sel = sel.to_3dmol()

        assert mol_sel["chain"] == "A"
        assert mol_sel["resi"] == "10-50"
        assert mol_sel["elem"] == "CA"


class TestLabel:
    """Test Label dataclass."""

    def test_label_creation(self):
        """Test creating a label."""
        label = Label(
            text="Active Site",
            position=(10.0, 20.0, 30.0),
            font_size=14,
            font_color="#ffffff",
        )

        assert label.text == "Active Site"
        assert label.position == (10.0, 20.0, 30.0)
        assert label.font_size == 14


class TestMolecularViewer:
    """Test MolecularViewer class."""

    def test_viewer_creation(self):
        """Test creating a viewer."""
        config = ViewerConfig(width=640, height=480)
        viewer = MolecularViewer(config)

        assert viewer.config.width == 640
        assert viewer.config.height == 480

    def test_add_style(self):
        """Test adding styles."""
        viewer = MolecularViewer()

        viewer.add_style(
            Selection(chain="A"),
            StyleSpec(style="cartoon", color="spectrum"),
        )

        assert len(viewer._styles) == 1

    def test_add_label(self):
        """Test adding labels."""
        viewer = MolecularViewer()

        viewer.add_label(
            text="Test Label",
            position=(0, 0, 0),
        )

        assert len(viewer._labels) == 1

    def test_clear_styles(self):
        """Test clearing styles."""
        viewer = MolecularViewer()
        viewer.add_style(Selection(), StyleSpec())
        viewer.add_style(Selection(), StyleSpec())

        viewer.clear_styles()

        assert len(viewer._styles) == 0

    def test_generate_js(self):
        """Test JavaScript generation."""
        viewer = MolecularViewer(ViewerConfig(width=800, height=600))
        viewer.add_style(Selection(), StyleSpec(style="cartoon"))

        js = viewer.generate_js()

        assert "3Dmol" in js or "$3Dmol" in js
        assert "cartoon" in js

    def test_generate_html(self):
        """Test HTML generation."""
        viewer = MolecularViewer(ViewerConfig(webxr_enabled=True))

        html = viewer.generate_html()

        assert "<html" in html
        assert "3Dmol" in html
        assert "</html>" in html

    def test_generate_full_page(self):
        """Test full page HTML generation."""
        viewer = MolecularViewer()

        html = viewer.generate_full_page(title="Test Viewer")

        assert "<title>Test Viewer</title>" in html
        assert "<!DOCTYPE html>" in html


class TestViewerIntegration:
    """Integration tests for the viewer."""

    def test_complete_workflow(self):
        """Test complete viewer workflow."""
        # Create viewer
        config = ViewerConfig(
            width=1024,
            height=768,
            background_color="#1a1a2e",
            webxr_enabled=True,
        )
        viewer = MolecularViewer(config)

        # Add multiple styles
        viewer.add_style(
            Selection(chain="A"),
            StyleSpec(style="cartoon", color="ssPyMOL"),
        )
        viewer.add_style(
            Selection(chain="B"),
            StyleSpec(style="stick", color="greenCarbon"),
        )

        # Add labels
        viewer.add_label(
            text="Binding Site",
            position=(10, 20, 30),
            font_size=12,
        )

        # Generate output
        html = viewer.generate_full_page()

        assert "cartoon" in html
        assert "stick" in html
        assert "Binding Site" in html
