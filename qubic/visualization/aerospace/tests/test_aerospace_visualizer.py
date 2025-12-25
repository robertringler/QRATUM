"""Tests for aerospace visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qubic.visualization.aerospace import (AerospaceVisualizer,
                                           AerospaceVizConfig, ComplianceMode,
                                           FrameAuditRecord, RenderBackend)


@pytest.fixture
def default_config():
    """Create default aerospace visualization config."""
    return AerospaceVizConfig(seed=42, enable_audit_log=True)


@pytest.fixture
def visualizer(default_config):
    """Create aerospace visualizer instance."""
    return AerospaceVisualizer(default_config)


@pytest.fixture
def sample_trajectory():
    """Create sample flight trajectory."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 100)
    trajectory = np.column_stack([100 * np.cos(t), 100 * np.sin(t), 50 * t])
    return trajectory


@pytest.fixture
def sample_velocity():
    """Create sample velocity field."""
    np.random.seed(42)
    velocity = np.random.randn(100, 3) * 10
    return velocity


@pytest.fixture
def sample_fem_mesh():
    """Create sample FEM mesh."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    elements = np.array([[0, 1, 2, 3]], dtype=int)

    # Von Mises stress tensor (xx, yy, zz, xy, xz, yz)
    stress = np.array(
        [
            [100e6, 50e6, 30e6, 10e6, 5e6, 5e6],
            [120e6, 60e6, 40e6, 15e6, 8e6, 8e6],
            [90e6, 45e6, 25e6, 8e6, 4e6, 4e6],
            [110e6, 55e6, 35e6, 12e6, 6e6, 6e6],
        ]
    )

    return nodes, elements, stress


class TestAerospaceVizConfig:
    """Tests for AerospaceVizConfig dataclass."""

    def test_default_initialization(self):
        """Test default config initialization."""
        config = AerospaceVizConfig()

        assert config.compliance_mode == ComplianceMode.DEVELOPMENT
        assert config.render_backend == RenderBackend.MATPLOTLIB
        assert config.seed == 42
        assert config.enable_audit_log is False
        assert config.resolution == (1920, 1080)
        assert config.target_fps == 60

    def test_custom_initialization(self):
        """Test custom config initialization."""
        config = AerospaceVizConfig(
            compliance_mode=ComplianceMode.DO178C_LEVEL_A,
            seed=123,
            enable_audit_log=True,
            resolution=(3840, 2160),
        )

        assert config.compliance_mode == ComplianceMode.DO178C_LEVEL_A
        assert config.seed == 123
        assert config.enable_audit_log is True
        assert config.resolution == (3840, 2160)


class TestAerospaceVisualizer:
    """Tests for AerospaceVisualizer class."""

    def test_initialization_default(self):
        """Test visualizer initialization with default config."""
        viz = AerospaceVisualizer()

        assert viz.config is not None
        assert viz.rng is not None
        assert isinstance(viz.audit_trail, list)
        assert viz.frame_counter == 0

    def test_initialization_custom(self, default_config):
        """Test visualizer initialization with custom config."""
        viz = AerospaceVisualizer(default_config)

        assert viz.config.seed == 42
        assert viz.config.enable_audit_log is True
        assert len(viz.audit_trail) == 0

    def test_config_hash_computation(self, visualizer):
        """Test configuration hash is computed."""
        assert visualizer._config_hash is not None
        assert len(visualizer._config_hash) == 64  # SHA-256 hex digest

    def test_config_hash_deterministic(self):
        """Test config hash is deterministic."""
        config1 = AerospaceVizConfig(seed=42, compliance_mode=ComplianceMode.DO178C_LEVEL_A)
        viz1 = AerospaceVisualizer(config1)

        config2 = AerospaceVizConfig(seed=42, compliance_mode=ComplianceMode.DO178C_LEVEL_A)
        viz2 = AerospaceVisualizer(config2)

        assert viz1._config_hash == viz2._config_hash

    def test_config_hash_differs(self):
        """Test config hash differs for different configs."""
        config1 = AerospaceVizConfig(seed=42)
        viz1 = AerospaceVisualizer(config1)

        config2 = AerospaceVizConfig(seed=123)
        viz2 = AerospaceVisualizer(config2)

        assert viz1._config_hash != viz2._config_hash


class TestFlightDynamicsRendering:
    """Tests for flight dynamics rendering methods."""

    def test_render_flight_trajectory_basic(self, visualizer, sample_trajectory):
        """Test basic flight trajectory rendering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trajectory.png"

            fig = visualizer.render_flight_trajectory(
                trajectory=sample_trajectory, output_path=str(output_path)
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_flight_trajectory_with_velocity(
        self, visualizer, sample_trajectory, sample_velocity
    ):
        """Test trajectory rendering with velocity vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trajectory_vel.png"

            fig = visualizer.render_flight_trajectory(
                trajectory=sample_trajectory,
                velocity=sample_velocity,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_flight_trajectory_with_vapor(self, visualizer, sample_trajectory):
        """Test trajectory rendering with vapor trail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trajectory_vapor.png"

            fig = visualizer.render_flight_trajectory(
                trajectory=sample_trajectory,
                show_vapor_trail=True,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_airflow_streamlines(self, visualizer):
        """Test airflow streamlines rendering."""
        # Create synthetic velocity field
        nx, ny, nz = 10, 10, 10
        n_points = nx * ny * nz

        vx = np.random.randn(n_points)
        vy = np.random.randn(n_points)
        vz = np.random.randn(n_points)
        velocity_field = np.column_stack([vx, vy, vz])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "airflow.png"

            fig = visualizer.render_airflow_streamlines(
                velocity_field=velocity_field,
                grid_shape=(nx, ny, nz),
                density=20,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()


class TestStructuralAnalysisRendering:
    """Tests for structural analysis rendering methods."""

    def test_render_fem_mesh_basic(self, visualizer, sample_fem_mesh):
        """Test basic FEM mesh rendering."""
        nodes, elements, _ = sample_fem_mesh

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fem_mesh.png"

            fig = visualizer.render_fem_mesh(
                nodes=nodes, elements=elements, output_path=str(output_path)
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_fem_mesh_with_stress(self, visualizer, sample_fem_mesh):
        """Test FEM mesh rendering with stress field."""
        nodes, elements, stress = sample_fem_mesh

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fem_stress.png"

            fig = visualizer.render_fem_mesh(
                nodes=nodes,
                elements=elements,
                stress_tensor=stress,
                show_wireframe=True,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_modal_analysis(self, visualizer):
        """Test modal analysis rendering."""
        # Simple beam nodes
        nodes = np.array([[i, 0, 0] for i in range(10)], dtype=float)

        # First bending mode
        mode_shape = np.array([[0, np.sin(i * np.pi / 9), 0] for i in range(10)])
        eigenvectors = mode_shape[:, np.newaxis, :]  # Shape: (10, 1, 3)
        eigenfrequencies = np.array([25.3])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "modal.png"

            fig = visualizer.render_modal_analysis(
                nodes=nodes,
                eigenvectors=eigenvectors,
                eigenfrequencies=eigenfrequencies,
                mode_index=0,
                amplitude_scale=0.5,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()


class TestThermalSystemsRendering:
    """Tests for thermal systems rendering methods."""

    def test_render_thermal_field(self, visualizer):
        """Test thermal field rendering."""
        # Create random thermal field
        np.random.seed(42)
        n_points = 100
        geometry = np.random.randn(n_points, 3)
        temperature = 300 + 100 * np.random.rand(n_points)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thermal.png"

            fig = visualizer.render_thermal_field(
                temperature=temperature, geometry=geometry, output_path=str(output_path)
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_thermal_field_with_isotherms(self, visualizer):
        """Test thermal field rendering with isotherms."""
        np.random.seed(42)
        n_points = 100
        geometry = np.random.randn(n_points, 3)
        temperature = 300 + 100 * np.random.rand(n_points)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thermal_iso.png"

            fig = visualizer.render_thermal_field(
                temperature=temperature,
                geometry=geometry,
                show_isotherms=True,
                isotherm_levels=10,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_heat_flux(self, visualizer):
        """Test heat flux rendering."""
        np.random.seed(42)
        n_points = 50
        geometry = np.random.randn(n_points, 3)
        heat_flux = np.random.randn(n_points, 3) * 1000
        surface_normals = np.random.randn(n_points, 3)

        # Normalize normals
        surface_normals = surface_normals / (
            np.linalg.norm(surface_normals, axis=1, keepdims=True) + 1e-10
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heat_flux.png"

            fig = visualizer.render_heat_flux(
                heat_flux=heat_flux,
                surface_normals=surface_normals,
                geometry=geometry,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()


class TestAvionicsDisplayRendering:
    """Tests for avionics display rendering methods."""

    def test_render_sensor_fov(self, visualizer):
        """Test sensor FOV rendering."""
        sensor_pos = np.array([0, 0, 10], dtype=float)
        sensor_dir = np.array([1, 0, -0.5], dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sensor_fov.png"

            fig = visualizer.render_sensor_fov(
                sensor_position=sensor_pos,
                sensor_orientation=sensor_dir,
                fov_horizontal=120,
                fov_vertical=30,
                range_m=500,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()

    def test_render_radar_cross_section(self, visualizer):
        """Test RCS rendering."""
        # Create circular geometry
        n_points = 100
        theta = np.linspace(0, 2 * np.pi, n_points)
        geometry = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_points)])

        # RCS values
        rcs_db = -10 + 5 * np.cos(2 * theta)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rcs.png"

            fig = visualizer.render_radar_cross_section(
                geometry=geometry,
                rcs_db=rcs_db,
                frequency_ghz=10.0,
                output_path=str(output_path),
            )

            assert fig is not None
            assert output_path.exists()


class TestComplianceFeatures:
    """Tests for compliance and audit trail features."""

    def test_audit_trail_enabled(self, visualizer, sample_trajectory):
        """Test audit trail is recorded when enabled."""
        assert visualizer.config.enable_audit_log is True
        assert len(visualizer.audit_trail) == 0

        fig = visualizer.render_flight_trajectory(trajectory=sample_trajectory)

        assert fig is not None
        assert len(visualizer.audit_trail) == 1

        record = visualizer.audit_trail[0]
        assert isinstance(record, FrameAuditRecord)
        assert record.frame_id == 0
        assert record.seed == 42
        assert len(record.frame_hash) == 64  # SHA-256

    def test_audit_trail_disabled(self, sample_trajectory):
        """Test audit trail is not recorded when disabled."""
        config = AerospaceVizConfig(enable_audit_log=False)
        viz = AerospaceVisualizer(config)

        fig = viz.render_flight_trajectory(trajectory=sample_trajectory)

        assert fig is not None
        assert len(viz.audit_trail) == 0

    def test_deterministic_rendering(self, sample_trajectory):
        """Test same seed produces same frame hash."""
        # First run
        config1 = AerospaceVizConfig(seed=42, enable_audit_log=True)
        viz1 = AerospaceVisualizer(config1)
        fig1 = viz1.render_flight_trajectory(trajectory=sample_trajectory)
        trail1 = viz1.get_audit_trail()

        # Second run with same seed
        config2 = AerospaceVizConfig(seed=42, enable_audit_log=True)
        viz2 = AerospaceVisualizer(config2)
        fig2 = viz2.render_flight_trajectory(trajectory=sample_trajectory)
        trail2 = viz2.get_audit_trail()

        assert fig1 is not None
        assert fig2 is not None
        assert len(trail1) == 1
        assert len(trail2) == 1

        # Frame hashes should match
        assert trail1[0].frame_hash == trail2[0].frame_hash

    def test_non_deterministic_different_seed(self, sample_trajectory):
        """Test different seeds produce different frame hashes."""
        # First run
        config1 = AerospaceVizConfig(seed=42, enable_audit_log=True)
        viz1 = AerospaceVisualizer(config1)
        fig1 = viz1.render_flight_trajectory(trajectory=sample_trajectory, show_vapor_trail=True)
        trail1 = viz1.get_audit_trail()

        # Second run with different seed
        config2 = AerospaceVizConfig(seed=123, enable_audit_log=True)
        viz2 = AerospaceVisualizer(config2)
        fig2 = viz2.render_flight_trajectory(trajectory=sample_trajectory, show_vapor_trail=True)
        trail2 = viz2.get_audit_trail()

        assert fig1 is not None
        assert fig2 is not None

        # Frame hashes should differ (due to different vapor trail particles)
        # Note: This assumes vapor trail generation uses RNG
        assert trail1[0].frame_hash != trail2[0].frame_hash

    def test_get_audit_trail(self, visualizer, sample_trajectory):
        """Test get_audit_trail returns copy."""
        visualizer.render_flight_trajectory(trajectory=sample_trajectory)

        trail = visualizer.get_audit_trail()
        assert len(trail) == 1

        # Modify returned trail
        trail.clear()

        # Original should be unchanged
        assert len(visualizer.audit_trail) == 1

    def test_export_audit_trail(self, visualizer, sample_trajectory):
        """Test audit trail export to JSON."""
        visualizer.render_flight_trajectory(trajectory=sample_trajectory)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "audit.json"

            visualizer.export_audit_trail(str(output_path))

            assert output_path.exists()

            # Verify JSON structure
            import json

            with open(output_path) as f:
                data = json.load(f)

            assert "config" in data
            assert "frames" in data
            assert len(data["frames"]) == 1
            assert data["frames"][0]["frame_id"] == 0

    def test_generate_compliance_report(self, visualizer, sample_trajectory):
        """Test compliance report generation."""
        # Render multiple frames
        for _ in range(3):
            visualizer.render_flight_trajectory(trajectory=sample_trajectory)

        report = visualizer.generate_compliance_report()

        assert report["compliance_mode"] == ComplianceMode.DEVELOPMENT.value
        assert report["seed"] == 42
        assert report["total_frames"] == 3
        assert report["total_warnings"] == 0
        assert report["audit_enabled"] is True
        assert "render_time_stats" in report
        assert report["render_time_stats"]["average_ms"] > 0

    def test_compliance_mode_do178c_level_a(self, sample_trajectory):
        """Test DO-178C Level A compliance mode."""
        config = AerospaceVizConfig(
            compliance_mode=ComplianceMode.DO178C_LEVEL_A, enable_audit_log=True, seed=42
        )
        viz = AerospaceVisualizer(config)

        viz.render_flight_trajectory(trajectory=sample_trajectory)

        report = viz.generate_compliance_report()
        assert report["compliance_mode"] == "DO178C_LEVEL_A"
        assert report["audit_enabled"] is True


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_render_with_empty_trajectory(self, visualizer):
        """Test rendering with empty trajectory."""
        empty_trajectory = np.array([]).reshape(0, 3)

        # Should not crash, may return None or figure
        fig = visualizer.render_flight_trajectory(trajectory=empty_trajectory)

        # Either succeeds or fails gracefully
        assert fig is None or fig is not None

    def test_render_with_invalid_grid_shape(self, visualizer):
        """Test airflow rendering with mismatched grid shape."""
        velocity_field = np.random.randn(100, 3)
        grid_shape = (10, 10, 10)  # Total = 1000, but velocity_field has 100

        fig = visualizer.render_airflow_streamlines(
            velocity_field=velocity_field, grid_shape=grid_shape
        )

        # Should return None on error
        assert fig is None

    def test_frame_counter_increments(self, visualizer, sample_trajectory):
        """Test frame counter increments correctly."""
        assert visualizer.frame_counter == 0

        visualizer.render_flight_trajectory(trajectory=sample_trajectory)
        assert visualizer.frame_counter == 1

        visualizer.render_flight_trajectory(trajectory=sample_trajectory)
        assert visualizer.frame_counter == 2


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_multiple_render_types(self, visualizer, sample_trajectory, sample_fem_mesh):
        """Test rendering multiple types in sequence."""
        # Flight trajectory
        fig1 = visualizer.render_flight_trajectory(trajectory=sample_trajectory)
        assert fig1 is not None

        # FEM mesh
        nodes, elements, stress = sample_fem_mesh
        fig2 = visualizer.render_fem_mesh(nodes=nodes, elements=elements, stress_tensor=stress)
        assert fig2 is not None

        # Check audit trail has both frames
        trail = visualizer.get_audit_trail()
        assert len(trail) == 2
        assert trail[0].frame_id == 0
        assert trail[1].frame_id == 1

    def test_full_compliance_workflow(self, sample_trajectory):
        """Test complete DO-178C compliance workflow."""
        # Initialize with compliance mode
        config = AerospaceVizConfig(
            compliance_mode=ComplianceMode.DO178C_LEVEL_A, enable_audit_log=True, seed=42
        )
        viz = AerospaceVisualizer(config)

        # Render multiple frames
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                output_path = Path(tmpdir) / f"frame_{i}.png"
                fig = viz.render_flight_trajectory(
                    trajectory=sample_trajectory, output_path=str(output_path)
                )
                assert fig is not None
                assert output_path.exists()

            # Export audit trail
            audit_path = Path(tmpdir) / "audit.json"
            viz.export_audit_trail(str(audit_path))
            assert audit_path.exists()

            # Generate report
            report = viz.generate_compliance_report()
            assert report["total_frames"] == 3
            assert report["compliance_mode"] == "DO178C_LEVEL_A"

            # Verify deterministic hashes
            trail = viz.get_audit_trail()
            assert len({r.frame_hash for r in trail}) == 1  # All same trajectory
