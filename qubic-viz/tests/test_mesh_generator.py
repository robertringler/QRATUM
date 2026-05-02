"""Tests for mesh generator."""

import sys
from pathlib import Path

import numpy as np

# Add qubic-viz to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.mesh_generator import TireMesh, TireMeshGenerator


class MockTireGeometry:
    """Mock tire geometry for testing."""

    def __init__(self):
        self.outer_diameter_mm = 700.0
        self.width_mm = 225.0
        self.rim_diameter_inch = 17.0


def test_tire_mesh_properties():
    """Test TireMesh properties."""

    vertices = np.random.rand(100, 3)
    faces = np.random.randint(0, 100, (50, 3))
    normals = np.random.rand(100, 3)
    uvs = np.random.rand(100, 2)

    mesh = TireMesh(vertices=vertices, faces=faces, normals=normals, uvs=uvs)

    assert mesh.num_vertices == 100
    assert mesh.num_faces == 50


def test_tire_mesh_compute_face_normals():
    """Test face normal computation."""

    # Create simple triangle
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    normals = np.zeros((4, 3))
    uvs = np.zeros((4, 2))

    mesh = TireMesh(vertices=vertices, faces=faces, normals=normals, uvs=uvs)
    face_normals = mesh.compute_face_normals()

    assert face_normals.shape == (2, 3)
    # Check that normals are normalized
    assert np.allclose(np.linalg.norm(face_normals, axis=1), 1.0)


def test_tire_mesh_recalculate_normals():
    """Test vertex normal recalculation."""

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    normals = np.zeros((3, 3))
    uvs = np.zeros((3, 2))

    mesh = TireMesh(vertices=vertices, faces=faces, normals=normals, uvs=uvs)
    mesh.recalculate_normals()

    # Check that normals are normalized
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)


def test_tire_mesh_generator_initialization():
    """Test TireMeshGenerator initialization."""

    gen = TireMeshGenerator(resolution=32)
    assert gen.resolution == 32


def test_generate_tire_mesh():
    """Test tire mesh generation."""

    gen = TireMeshGenerator(resolution=16)
    geometry = MockTireGeometry()

    mesh = gen.generate_tire_mesh(geometry)

    assert isinstance(mesh, TireMesh)
    assert mesh.num_vertices > 0
    assert mesh.num_faces > 0
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert mesh.normals.shape == mesh.vertices.shape
    assert mesh.uvs.shape[1] == 2


def test_generate_tire_mesh_resolution():
    """Test that resolution affects mesh detail."""

    geometry = MockTireGeometry()

    gen_low = TireMeshGenerator(resolution=8)
    mesh_low = gen_low.generate_tire_mesh(geometry)

    gen_high = TireMeshGenerator(resolution=32)
    mesh_high = gen_high.generate_tire_mesh(geometry)

    # Higher resolution should produce more vertices
    assert mesh_high.num_vertices > mesh_low.num_vertices
