import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from quasim.simulation.quantacosmic import (ActionFunctional, CouplingOperator,
                                            FieldLattice, MetricTensor,
                                            QuantumManifold, coupling_matrix,
                                            curvature_scalar, propagate_field,
                                            revultra_temporal_curvature)


def test_quantum_manifold_volume_and_projection():
    metric = [[2.0, 0.0], [0.0, 0.5]]
    manifold = QuantumManifold(dimension=2, metric=metric)

    assert pytest.approx(manifold.volume_element()) == math.sqrt(1.0)
    projection = manifold.project_vector([1.0, 2.0])
    assert projection == [2.0, 1.0]
    assert pytest.approx(curvature_scalar(metric)) == 2.5


def test_coupling_and_field_propagation():
    operator = CouplingOperator([[0.0, 1.0], [2.0, 0.0]])
    symmetric = coupling_matrix(operator.matrix())
    assert symmetric[0][1] == symmetric[1][0]

    final_state = propagate_field([1.0, 0.0], operator, timestep=0.5, steps=2)
    assert pytest.approx(final_state[0]) == 0.5
    assert pytest.approx(final_state[1]) == 0.0


def test_action_functional_midpoint_rule():
    contour = [0.0, 0.5, 1.0]

    def lagrangian(time, field):
        return time + field

    action = ActionFunctional(lagrangian=lagrangian, contour=contour)
    value = action.evaluate([0.0, 1.0, 2.0])
    assert pytest.approx(value) == 1.5


def test_revultra_curvature_normalisation():
    metric = MetricTensor([[3.0, 0.2], [0.2, 2.0]])
    curvature = revultra_temporal_curvature(metric, temporal_frequency=1.5, cognitive_twist=0.8)
    assert curvature > 0
    normalized_metric = metric.normalize()
    assert pytest.approx(normalized_metric.determinant(), rel=1e-5) == 1.0

    lattice = FieldLattice(positions=[0.0, 1.0, 2.0], timestep=0.1)
    kernel = lattice.propagation_kernel(CouplingOperator([[1.0, 0.0, 0.0]] * 3))
    assert all(len(row) == 3 for row in kernel)
