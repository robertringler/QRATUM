"""SBML round-trip tests (Phase 2). Skips when python-libsbml is unavailable."""

from __future__ import annotations

import pytest

pytest.importorskip("libsbml")

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.learning.sbml_io import mechanism_to_sbml, sbml_to_mechanism


def _mechanism():
    mech = BioMechanism("ABC")
    mech.add_state(MolecularState(name="A", molecule="A", concentration=100.0))
    mech.add_state(MolecularState(name="B", molecule="B", concentration=0.0))
    mech.add_state(MolecularState(name="C", molecule="C", concentration=0.0))
    mech.add_transition(Transition(source="A", target="B", rate_constant=0.8))
    mech.add_transition(Transition(source="B", target="C", rate_constant=0.3))
    return mech


def test_sbml_roundtrip_preserves_topology_and_rates():
    mech = _mechanism()
    sbml = mechanism_to_sbml(mech)
    assert "<sbml" in sbml and "reaction" in sbml

    restored = sbml_to_mechanism(sbml)

    assert {s.name for s in restored.states} == {"A", "B", "C"}
    edges = {(t.source, t.target): t.rate_constant for t in restored.transitions}
    assert edges[("A", "B")] == pytest.approx(0.8)
    assert edges[("B", "C")] == pytest.approx(0.3)

    concs = {s.name: s.concentration for s in restored.states}
    assert concs["A"] == pytest.approx(100.0)


def test_exported_sbml_is_valid():
    import libsbml

    sbml = mechanism_to_sbml(_mechanism())
    doc = libsbml.readSBMLFromString(sbml)
    assert doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) == 0
    assert doc.getModel().getNumReactions() == 2
