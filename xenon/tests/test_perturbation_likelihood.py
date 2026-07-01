"""Tests for the perturbation likelihood channel (audit finding F3).

Verifies the producer (kernel ``_execute_experiment``) and consumer
(``BayesianUpdater._likelihood_perturbation``) share a schema, and that the
likelihood actually depends on mechanism topology rather than returning a
constant fallback.
"""

from __future__ import annotations

import numpy as np

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.learning.bayesian_updater import BayesianUpdater, ExperimentResult
from xenon.runtime.xenon_kernel import Target, XENONRuntime

PROTEIN = "EGFR"


def _two_state(with_path: bool) -> BioMechanism:
    m = BioMechanism("m")
    m.add_state(MolecularState(name=f"{PROTEIN}_inactive", molecule=PROTEIN, concentration=50.0))
    m.add_state(MolecularState(name=f"{PROTEIN}_active", molecule=PROTEIN, concentration=50.0))
    if with_path:
        m.add_transition(
            Transition(source=f"{PROTEIN}_inactive", target=f"{PROTEIN}_active", rate_constant=1.0)
        )
    return m


def test_producer_emits_perturbation_schema():
    """The kernel must emit perturbation_source/target the likelihood consumes."""

    rt = XENONRuntime()
    rt._rng = np.random.default_rng(0)
    exp = rt._execute_experiment(
        Target(name="EGFR_target", protein=PROTEIN, objective="characterize"),
        {"type": "perturbation", "target": "EGFR_target", "iteration": 0},
    )
    assert exp.conditions.get("perturbation_source") == f"{PROTEIN}_inactive"
    assert exp.conditions.get("perturbation_target") == f"{PROTEIN}_active"
    assert "response" in exp.observations


def test_likelihood_depends_on_topology():
    """A mechanism that can propagate the perturbation outscores one that cannot."""

    exp = ExperimentResult(
        experiment_type="perturbation",
        observations={"response": 0.5},
        uncertainties={"response": 0.1},
        conditions={
            "perturbation_source": f"{PROTEIN}_inactive",
            "perturbation_target": f"{PROTEIN}_active",
            "temperature": 310.0,
        },
    )
    bu = BayesianUpdater()
    lik_path = bu.compute_likelihood(_two_state(with_path=True), exp)
    lik_no_path = bu.compute_likelihood(_two_state(with_path=False), exp)

    # No longer the inert 0.1 constant, and topology-discriminating.
    assert lik_path != 0.1
    assert lik_path > lik_no_path


def test_end_to_end_schema_contract():
    """Experiment produced by the kernel yields a mechanism-dependent likelihood."""

    rt = XENONRuntime()
    rt._rng = np.random.default_rng(1)
    exp = rt._execute_experiment(
        Target(name="EGFR_target", protein=PROTEIN, objective="characterize"),
        {"type": "perturbation", "target": "EGFR_target", "iteration": 0},
    )
    bu = BayesianUpdater()
    assert bu.compute_likelihood(_two_state(True), exp) > bu.compute_likelihood(
        _two_state(False), exp
    )


def test_causal_paths_without_networkx():
    """get_causal_paths has a networkx-free fallback (perturbation channel robustness)."""

    import xenon.core.mechanism as mech_mod

    original = mech_mod.nx
    try:
        mech_mod.nx = None
        m = BioMechanism("chain")
        for name in ("A", "B", "C"):
            m.add_state(MolecularState(name=name, molecule=name))
        m.add_transition(Transition(source="A", target="B", rate_constant=1.0))
        m.add_transition(Transition(source="B", target="C", rate_constant=1.0))
        assert m.get_causal_paths("A", "C") == [["A", "B", "C"]]
        assert m.get_causal_paths("C", "A") == []
    finally:
        mech_mod.nx = original
