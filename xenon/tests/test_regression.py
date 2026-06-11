"""Permanent regression tests for defects fixed during production hardening.

Each test documents a specific bug, links it to the fix, and fails if the
defect is reintroduced. These complement the broader behavioural suites by
pinning the exact failure modes that were previously shipped.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestHistogram2DRegression:
    """numpy.histogram2d does not accept bins='auto' (unlike 1-D histogram).

    InformationEngine._compute_joint_entropy passed bins='auto', which raised a
    numpy UFuncTypeError for *all* valid inputs, silently breaking every mutual
    information / joint-entropy computation. Fixed by deriving per-axis bin
    edges with np.histogram_bin_edges.
    """

    def test_mutual_information_on_valid_data_does_not_raise(self):
        info = pytest.importorskip(
            "qratum.bioinformatics.xenon.omics.information_engine"
        )
        engine = info.InformationEngine(seed=42)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((200, 1))
        y = rng.standard_normal((200, 1))

        result = engine.compute_mutual_information(x, y)

        assert "mutual_information" in result
        assert np.isfinite(result["mutual_information"])

    def test_nan_input_still_rejected(self):
        info = pytest.importorskip(
            "qratum.bioinformatics.xenon.omics.information_engine"
        )
        engine = info.InformationEngine(seed=42)
        y = np.random.default_rng(0).standard_normal((3, 1))
        nan_x = np.array([[1.0], [np.nan], [3.0]])
        with pytest.raises(ValueError):
            engine.compute_mutual_information(nan_x, y)


class TestConservationScoreRegression:
    """Conservation score normalized column entropy by a fixed log2(20).

    For a shallow alignment a column of all-distinct residues scored ~0.63
    (i.e. looked conserved). Fixed to normalize by the entropy attainable for
    the observed residue count, so all-distinct -> ~0 and all-identical -> 1.
    """

    def test_variable_column_scores_low(self):
        from xenon.bioinformatics.sequence_analyzer import SequenceAnalyzer

        analyzer = SequenceAnalyzer()
        scores = analyzer.compute_conservation_score(["AAAAAAA", "CCCCCCC", "GGGGGGG"])
        assert len(scores) == 7
        assert all(s < 0.5 for s in scores)

    def test_conserved_column_scores_high(self):
        from xenon.bioinformatics.sequence_analyzer import SequenceAnalyzer

        analyzer = SequenceAnalyzer()
        scores = analyzer.compute_conservation_score(["ACDEFGH", "ACDEFGH", "ACDEFGH"])
        assert all(s > 0.9 for s in scores)


class TestMechanismApiRegression:
    """The mechanism data model must serve both the canonical (keyword) and the
    legacy (positional / adapter) contracts after reconciliation.
    """

    def test_legacy_positional_and_aliases(self):
        from xenon.core.mechanism import BioMechanism, MolecularState, Transition

        state = MolecularState("S1", "ProteinA", -10.0, 0.5)
        assert state.state_id == "S1"  # alias -> name
        assert state.protein_name == "ProteinA"  # alias -> molecule
        assert state.free_energy == -10.0
        assert state.concentration == 0.5

        trans = Transition("S1", "S2", 1.5, -5.0, 20.0)
        assert trans.source_state == "S1" and trans.target_state == "S2"
        assert trans.delta_g == -5.0 and trans.activation_energy == 20.0

        mech = BioMechanism("M1", [state], [trans], 0.9)
        assert mech.evidence_score == 0.9
        assert [s.name for s in mech.states] == ["S1"]
        assert len(mech.transitions) == 1

    def test_canonical_keyword_form_still_works(self):
        from xenon.core.mechanism import BioMechanism, MolecularState, Transition

        mech = BioMechanism("test")
        mech.add_state(MolecularState(name="S1", molecule="P"))
        mech.add_state(MolecularState(name="S2", molecule="P"))
        mech.add_transition(Transition(source="S1", target="S2", rate_constant=1.0))
        assert len(mech.transitions) == 1


class TestMdSimulatorApiRegression:
    """set_positions on a fresh simulator must initialize (not raise a shape
    mismatch), and the str-enum / flat-attribute contract must hold.
    """

    def test_set_positions_initializes_fresh_simulator(self):
        from xenon.molecular_dynamics_lab.core.md_simulator import MDConfig, MDSimulator

        sim = MDSimulator()
        sim.set_positions(np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]]))
        assert sim._positions.shape == (3, 3)

        assert MDConfig().thermostat == "berendsen"
        assert MDConfig(thermostat="nose-hoover").thermostat == "nose_hoover"
