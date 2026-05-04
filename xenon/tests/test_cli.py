"""Tests for XENON CLI."""

import json
import os
import tempfile

from xenon.cli import create_sample_mechanism, main


class TestCLI:
    """Tests for XENON CLI functions."""

    def test_create_sample_mechanism(self):
        """Test sample mechanism creation."""

        mechanism = create_sample_mechanism(num_states=3)

        assert mechanism.mechanism_id == "SAMPLE_MECH_001"
        assert len(mechanism.states) == 3
        assert len(mechanism.transitions) >= 2  # At least forward transitions
        assert 0.0 <= mechanism.evidence_score <= 1.0

        # Check state properties
        for state in mechanism.states:
            assert state.state_id.startswith("S")
            assert state.protein_name.startswith("Protein_")
            assert state.free_energy <= 0.0  # Free energy should be negative
            assert 0.0 <= state.concentration <= 1.0

        # Check transition properties
        for transition in mechanism.transitions:
            assert transition.source_state.startswith("S")
            assert transition.target_state.startswith("S")
            assert transition.rate_constant > 0.0

    def test_cli_basic_run(self, monkeypatch):
        """Test basic CLI execution without visualization."""

        # Mock sys.argv
        monkeypatch.setattr(
            "sys.argv",
            ["xenon", "--num-states", "3"],
        )

        exit_code = main()
        assert exit_code == 0

    def test_cli_with_export(self, monkeypatch):
        """Test CLI with JSON export."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.json")

            monkeypatch.setattr(
                "sys.argv",
                [
                    "xenon",
                    "--num-states",
                    "4",
                    "--visualize",
                    "--export-json",
                    output_file,
                ],
            )

            exit_code = main()
            assert exit_code == 0

            # Check that JSON file was created
            assert os.path.exists(output_file)

            # Validate JSON content
            with open(output_file) as f:
                data = json.load(f)

            assert "nodes" in data
            assert "edges" in data
            assert "evidence_score" in data
            assert len(data["nodes"]) == 4

    def test_cli_verbose(self, monkeypatch, capsys):
        """Test CLI with verbose output."""

        monkeypatch.setattr(
            "sys.argv",
            ["xenon", "--num-states", "2", "--verbose"],
        )

        exit_code = main()
        assert exit_code == 0

        captured = capsys.readouterr()
        # Check that verbose output includes state and transition info
        assert "Mechanism States:" in captured.out or "states" in captured.out.lower()
