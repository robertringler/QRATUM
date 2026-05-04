import json

from qnx import cli


def test_cli_simulate_produces_structured_output(capsys):
    exit_code = cli.cli(["simulate", "--scenario", "cli-smoke", "--timesteps", "1"])

    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert exit_code == 0
    assert set(payload) >= {"backend", "hash", "execution_time_ms"}
    assert payload["backend"] == "quasim_modern"
