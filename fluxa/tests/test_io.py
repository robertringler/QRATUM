"""Artefact serialisation: a uniform, loadable table for every scenario."""

from __future__ import annotations

import csv
import json

import pytest

from fluxa.io import write_json, write_rows_csv, write_states_csv, write_states_parquet, write_result
from fluxa.tests.conftest import make_engine

VARYING_SCENARIOS = ("A_BASELINE", "B_SOLAR_SHOCK", "F_COMPOUND")


@pytest.mark.parametrize("scenario_id", VARYING_SCENARIOS)
def test_every_timestep_exports_the_same_columns(loaded, scenarios, scenario_id, tmp_path):
    """Perturbations come and go during a run, so a naive flattening gives
    different timesteps different columns and the table cannot be loaded."""
    result = make_engine(loaded, run_id="io", n_steps=288, authorized=True).run(
        scenarios[scenario_id]
    )
    path = write_states_csv(tmp_path / "states.csv", result.states)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 288
    widths = {len(row) for row in rows}
    assert len(widths) == 1, f"ragged table: row widths {widths}"


def test_active_perturbations_survive_as_canonical_json(loaded, scenarios, tmp_path):
    result = make_engine(loaded, run_id="io-pert", n_steps=288, authorized=True).run(
        scenarios["F_COMPOUND"]
    )
    path = write_states_csv(tmp_path / "states.csv", result.states)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    active = [json.loads(row["active_perturbations"]) for row in rows]
    assert any(len(a) == 3 for a in active), "the compound event must show three perturbations"
    assert any(len(a) == 0 for a in active), "pre-onset steps must show none"
    for row, mapping in zip(rows, active, strict=True):
        assert int(row["active_perturbations.count"]) == len(mapping)


def test_parquet_export_round_trips(loaded, scenarios, tmp_path):
    pyarrow = pytest.importorskip("pyarrow.parquet")
    result = make_engine(loaded, run_id="io-parquet", n_steps=96).run(scenarios["A_BASELINE"])
    path = write_states_parquet(tmp_path / "states.parquet", result.states)
    assert path is not None
    table = pyarrow.read_table(path)
    assert table.num_rows == 96
    assert "total_load_mw" in table.column_names


def test_write_result_produces_every_artefact(loaded, scenarios, tmp_path):
    result = make_engine(loaded, run_id="io-all", n_steps=48).run(scenarios["A_BASELINE"])
    written = write_result(result, tmp_path, prefix="run")
    for key in ("summary", "metrics", "provenance", "events", "states_csv"):
        assert key in written
        assert (tmp_path / f"run_{key.split('_')[0]}.json").exists() or written[key]
    events = json.loads((tmp_path / "run_events.json").read_text())
    assert events["root_hash"] == result.ledger.root_hash()
    assert len(events["events"]) == len(result.ledger)
    provenance = json.loads((tmp_path / "run_provenance.json").read_text())
    assert provenance["identity_hash"] == result.provenance.identity_hash()


def test_exported_events_carry_their_chain_hashes(loaded, scenarios, tmp_path):
    result = make_engine(loaded, run_id="io-chain", n_steps=24).run(scenarios["A_BASELINE"])
    written = write_result(result, tmp_path, prefix="chain", write_states=False)
    events = json.loads((tmp_path / "chain_events.json").read_text())["events"]
    assert events[0]["previous_hash"]
    for earlier, later in zip(events, events[1:], strict=False):
        assert later["previous_hash"] == earlier["node_hash"]
    assert "states_csv" not in written


def test_empty_trajectory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty trajectory"):
        write_states_csv(tmp_path / "x.csv", [])


def test_zero_rows_is_refused(tmp_path):
    with pytest.raises(ValueError, match="zero rows"):
        write_rows_csv(tmp_path / "x.csv", [])


def test_rows_csv_unions_keys_in_first_seen_order(tmp_path):
    path = write_rows_csv(tmp_path / "u.csv", [{"a": 1}, {"b": 2}])
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["a", "b"]
        rows = list(reader)
    assert rows[0]["b"] == "" and rows[1]["a"] == ""


def test_json_is_written_with_sorted_keys(tmp_path):
    path = write_json(tmp_path / "j.json", {"b": 1, "a": 2})
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index('"b"')
