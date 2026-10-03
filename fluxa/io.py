"""Persistence of FLUXA results in machine-readable formats.

Artefacts written per run
-------------------------
``<prefix>_states.csv``      one row per timestep, flattened
``<prefix>_states.parquet``  same, when pyarrow is available
``<prefix>_events.json``     full event ledger with per-node hashes
``<prefix>_provenance.json`` the provenance bundle
``<prefix>_metrics.json``    all metric families
``<prefix>_summary.json``    the compact summary

Parquet is written through pyarrow when it is importable and skipped with a
logged warning otherwise, so a missing optional dependency degrades the
artefact set rather than failing the experiment. CSV and JSON are always
written.

Version: 1.0.0
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from fluxa.engine import SimulationResult
from fluxa.state import FluxaState

LOGGER = logging.getLogger("fluxa.io")

#: Default root for FLUXA artefacts, relative to the repository root.
DEFAULT_RESULTS_DIR = Path("results/fluxa")


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as indented, key-sorted JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p


def write_states_csv(path: str | Path, states: Sequence[FluxaState]) -> Path:
    """Write the timestep trajectory as CSV.

    Raises:
        ValueError: if ``states`` is empty, since the header would be unknown.
    """
    if not states:
        raise ValueError("cannot write CSV for an empty trajectory")
    rows = [s.flat_record() for s in states]
    # The header is taken from the first row; every FluxaState has identical
    # structure for a given system, so a mismatch indicates a real bug.
    fieldnames = list(rows[0].keys())
    for i, row in enumerate(rows[1:], start=1):
        if list(row.keys()) != fieldnames:
            raise ValueError(f"state {i} has a different schema than state 0")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p


def write_states_parquet(path: str | Path, states: Sequence[FluxaState]) -> Path | None:
    """Write the trajectory as Parquet, or return None if pyarrow is absent."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        LOGGER.warning("pyarrow not available; skipping Parquet output for %s", path)
        return None
    if not states:
        raise ValueError("cannot write Parquet for an empty trajectory")
    rows = [s.flat_record() for s in states]
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    table = pa.table(columns)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, p)
    return p


def write_rows_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write an iterable of uniform dicts as CSV, union-ing keys in first-seen order."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write CSV for zero rows")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return p


def write_result(
    result: SimulationResult,
    out_dir: str | Path = DEFAULT_RESULTS_DIR,
    prefix: str | None = None,
    *,
    write_states: bool = True,
    write_events: bool = True,
) -> dict[str, str]:
    """Write all artefacts for one run. Returns {artefact_name: path}."""
    directory = ensure_dir(out_dir)
    stem = prefix or f"{result.scenario.scenario_id}_{result.run_config.run_id}"
    written: dict[str, str] = {}

    written["summary"] = str(write_json(directory / f"{stem}_summary.json", result.summary()))
    written["metrics"] = str(
        write_json(directory / f"{stem}_metrics.json", result.metrics.to_dict())
    )
    written["provenance"] = str(
        write_json(directory / f"{stem}_provenance.json", result.provenance.to_dict())
    )
    if write_events:
        written["events"] = str(
            write_json(
                directory / f"{stem}_events.json",
                {
                    "ledger_id": result.ledger.ledger_id,
                    "root_hash": result.ledger.root_hash(),
                    "counts_by_type": result.ledger.counts_by_type(),
                    "events": result.ledger.export(),
                },
            )
        )
    if write_states:
        written["states_csv"] = str(
            write_states_csv(directory / f"{stem}_states.csv", result.states)
        )
        parquet = write_states_parquet(directory / f"{stem}_states.parquet", result.states)
        if parquet is not None:
            written["states_parquet"] = str(parquet)
    return written
