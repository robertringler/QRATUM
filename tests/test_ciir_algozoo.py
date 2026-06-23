"""Regression tests for the CIIR Algorithm Zoo (quasim.ciir.algozoo).

These lock in the anti-fabrication guarantees of the catalog: every generated
algorithm is a real circuit executed on qratum.core and verified against an
INDEPENDENT oracle (brute-force optimum / scipy.expm fidelity / analytic
overlap / known phase). The tests assert on a bounded slice so they stay fast,
plus the cross-backend (qiskit) agreement check when qiskit is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasim.ciir import algozoo


def test_generates_a_large_distinct_catalog():
    specs = algozoo.generate_specs()
    # The catalog is well over 1000 distinct (class, instance) algorithms.
    assert len(specs) > 1000
    families = {s[0] for s in specs}
    assert families == {"F1", "F2", "F3", "F4"}  # all four families present


def test_every_run_record_passes_its_independent_oracle():
    # Bounded slice across all families for speed.
    specs = algozoo.generate_specs()
    rng = np.random.default_rng(0)
    sample = [specs[i] for i in rng.choice(len(specs), size=120, replace=False)]
    records = algozoo.run_all(sample)
    assert len(records) == len(sample)
    assert all(r.passed for r in records), [
        (r.algo_id, r.metric_name, r.observed, r.expected) for r in records if not r.passed
    ]
    # Structures are genuinely distinct circuits, not relabeled duplicates.
    assert len({r.structure_hash for r in records}) > len(records) * 0.8


def test_summary_reports_honest_aggregate():
    specs = algozoo.generate_specs()[:200]
    records = algozoo.run_all(specs)
    summary = algozoo.summarize(records)
    assert summary["total"] == len(records)
    assert summary["passed"] == sum(1 for r in records if r.passed)
    assert 0.0 <= summary["pass_rate"] <= 1.0


def test_qiskit_cross_check_agrees_to_machine_precision():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    specs = algozoo.generate_specs()
    records = algozoo.run_all(specs[:50])
    result = algozoo.cross_check_qiskit(records, specs[:50], n_sample=12, seed=1)
    assert result.get("checked", 0) > 0
    # Independent state vectors must agree to ~machine precision.
    assert result["max_fidelity_deviation"] < 1e-9, result
