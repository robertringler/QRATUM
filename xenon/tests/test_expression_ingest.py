"""Tests for the production real-matrix ingest path + CLI (option c)."""

from __future__ import annotations

import json

import numpy as np

from xenon.bioinformatics.expression_to_mechanism import (
    deduplicate_genes,
    impute_missing,
    main,
    normalize_matrix,
    parse_metadata,
    run_pipeline,
)

MARKERS = """gene\tmodule
Zar1\tmaternal
Npm2\tmaternal
Zp3\tmaternal
Zscan4d\tzygotic
Dux\tzygotic
Zfp352\tzygotic
Hist1h1a\tpolya
"""

SAMPLES = ["t0_rep1", "t0_rep2", "t12_rep1", "t12_rep2", "t17_rep1", "t17_rep2"]


def _matrix_text(maternal, zygotic, samples=SAMPLES):
    lines = ["Gene\t" + "\t".join(samples)]
    for g, vals in [("Zar1", maternal), ("Npm2", maternal), ("Zp3", maternal),
                    ("Zscan4d", zygotic), ("Dux", zygotic), ("Zfp352", zygotic),
                    ("Hist1h1a", [50] * len(samples))]:
        lines.append(g + "\t" + "\t".join(str(v) for v in vals))
    return "\n".join(lines) + "\n"


def _sdrf_text(samples=SAMPLES):
    lines = ["Source Name\tCharacteristics[age]\tComment[library construction]\tCharacteristics[replicate]"]
    for s in samples:
        hrs = s.split("_")[0][1:]
        rep = s.split("rep")[1]
        lines.append(f"{s}\t{hrs}\tpolyA+ selection\t{rep}")
    return "\n".join(lines) + "\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_dedup_and_impute():
    genes, X = deduplicate_genes(["A", "A", "B"], np.array([[1.0, 3.0], [3.0, 5.0], [10.0, 10.0]]))
    assert genes == ["A", "B"]
    np.testing.assert_allclose(X[0], [2.0, 4.0])  # averaged
    Y = impute_missing(np.array([[1.0, np.nan, 3.0]]))
    assert not np.isnan(Y).any()


def test_normalization_methods():
    X = np.array([[10.0, 0.0], [30.0, 100.0]])
    np.testing.assert_array_equal(normalize_matrix(X, "raw"), X)
    np.testing.assert_array_equal(normalize_matrix(X, "tpm"), X)  # pass-through
    assert np.allclose(normalize_matrix(X, "log1p"), np.log1p(X))
    cpm = normalize_matrix(X, "cpm")
    assert np.allclose(cpm.sum(axis=0), [1e6, 1e6])


def test_real_metadata_parse_from_sra_run_table():
    """Parser handles the actual ENA/SRA run-table column layout."""
    import os
    upload = "/root/.claude/uploads/94427ff4-358c-5b6f-9822-41db27626bc2/6e707a87-SraRunTable.csv"
    if not os.path.exists(upload):
        return  # fixture only present in the original session
    meta = parse_metadata(upload)
    any_rec = next(iter(meta.values()))
    assert any_rec.timepoint_hours == 17.0
    assert any_rec.treatment == "polyA+"
    assert any_rec.dev_stage == "2-cell"


def test_multitimepoint_recovers_K(tmp_path):
    # late (t17) ratio zygotic:maternal = 66:33 = 2 => K=2
    maternal = [100, 100, 55, 55, 33, 33]
    zygotic = [5, 5, 45, 45, 66, 66]
    mtx = _write(tmp_path, "counts.tsv", _matrix_text(maternal, zygotic))
    sdrf = _write(tmp_path, "sdrf.tsv", _sdrf_text())
    markers = _write(tmp_path, "markers.tsv", MARKERS)

    summary = run_pipeline(mtx, str(tmp_path / "out"), metadata_path=sdrf,
                           gene_sets_path=markers, normalization="tpm", seed=42)
    assert summary["mechanism_recovery"] is True
    assert summary["inference"]["status"] == "OK"
    assert summary["inference"]["recovered_K"] == 2.0
    # outputs written
    for f in ("normalized_expression.tsv", "selected_modules.tsv", "state_variables.tsv",
              "candidate_mechanisms.json", "xenon_inference_results.json", "report.md"):
        assert (tmp_path / "out" / f).exists()


def test_one_timepoint_refuses(tmp_path):
    samples = ["t17_rep1", "t17_rep2"]
    mtx = _write(tmp_path, "counts1.tsv", _matrix_text([33, 33], [66, 66], samples))
    sdrf = _write(tmp_path, "sdrf1.tsv", _sdrf_text(samples))
    markers = _write(tmp_path, "markers.tsv", MARKERS)

    summary = run_pipeline(mtx, str(tmp_path / "out1"), metadata_path=sdrf,
                           gene_sets_path=markers, normalization="tpm", seed=42)
    assert summary["mechanism_recovery"] is False
    assert summary["inference"]["status"] == "REFUSED"
    assert "insufficient temporal structure" in summary["inference"]["reason"]
    # ingestion still produced modules + state variables
    assert (tmp_path / "out1" / "state_variables.tsv").exists()
    assert (tmp_path / "out1" / "selected_modules.tsv").exists()


def test_cli_entry_point(tmp_path):
    maternal = [100, 100, 33, 33]
    zygotic = [5, 5, 66, 66]
    samples = ["t0_rep1", "t0_rep2", "t17_rep1", "t17_rep2"]
    mtx = _write(tmp_path, "c.tsv", _matrix_text(maternal, zygotic, samples))
    sdrf = _write(tmp_path, "s.tsv", _sdrf_text(samples))
    markers = _write(tmp_path, "m.tsv", MARKERS)
    rc = main(["--matrix", mtx, "--metadata", sdrf, "--gene-sets", markers,
               "--normalization", "tpm", "--out", str(tmp_path / "cliout")])
    assert rc == 0
    res = json.loads((tmp_path / "cliout" / "xenon_inference_results.json").read_text())
    assert res["status"] == "OK"
