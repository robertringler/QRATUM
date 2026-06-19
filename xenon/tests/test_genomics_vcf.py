"""Phase 0 genomics foundation tests: real VCF I/O + truth reconciliation.

Skips when cyvcf2 is unavailable. These prove the genomics path is real (parses
actual VCF semantics, normalizes variants, computes honest precision/recall) -
no np.random anywhere.
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("cyvcf2")

from xenon.genomics.truth_eval import reconcile_keys, reconcile_vcf
from xenon.genomics.vcf_io import (
    VariantRecord,
    normalize_variant,
    read_vcf,
    variant_keys,
    write_vcf,
)

_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
)


def _write(text: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), "v.vcf")
    with open(path, "w") as fh:
        fh.write(_VCF_HEADER + text)
    return path


def test_reads_real_vcf_semantics():
    path = _write(
        "1\t100\t.\tA\tT\t50\tPASS\t.\tGT\t0/1\n"
        "1\t200\t.\tAC\tA\t40\tPASS\t.\tGT\t1/1\n"  # deletion
    )
    recs = read_vcf(path)
    assert len(recs) == 2
    snv = recs[0]
    assert (snv.chrom, snv.pos, snv.ref, snv.alt) == ("1", 100, "A", "T")
    assert snv.qual == 50.0 and snv.genotype == "0/1"
    assert recs[0].key.is_snv and recs[1].key.is_indel


def test_normalization_makes_equivalent_indels_equal():
    # Parsimonious trimming collapses padded representations of the same event.
    # Right-trim: a shared suffix base is removed.
    assert normalize_variant("1", 100, "CT", "CGT") == normalize_variant("1", 100, "C", "CG")
    # Left-trim: a shared prefix base is removed and the position advances.
    assert normalize_variant("1", 100, "AG", "AC") == normalize_variant("1", 101, "G", "C")
    # (Full repeat left-alignment requires the reference genome; documented as
    # the next normalization step.)


def test_truth_reconciliation_precision_recall():
    truth = _write(
        "1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n"
        "1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t0/1\n"
        "1\t300\t.\tAC\tA\t.\tPASS\t.\tGT\t0/1\n"
    )
    query = _write(
        "1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n"  # TP (snv)
        "1\t250\t.\tC\tA\t.\tPASS\t.\tGT\t0/1\n"  # FP (snv, not in truth)
        "1\t300\t.\tAC\tA\t.\tPASS\t.\tGT\t0/1\n"  # TP (indel)
    )
    result = reconcile_vcf(query, truth)
    # SNV: 1 TP (100), 1 FP (250), 1 FN (200) -> P=0.5, R=0.5
    assert result["snv"]["tp"] == 1 and result["snv"]["fp"] == 1 and result["snv"]["fn"] == 1
    assert result["snv"]["precision"] == pytest.approx(0.5)
    assert result["snv"]["recall"] == pytest.approx(0.5)
    # Indel: 1 TP (300), 0 FP, 0 FN -> perfect
    assert result["indel"]["f1"] == pytest.approx(1.0)
    # Overall: 2 TP, 1 FP, 1 FN
    assert result["all"]["tp"] == 2 and result["all"]["fp"] == 1 and result["all"]["fn"] == 1


def test_perfect_self_concordance():
    path = _write("1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n1\t200\t.\tG\tC\t.\tPASS\t.\tGT\t1/1\n")
    keys = variant_keys(path)
    metrics = reconcile_keys(keys, keys)["all"]
    assert metrics.precision == 1.0 and metrics.recall == 1.0 and metrics.f1 == 1.0


def test_vcf_roundtrip_is_lossless_on_keys():
    path = _write(
        "1\t100\t.\tA\tT\t50\tPASS\t.\tGT\t0/1\n"
        "2\t500\t.\tACGT\tA\t30\tPASS\t.\tGT\t1/1\n"
    )
    recs = read_vcf(path)
    out = os.path.join(tempfile.mkdtemp(), "out.vcf")
    write_vcf(recs, out)
    reparsed = read_vcf(out)
    assert {r.key for r in recs} == {r.key for r in reparsed}
