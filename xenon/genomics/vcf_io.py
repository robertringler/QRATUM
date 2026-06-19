"""Real VCF I/O and variant normalization (cyvcf2-backed).

Replaces the previous line-splitting + np.random VCF stub. Reads real VCF files
with cyvcf2 (full INFO/FORMAT/genotype semantics), normalizes variants
parsimoniously so equivalent representations compare equal, and writes valid
minimal VCF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class VariantKey:
    """A normalized, hashable variant identity for set comparison."""

    chrom: str
    pos: int
    ref: str
    alt: str

    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def is_indel(self) -> bool:
        return len(self.ref) != len(self.alt)


def normalize_variant(chrom: str, pos: int, ref: str, alt: str) -> VariantKey:
    """Parsimonious normalization (right- then left-trim shared bases).

    Makes different but equivalent representations of the same indel compare
    equal, e.g. (pos=100, REF=GA, ALT=GAA) and (pos=101, REF=A, ALT=AA).
    """
    ref, alt = ref.upper(), alt.upper()
    # Right-trim shared suffix.
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    # Left-trim shared prefix (advancing position).
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return VariantKey(str(chrom), int(pos), ref, alt)


@dataclass
class VariantRecord:
    """A single (biallelic) variant observation parsed from a VCF."""

    chrom: str
    pos: int
    ref: str
    alt: str
    qual: Optional[float] = None
    filter: Optional[str] = None
    info: dict = field(default_factory=dict)
    genotype: Optional[str] = None  # e.g. "0/1", "1/1"

    @property
    def key(self) -> VariantKey:
        return normalize_variant(self.chrom, self.pos, self.ref, self.alt)


def read_vcf(path: str, pass_only: bool = False) -> list[VariantRecord]:
    """Parse a VCF file into VariantRecords (one per ALT allele)."""
    import cyvcf2  # lazy; optional

    records: list[VariantRecord] = []
    vcf = cyvcf2.VCF(path)
    for v in vcf:
        filt = v.FILTER  # None means PASS in cyvcf2
        filt_str = "PASS" if filt is None else str(filt)
        if pass_only and filt_str not in ("PASS", "."):
            continue
        gt = None
        try:
            if v.genotypes:
                a = v.genotypes[0]
                gt = f"{a[0]}/{a[1]}" if len(a) >= 2 else None
        except Exception:
            gt = None
        for alt in v.ALT:
            records.append(
                VariantRecord(
                    chrom=v.CHROM,
                    pos=v.POS,
                    ref=v.REF,
                    alt=alt,
                    qual=float(v.QUAL) if v.QUAL is not None else None,
                    filter=filt_str,
                    info=dict(v.INFO),
                    genotype=gt,
                )
            )
    vcf.close()
    return records


def variant_keys(path: str, pass_only: bool = False) -> set[VariantKey]:
    """Return the set of normalized variant keys in a VCF."""
    return {r.key for r in read_vcf(path, pass_only=pass_only)}


def write_vcf(records: list[VariantRecord], path: str, sample: str = "SAMPLE") -> None:
    """Write a valid minimal VCF (4.2) for the given records."""
    lines = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "\t".join(
            ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", sample]
        ),
    ]
    for r in records:
        qual = "." if r.qual is None else f"{r.qual:g}"
        filt = r.filter or "PASS"
        gt = r.genotype or "0/1"
        lines.append(
            "\t".join(
                [r.chrom, str(r.pos), ".", r.ref, r.alt, qual, filt, ".", "GT", gt]
            )
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


__all__ = [
    "VariantKey",
    "VariantRecord",
    "normalize_variant",
    "read_vcf",
    "variant_keys",
    "write_vcf",
]
