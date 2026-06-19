"""Real genomics I/O and truth evaluation for XENON (no simulation).

This package replaces the previous np.random / "[SIMULATION]" genomics paths with
real, library-backed implementations:

* :mod:`xenon.genomics.vcf_io` - VCF parsing/writing via cyvcf2, with parsimonious
  variant normalization (the foundation for honest variant handling).
* :mod:`xenon.genomics.truth_eval` - GIAB-style truth reconciliation
  (precision / recall / F1 by variant type) for validating a callset against a
  truth VCF.

Heavy parser dependencies (cyvcf2, pysam) are optional; importing this package
does not require them - each module imports its backend lazily.
"""
