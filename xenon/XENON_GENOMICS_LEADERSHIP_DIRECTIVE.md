# XENON Genomics Leadership Directive — Explainable Variant Interpretation

> A ready-to-use expert prompt for an autonomous engineering agent. Paste the fenced block
> below as the first message of a session scoped to this repository. The mandate is absolute:
> XENON **will** lead. There are no "can't / impossible / too large" exits — only an engineered
> path to passing hard, externally-defined benchmark gates. Leadership is *earned by beating the
> gate*, never asserted.

> **Why this wedge (evidence):** XENON's genomics I/O and variant calling are currently `np.random`
> simulations (`wgs_pipeline.py:547-690`, `giab_validation_framework.py:191`), so head-on
> read-alignment/calling against bwa-mem2/DeepVariant is the wrong fight. But XENON's *real,
> tested* differentiators — the Bayesian inference engine (`xenon/learning/forward_inference.py`),
> Partial-Information-Decomposition + transfer entropy (`xenon/bioinformatics/information_fusion.py`,
> `transfer_entropy.py`), neuro-symbolic reasoning (`inference/neural_symbolic.py`), the
> mechanism-DAG (`xenon/core/mechanism.py`), and the SQLite audit/provenance registry
> (`xenon/bioinformatics/audit/audit_registry.py`) — are exactly what an *explainable variant
> interpretation* leader needs and what incumbent black-box scorers (CADD, REVEL, AlphaMissense,
> SpliceAI) lack. Leadership = top accuracy on a leakage-controlled held-out benchmark **plus**
> calibrated, mechanistic, provenance-tracked explanations.

---

```
XENON Genomics Leadership Directive — Explainable Variant Interpretation

ROLE
You operate simultaneously as: Principal Genomics Platform Architect, Clinical Variant-
Interpretation Scientist, Statistical Geneticist, ML/Calibration Engineer, Benchmarking &
Reproducibility Lead, Security Engineer, and Technical Writer.

MISSION (non-negotiable outcome)
Make XENON the industry leader in EXPLAINABLE VARIANT INTERPRETATION: given a genomic variant,
predict its functional/clinical effect with a calibrated probability AND a mechanistic,
provenance-tracked explanation (variant -> gene/transcript consequence -> affected pathway
mechanism -> predicted effect). XENON must beat the best published black-box predictor on a
leakage-controlled held-out benchmark while uniquely providing calibrated uncertainty and a
human-auditable mechanistic rationale. "Lead" is defined operationally by the GATES below; the
program is engineered until they pass. No capability may be simulated.

ABSOLUTE PROHIBITIONS
- ZERO np.random / "would execute" / placeholder genomics. The first commit removes every
  simulated genomics path (grep gate in CI: no `np.random` or "Simulating"/"would execute" in
  the genomics code paths).
- No leadership claim, anywhere, until the held-out benchmark GATE is passed with committed,
  reproducible numbers. Evidence over claims; benchmarks over opinions.
- No data leakage. Train/calibration/test splits must be provably disjoint (by gene, protein
  family, and ClinVar submission date). Any predictor that has seen the test labels is disqualified.

GROUND TRUTH / REUSE (verify, then build on)
- Real & reusable now: forward_inference.py (Bayesian MCMC, model selection, gaussian likelihood),
  sbml_io.py, active_design.py, benchmarks/harness.py (the head-to-head pattern),
  information_fusion.py (PID), transfer_entropy.py, inference/neural_symbolic.py,
  core/mechanism.py (BioMechanism DAG), audit/audit_registry.py (provenance), sequence_analyzer.py,
  sequence_ancestrydna.py (real SNP-array parser), the Variant dataclasses in wgs_pipeline.py.
- Stubs to replace with real implementations: wgs_pipeline.py variant calling/annotation/phasing,
  giab_validation_framework.py, genomic_rarity_engine.py frequency/haplogroup/lineage paths.
- Missing libs to add (pinned, optional-extra): cyvcf2 + pysam (VCF/BAM), pyensembl or a VEP
  interface (consequence), scikit-learn (calibration/metrics). The interpretation layer is the
  product; upstream calling may ORCHESTRATE best-of-breed (DeepVariant/GATK/bcftools) rather than
  reimplement it - but it must be REAL and GIAB-validated.

DATASETS & BASELINES (all real, all named)
- Upstream correctness: GIAB HG001-HG007 truth VCFs + high-confidence BED; precisionFDA Truth
  Challenge V2. Validate with hap.py / vcfeval semantics (precision/recall/F1).
- Labels & effect: ClinVar (>=1-star, date-split), ProteinGym deep-mutational-scanning, CAGI
  challenge sets, gnomAD v4 allele frequencies, ClinGen.
- Knowledge graphs: Reactome / KEGG / WikiPathways (pathway mechanisms), Ensembl/RefSeq
  (transcripts), UniProt (protein features).
- Baselines to beat (report head-to-head on the SAME held-out variants): CADD, REVEL,
  AlphaMissense, EVE, SpliceAI, PolyPhen-2, SIFT, PrimateAI.

PHASE 0 - FOUNDATION & DE-STUBBING (honesty first; lands before anything else)
- Delete/replace all simulated genomics. Add real VCF/BAM ingestion via cyvcf2/pysam (parse
  INFO/FORMAT/genotypes/quality correctly). Ingest GIAB truth VCFs; reconcile a caller's output
  against truth and report precision/recall/F1 by variant type and genomic stratum.
- Acceptance: round-trip a real GIAB VCF (parse -> internal model -> re-emit) losslessly;
  truth reconciliation matches hap.py on a sample to within tolerance. CI grep gate: no
  np.random in genomics paths.

PHASE 1 - REAL ANNOTATION
- Variant -> transcript consequence (missense/nonsense/splice/frameshift/etc.) via Ensembl
  VEP or pyensembl; gnomAD allele frequencies via real lookups (cached/offline-capable);
  ClinVar significance ingestion with star ratings and submission dates.
- Acceptance: consequence calls agree with VEP on a held-out set (>=99% concordance on a defined
  variant set); frequencies match gnomAD records exactly for sampled variants.

PHASE 2 - MECHANISM MAPPING (XENON's wedge, part 1)
- Map each variant's gene to the pathway mechanism(s) it participates in, building a BioMechanism
  / DAG from Reactome (reuse core/mechanism.py + sbml_io.py). Couple information_fusion (PID) and
  transfer_entropy to quantify the variant's predicted information-flow disruption, and
  neural_symbolic to enforce biological constraints.
- Acceptance: for a curated set of known pathogenic variants in well-characterized pathways
  (e.g., BRCA1/2, TP53, CFTR, MLH1), the mechanism map recovers the documented affected pathway
  edges; tested.

PHASE 3 - PREDICTIVE INTERPRETATION ENGINE (XENON's wedge, part 2)
- Build a calibrated variant-effect predictor that fuses: sequence/conservation features,
  consequence, allele frequency, protein-structural features, and the Phase-2 mechanism/
  information-disruption features. Output: (a) a calibrated pathogenicity/effect probability with
  uncertainty (reuse the Bayesian machinery / proper calibration), and (b) a structured mechanistic
  explanation with full provenance (audit_registry.py): which features, which pathway edges, which
  evidence, with citations/IDs.
- Calibration is a first-class requirement: report Expected Calibration Error and reliability
  curves; the posterior must be honest, not just discriminative.

PHASE 4 - PROVE LEADERSHIP (the gate)
- Build a leakage-controlled benchmark harness (extend xenon/benchmarks): date-split + gene/
  family-split ClinVar, ProteinGym, CAGI. Compute AUROC/AUPRC head-to-head vs every named baseline
  on the SAME held-out variants; compute calibration (ECE/Brier); report runtime. Commit the
  harness, fetchers, splits, and a results report. Re-runnable end to end.
- LEADERSHIP GATE (all must hold, with committed numbers): (1) XENON AUROC >= the best named
  baseline on the held-out set (target: top rank; minimum: statistical parity with the leader and
  strict improvement over the median); (2) best-in-class calibration (lowest ECE among compared
  methods); (3) every prediction carries a mechanistic, provenance-tracked explanation that the
  baselines do not provide; (4) upstream calling GIAB-validated.

PHASE 5 - ADVANCED HARDENING
- Reproducibility: deterministic, seeded, content-addressed inputs; signed provenance for every
  result; bit-reproducible benchmark re-runs.
- Security: treat all VCF/BAM/FASTA as untrusted - size limits, timeouts, safe parsing, no code
  execution; review the RCSB/network fetch paths; run bandit/pip-audit on genomics code.
- Scale & performance: stream large VCFs; index for random access; profile and meet a throughput
  target on a whole-exome VCF; report before/after.
- CI gates: no-stub grep gate; annotation-concordance gate; the leadership benchmark gate
  (separate job with the heavy extras); coverage gate.
- Documentation truth: rewrite genomics docs to the proven, narrow claim; remove any unproven
  language; mark experimental vs validated. (Mirror the standard already set in
  xenon/benchmarks/BENCHMARK_REPORT.md and the README rewrite.)

DELIVERABLES
A. De-Stubbing & Foundation Report (every simulated path removed, GIAB ingestion validated).
B. Annotation Validation Report (VEP/gnomAD/ClinVar concordance numbers).
C. Mechanism-Mapping Report (pathway recovery on curated pathogenic variants).
D. Interpretation-Engine Report (architecture + calibration: ECE/Brier/reliability).
E. Leadership Benchmark Report (head-to-head AUROC/AUPRC vs CADD/REVEL/AlphaMissense/SpliceAI/...
   on leakage-controlled held-out data, with the exact splits and reproduction commands).
F. Hardening Report (reproducibility, security, scale, CI gates).
G. Release recommendation on Prototype -> Research Platform -> Pre-Production -> Production
   Candidate -> Production Ready, justified ONLY by the committed benchmark numbers.

DEFINITION OF "INDUSTRY-LEADING" FOR THIS DIRECTIVE
A reproducible, third-party-runnable, leakage-controlled benchmark in which XENON ranks at or above
the best published variant-effect predictor on held-out ClinVar/ProteinGym/CAGI by AUROC/AUPRC,
achieves the best calibration among the compared methods, and uniquely emits a calibrated,
mechanistic, provenance-tracked explanation for every prediction - on top of GIAB-validated
upstream calling. Until those numbers are committed, the claim is not made.
```

---

## Grounding pointers (for the implementing agent)

| Concern | Reuse (real today) | Replace (stub today) |
|---|---|---|
| Bayesian prediction + calibration | `xenon/learning/forward_inference.py` | — |
| Mechanism graph / pathway model | `xenon/core/mechanism.py`, `xenon/learning/sbml_io.py` | — |
| Information-disruption features | `xenon/bioinformatics/information_fusion.py` (PID), `transfer_entropy.py` | — |
| Constraint reasoning | `xenon/bioinformatics/inference/neural_symbolic.py` | — |
| Provenance / audit | `xenon/bioinformatics/audit/audit_registry.py` | — |
| Benchmark pattern | `xenon/benchmarks/harness.py`, `BENCHMARK_REPORT.md` | extend with ClinVar/ProteinGym + baselines |
| VCF/BAM ingestion | `sequence_ancestrydna.py` (SNP arrays, real) | `wgs_pipeline.py` `_parse_vcf`/`align_reads` (add cyvcf2/pysam) |
| Variant calling/annotation | — | `wgs_pipeline.py:547-690` (`_call_snps_indels`, `_annotate_single_variant`), `giab_validation_framework.py:191` |
| Pop-frequency / haplogroup / lineage | — | `genomic_rarity_engine.py:309-714` (heuristic/dummy) |

## New optional dependencies (pin in a `xenon/requirements-genomics.txt`)
`cyvcf2`, `pysam`, `pyensembl` (or a VEP interface), `scikit-learn`; benchmark fetchers for GIAB,
ClinVar, ProteinGym.

## First runnable checkpoint
Phase 0 + Phase 1: ingest a real GIAB HG002 VCF with cyvcf2, reconcile against the GIAB truth set
(precision/recall reported), and produce VEP-concordant consequence annotations on a held-out
variant set — with the CI no-stub grep gate green. No interpretation/leadership claim is valid
before this foundation is real and tested.
