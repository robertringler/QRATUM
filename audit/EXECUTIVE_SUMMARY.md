# QRATUM Audit — Executive Summary
_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

## Scope
- Tracked files audited: **3991** (every file inventoried with sha256 + line count; see `audit/inventory/file_index.jsonl`).
- Python files parsed via AST: **2089** with **5977** public symbols.
- Workflows audited: **56** under `.github/workflows/`.
- Manuscript claims extracted: **397** from `.tex` (theorem/lemma/proposition/corollary/definition/proof/algorithm).
- Binary/figure/data artifacts inventoried: **202** (see `audit/artifacts/manifest.csv`).

## Methodology disclosure (honest)
- **Full machine coverage**: every tracked file enumerated, sha256-hashed, line-counted, language-classified, and grep+AST-scanned for the rule set in `audit/code/findings.jsonl` and `audit/security/findings.jsonl`. `lines_read == lines_total` in `file_index.jsonl` for every non-binary file ≤ 5 MB.
- **Targeted human-style review** is concentrated on the CIIR/CRS/RIC/MVRI/Trajectory/RealWorld-Bridge/Control-Geometry/Adapters surfaces (~30 files, line-by-line) and on the manuscript `manuscripts/ciir_monograph/chapters/ch03..ch08.tex`.
- Exhaustive line-by-line *human* reading of all 3991 files in a single agent session is not feasible; this audit is driven by deterministic automated rules with targeted deep reads where human judgment is required.

## Headline counts
- **Gap registry**: 80 total — `block`=7, `info`=1, `major`=54, `minor`=18
- **Code findings (info+minor+major)**: 6615; `STUB-NOTIMPL`=36, `STUB-TODO`=196.
- **Security findings**: 28; CWE-tagged in `audit/security/findings.jsonl`.
- **Vulnerable pinned dependencies (advisory DB)**: torch 2.1.2 (3 advisories), qiskit 0.45.1 (multiple QPY RCE/DoS), gunicorn 21.2.0 (request smuggling). See `audit/security/sbom_diff.md`.
- **Untested public symbols**: 2218 of 5977 (37%).
- **Orphan code files** (no inbound import/doc reference): 179.

## Top blockers (≤ 20)
1. **GAP-SEC-SECRET-011** [block/harden] — Hardcoded credential pattern at qratum/exascale/security/pqc.py:72 (evidence: `qratum/exascale/security/pqc.py:72 :: SHARED_SECRET = "Shared Secret"`)
2. **GAP-SEC-SECRET-012** [block/harden] — Hardcoded credential pattern at quasim/qunimbus/auth.py:460 (evidence: `quasim/qunimbus/auth.py:460 :: export QUNIMBUS_TOKEN="your-jwt-token"`)
3. **GAP-SEC-SECRET-013** [block/harden] — Hardcoded credential pattern at tests/qunimbus/test_qunimbus_enhancements.py:347 (evidence: `tests/qunimbus/test_qunimbus_enhancements.py:347 :: token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4f`)
4. **GAP-SEC-SECRET-014** [block/harden] — Hardcoded credential pattern at tests/test_sdk.py:87 (evidence: `tests/test_sdk.py:87 :: api_key="test-key-123",`)
5. **GAP-DOC-MONO-001** [block/write-proof] — CIIR monograph chapters 9-22 are missing from repository (evidence: `manuscripts/ciir_monograph/chapters/ :: only ch03..ch08 present; no ch21_multi_qubit_controller.tex, no ch22_unified_extensions.tex`)
6. **GAP-DOC-MONO-002** [block/write-proof] — Monograph main file 'ciir_monograph.tex' (referenced by prior task notes) does not exist; actual file is 'ciir_quasim_formalization.tex' which has no \input/\include of chapter files (evidence: `manuscripts/ciir_monograph/ciir_quasim_formalization.tex :: standalone, no chapter inclusion`)
7. **GAP-CIIR-MQ-001** [block/finish-stub] — Promised quasim/ciir/multi_qubit/ subsystem (controller, density_matrix, fractal_qec, hybrid_sim, stability, plots, non_markovian, geometric_control, hardware_params, tensor_network, alpha_derivation, causal_claim, unified_evolution, falsification) is absent (evidence: `quasim/ciir/ :: only plasma/, simulation/, integration/, ciir.py present; no multi_qubit/ directory exists`)
8. **GAP-STUB-001** [major/finish-stub] — NotImplementedError stub at production_workflow_checkpointing.py:519 (evidence: `production_workflow_checkpointing.py:519 :: raise NotImplementedError("Subclass must implement _execute_stages")`)
9. **GAP-STUB-002** [major/finish-stub] — NotImplementedError stub at qagents/strategy.py:23 (evidence: `qagents/strategy.py:23 :: raise NotImplementedError`)
10. **GAP-STUB-003** [major/finish-stub] — NotImplementedError stub at qcore/hamiltonian.py:105 (evidence: `qcore/hamiltonian.py:105 :: raise NotImplementedError(`)
11. **GAP-STUB-004** [major/finish-stub] — NotImplementedError stub at qcore/hamiltonian.py:125 (evidence: `qcore/hamiltonian.py:125 :: raise NotImplementedError(`)
12. **GAP-STUB-005** [major/finish-stub] — NotImplementedError stub at qdl/ast.py:11 (evidence: `qdl/ast.py:11 :: raise NotImplementedError`)
13. **GAP-STUB-006** [major/finish-stub] — NotImplementedError stub at qnx/backends/cpu.py:36 (evidence: `qnx/backends/cpu.py:36 :: raise NotImplementedError(`)
14. **GAP-STUB-007** [major/finish-stub] — NotImplementedError stub at qnx/backends/gpu.py:36 (evidence: `qnx/backends/gpu.py:36 :: raise NotImplementedError(`)
15. **GAP-STUB-008** [major/finish-stub] — NotImplementedError stub at qnx/backends/qpu.py:36 (evidence: `qnx/backends/qpu.py:36 :: raise NotImplementedError(`)
16. **GAP-STUB-009** [major/finish-stub] — NotImplementedError stub at qos/envelopes.py:49 (evidence: `qos/envelopes.py:49 :: raise NotImplementedError(`)
17. **GAP-STUB-010** [major/finish-stub] — NotImplementedError stub at qos/policy.py:47 (evidence: `qos/policy.py:47 :: raise NotImplementedError(`)
18. **GAP-STUB-011** [major/finish-stub] — NotImplementedError stub at qratum-rust/src/q_substrate.rs:670 (evidence: `qratum-rust/src/q_substrate.rs:670 :: raise NotImplementedError("Generated stub for: {}")"#, desc, desc),`)
19. **GAP-STUB-012** [major/finish-stub] — NotImplementedError stub at qreal/base_adapter.py:55 (evidence: `qreal/base_adapter.py:55 :: raise NotImplementedError`)

## Hand-off
`audit/GAP_REGISTRY.jsonl` (80 entries, deduplicated, every entry has id/area/severity/evidence/proposed_fix_class/proposed_owner_subsystem) is the unit of work for Prompt #2.

**AUDIT COMPLETE — handing GAP_REGISTRY to Prompt #2.**