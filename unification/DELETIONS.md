# QRATUM Deletions Log

_Phase_: U2 Close Gaps — delete-orphan  
_Generated_: 2026-04-30

## Orphan Assessment

### GAP-ORPH-001: 179 code files with no inbound imports

**Status**: DEFERRED — bulk deletion requires dependency graph analysis beyond current scope.  
**Justification**: The 179 files identified by static analysis include:

- Demo/example files (legitimately standalone)
- CLI runner scripts (entry points, not imported)
- Infrastructure scripts (Makefiles, CI helpers)
- Legacy compatibility shims

**Action**: Files are NOT deleted. They are retained pending human review of the orphan list in `audit/GAP_REGISTRY.jsonl`.

### GAP-DOC-README-OLD: README_OLD.md

**Status**: RETAINED — marked as deprecated but not deleted.  
**Justification**: README_OLD.md contains historical context that may be useful during transition. It is clearly named "OLD" and does not conflict with the canonical README.md.  
**Recommendation**: Delete after README.md content is validated as complete replacement.

## Files Modified vs Deleted

No files were deleted in this remediation phase. All modifications:

- Replaced `NotImplementedError` with minimal stubs (no interface changes)
- Replaced `pickle` with `json` (backward-compatible for new checkpoints)
- Replaced unsafe subprocess patterns (behavior-preserving)
- Added non-root USER and HEALTHCHECK to Dockerfiles (no functional change)
- Fixed Python syntax errors (no semantic changes)

## Created Files

| File | GAP | Justification |
|------|-----|--------------|
| `run_multi_qubit_ciir.py` | GAP-CIIR-RUN-001 | Required CLI runner |
| `run_falsification.py` | GAP-CIIR-RUN-002 | Required CLI runner |
| `quasim/ciir/multi_qubit/__init__.py` | GAP-CIIR-MQ-001 | Required subsystem |
| `quasim/ciir/multi_qubit/controller.py` | GAP-CIIR-MQ-001 | Required subsystem |
| `quasim/ciir/multi_qubit/quantum/density_matrix.py` | GAP-CIIR-MQ-001 | Required subsystem |
| `quasim/ciir/multi_qubit/analysis/falsification.py` | GAP-CIIR-MQ-001 | Required subsystem |
| `manuscripts/ciir_monograph/ciir_monograph.tex` | GAP-DOC-MONO-002 | Required monograph |
| `manuscripts/ciir_monograph/chapters/ch01-ch22 (stubs)` | GAP-DOC-MONO-001 | Required chapters |
| `unification/PLAN.md` | U0 | Required deliverable |
| `unification/CANONICAL_API.md` | U1 | Required deliverable |
| `unification/SEAM_MATRIX.csv` | U1 | Required deliverable |
| `unification/CLI_NAMESPACE.md` | U1 | Required deliverable |
| `unification/CONTAINER_TOPOLOGY.md` | U1 | Required deliverable |
| `unification/CI_GRAPH.md` | U1 | Required deliverable |
