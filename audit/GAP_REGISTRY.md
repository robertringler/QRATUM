# QRATUM Gap Registry

_Commit_: `8fc58a9107334a3b53b69a47580df64b185d3317`  _Generated_: 2026-04-29T22:27:58Z

## Counts

- Total: **80**
- By severity: `block`=7, `info`=1, `major`=54, `minor`=18
- By area: `security`=27, `quasim`=12, `cicd`=7, `qnx`=3, `qubic`=3, `sdk`=3, `theory`=3, `code`=3, `qagents`=2, `qcore`=2, `qos`=2, `qreal`=2, `cli`=2, `production_workflow_checkpointing.py`=1, `qdl`=1, `qratum-rust`=1, `qstack`=1, `quasim/ciir`=1, `tests`=1, `inventory`=1, `artifacts`=1, `documentation`=1
- By proposed fix class: `finish-stub`=34, `harden`=34, `write-proof`=3, `wire-seam`=2, `write-test`=2, `delete-orphan`=2, `refactor`=2, `document`=1

## Detail (grouped by area, severity)

### security

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-SEC-SECRET-001` | minor | harden | Hardcoded credential pattern at README_OLD.md:186 |
| `GAP-SEC-SECRET-002` | minor | harden | Hardcoded credential pattern at docs-site/docs/advanced/troubleshooting.md:127 |
| `GAP-SEC-SECRET-003` | minor | harden | Hardcoded credential pattern at docs-site/docs/getting-started/installation.md:120 |
| `GAP-SEC-SECRET-004` | minor | harden | Hardcoded credential pattern at docs-site/docs/reference/api-reference.md:46 |
| `GAP-SEC-SECRET-005` | minor | harden | Hardcoded credential pattern at docs-site/docs/tutorials/vqe-h2-molecule.md:138 |
| `GAP-SEC-SECRET-006` | minor | harden | Hardcoded credential pattern at docs/API_REFERENCE.md:430 |
| `GAP-SEC-SECRET-007` | minor | harden | Hardcoded credential pattern at docs/AUTO_MERGE_SYSTEM.md:57 |
| `GAP-SEC-SECRET-008` | minor | harden | Hardcoded credential pattern at docs/PR_AUTO_RESOLUTION.md:80 |
| `GAP-SEC-SECRET-009` | minor | harden | Hardcoded credential pattern at docs/PR_AUTO_RESOLUTION.md:87 |
| `GAP-SEC-SECRET-010` | minor | harden | Hardcoded credential pattern at figma/README.md:117 |
| `GAP-SEC-SECRET-011` | block | harden | Hardcoded credential pattern at qratum/exascale/security/pqc.py:72 |
| `GAP-SEC-SECRET-012` | block | harden | Hardcoded credential pattern at quasim/qunimbus/auth.py:460 |
| `GAP-SEC-SECRET-013` | block | harden | Hardcoded credential pattern at tests/qunimbus/test_qunimbus_enhancements.py:347 |
| `GAP-SEC-SECRET-014` | block | harden | Hardcoded credential pattern at tests/test_sdk.py:87 |
| `GAP-SEC-UNSAFE-001` | major | harden | UNSAFE-SUBPROCESS at .github/workflows/sbom.yml:182 |
| `GAP-SEC-UNSAFE-002` | major | harden | UNSAFE-EVAL at aion/runtime/**init**.py:287 |
| `GAP-SEC-UNSAFE-003` | major | harden | UNSAFE-PICKLE at production_workflow_checkpointing.py:279 |
| `GAP-SEC-UNSAFE-004` | major | harden | UNSAFE-PICKLE at production_workflow_checkpointing.py:282 |
| `GAP-SEC-UNSAFE-005` | major | harden | UNSAFE-PICKLE at qratum/temporal/state.py:150 |
| `GAP-SEC-UNSAFE-006` | major | harden | UNSAFE-SUBPROCESS at scripts/validate_quasim_master.py:29 |
| `GAP-DEP-001` | major | harden | Vulnerable pinned dependency torch==2.1.2: heap buffer overflow (affected <2.2.0) |
| `GAP-DEP-002` | major | harden | Vulnerable pinned dependency torch==2.1.2: use-after-free (affected <2.2.0) |
| `GAP-DEP-003` | major | harden | Vulnerable pinned dependency torch==2.1.2: torch.load weights_only RCE (affected <2.6.0) |
| `GAP-DEP-004` | major | harden | Vulnerable pinned dependency qiskit==0.45.1: QPY file DoS (affected <1.3.0) |
| `GAP-DEP-005` | major | harden | Vulnerable pinned dependency qiskit==0.45.1: QPY arbitrary code execution (affected <=1.4.1) |
| `GAP-DEP-006` | major | harden | Vulnerable pinned dependency gunicorn==21.2.0: HTTP request smuggling (affected <22.0.0) |
| `GAP-DEP-007` | major | harden | Vulnerable pinned dependency gunicorn==21.2.0: endpoint restriction bypass (affected <22.0.0) |

**Evidence:**

- `GAP-SEC-SECRET-001` — `README_OLD.md:186 :: ibmq_token="YOUR_API_TOKEN_HERE",`
- `GAP-SEC-SECRET-002` — `docs-site/docs/advanced/troubleshooting.md:127 :: ibmq_token="your_actual_token"`
- `GAP-SEC-SECRET-003` — `docs-site/docs/getting-started/installation.md:120 :: ibmq_token="YOUR_API_TOKEN_HERE",`
- `GAP-SEC-SECRET-004` — `docs-site/docs/reference/api-reference.md:46 :: ibmq_token="your_api_token",`
- `GAP-SEC-SECRET-005` — `docs-site/docs/tutorials/vqe-h2-molecule.md:138 :: ibmq_token="YOUR_API_TOKEN",`
- `GAP-SEC-SECRET-006` — `docs/API_REFERENCE.md:430 :: client = QRATUMClient(api_key="your-api-key")`
- `GAP-SEC-SECRET-007` — `docs/AUTO_MERGE_SYSTEM.md:57 :: export GITHUB_TOKEN="your_github_token"`
- `GAP-SEC-SECRET-008` — `docs/PR_AUTO_RESOLUTION.md:80 :: export GITHUB_TOKEN="your_token_here"`
- `GAP-SEC-SECRET-009` — `docs/PR_AUTO_RESOLUTION.md:87 :: export GITHUB_TOKEN="your_token_here"`
- `GAP-SEC-SECRET-010` — `figma/README.md:117 :: export FIGMA_TOKEN="your_figma_token"`
- `GAP-SEC-SECRET-011` — `qratum/exascale/security/pqc.py:72 :: SHARED_SECRET = "Shared Secret"`
- `GAP-SEC-SECRET-012` — `quasim/qunimbus/auth.py:460 :: export QUNIMBUS_TOKEN="your-jwt-token"`
- `GAP-SEC-SECRET-013` — `tests/qunimbus/test_qunimbus_enhancements.py:347 :: token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4f`
- `GAP-SEC-SECRET-014` — `tests/test_sdk.py:87 :: api_key="test-key-123",`
- `GAP-SEC-UNSAFE-001` — `.github/workflows/sbom.yml:182 :: result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)`
- `GAP-SEC-UNSAFE-002` — `aion/runtime/__init__.py:287 :: result = self.eval(line)`
- `GAP-SEC-UNSAFE-003` — `production_workflow_checkpointing.py:279 :: data = pickle.load(f)`
- `GAP-SEC-UNSAFE-004` — `production_workflow_checkpointing.py:282 :: data = pickle.load(f)`
- `GAP-SEC-UNSAFE-005` — `qratum/temporal/state.py:150 :: state_dict = pickle.loads(data)`
- `GAP-SEC-UNSAFE-006` — `scripts/validate_quasim_master.py:29 :: result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)`
- `GAP-DEP-001` — `requirements-prod.txt :: torch==2.1.2`
- `GAP-DEP-002` — `requirements-prod.txt :: torch==2.1.2`
- `GAP-DEP-003` — `requirements-prod.txt :: torch==2.1.2`
- `GAP-DEP-004` — `requirements-prod.txt :: qiskit==0.45.1`
- `GAP-DEP-005` — `requirements-prod.txt :: qiskit==0.45.1`
- `GAP-DEP-006` — `requirements-prod.txt :: gunicorn==21.2.0`
- `GAP-DEP-007` — `requirements-prod.txt :: gunicorn==21.2.0`

### quasim

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-015` | major | finish-stub | NotImplementedError stub at quasim/hcal/backends/**init**.py:89 |
| `GAP-STUB-016` | major | finish-stub | NotImplementedError stub at quasim/hcal/backends/**init**.py:102 |
| `GAP-STUB-017` | major | finish-stub | NotImplementedError stub at quasim/hcal/backends/**init**.py:114 |
| `GAP-STUB-018` | major | finish-stub | NotImplementedError stub at quasim/hcal/backends/**init**.py:126 |
| `GAP-STUB-019` | major | finish-stub | NotImplementedError stub at quasim/opt/post_dijkstra_sssp.py:381 |
| `GAP-STUB-020` | major | finish-stub | NotImplementedError stub at quasim/opt/post_dijkstra_sssp.py:395 |
| `GAP-STUB-021` | major | finish-stub | NotImplementedError stub at quasim/opt/problems.py:47 |
| `GAP-STUB-022` | major | finish-stub | NotImplementedError stub at quasim/quantum/core.py:301 |
| `GAP-STUB-023` | major | finish-stub | NotImplementedError stub at quasim/quantum/vqe_molecule.py:244 |
| `GAP-STUB-024` | major | finish-stub | NotImplementedError stub at quasim/quantum/vqe_molecule.py:246 |
| `GAP-STUB-025` | major | finish-stub | NotImplementedError stub at quasim/qunimbus/auth.py:166 |
| `GAP-STUB-026` | major | finish-stub | NotImplementedError stub at quasim/qunimbus/auth.py:407 |

**Evidence:**

- `GAP-STUB-015` — `quasim/hcal/backends/__init__.py:89 :: raise NotImplementedError`
- `GAP-STUB-016` — `quasim/hcal/backends/__init__.py:102 :: raise NotImplementedError`
- `GAP-STUB-017` — `quasim/hcal/backends/__init__.py:114 :: raise NotImplementedError`
- `GAP-STUB-018` — `quasim/hcal/backends/__init__.py:126 :: raise NotImplementedError`
- `GAP-STUB-019` — `quasim/opt/post_dijkstra_sssp.py:381 :: raise NotImplementedError`
- `GAP-STUB-020` — `quasim/opt/post_dijkstra_sssp.py:395 :: raise NotImplementedError`
- `GAP-STUB-021` — `quasim/opt/problems.py:47 :: raise NotImplementedError("Subclasses must implement evaluate()")`
- `GAP-STUB-022` — `quasim/quantum/core.py:301 :: raise NotImplementedError("cuQuantum backend planned for Phase 2 (2026)")`
- `GAP-STUB-023` — `quasim/quantum/vqe_molecule.py:244 :: raise NotImplementedError("LiH VQE planned for Phase 2 (2026)")`
- `GAP-STUB-024` — `quasim/quantum/vqe_molecule.py:246 :: raise NotImplementedError("BeH2 VQE planned for Phase 2 (2026)")`
- `GAP-STUB-025` — `quasim/qunimbus/auth.py:166 :: raise NotImplementedError("JWT refresh scheduled for Q1 2026")`
- `GAP-STUB-026` — `quasim/qunimbus/auth.py:407 :: raise NotImplementedError(`

### cicd

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-CTR-NOROOT-Dockerfile` | major | harden | Dockerfile runs as root (no USER directive) |
| `GAP-CTR-HEALTH-Dockerfile` | minor | harden | Dockerfile lacks HEALTHCHECK |
| `GAP-CTR-NOROOT-Dockerfile.sandbox-platform` | major | harden | Dockerfile.sandbox-platform runs as root (no USER directive) |
| `GAP-CTR-HEALTH-Dockerfile.sandbox-platform` | minor | harden | Dockerfile.sandbox-platform lacks HEALTHCHECK |
| `GAP-CTR-NOROOT-Dockerfile.sandbox-qradle` | major | harden | Dockerfile.sandbox-qradle runs as root (no USER directive) |
| `GAP-CTR-HEALTH-Dockerfile.sandbox-qradle` | minor | harden | Dockerfile.sandbox-qradle lacks HEALTHCHECK |
| `GAP-DEP-PIN-001` | major | harden | requirements.txt has 4 unpinned top-level dependencies |

**Evidence:**

- `GAP-CTR-NOROOT-Dockerfile` — `Dockerfile :: no`USER`directive present`
- `GAP-CTR-HEALTH-Dockerfile` — `Dockerfile :: no`HEALTHCHECK`directive`
- `GAP-CTR-NOROOT-Dockerfile.sandbox-platform` — `Dockerfile.sandbox-platform :: no`USER`directive present`
- `GAP-CTR-HEALTH-Dockerfile.sandbox-platform` — `Dockerfile.sandbox-platform :: no`HEALTHCHECK`directive`
- `GAP-CTR-NOROOT-Dockerfile.sandbox-qradle` — `Dockerfile.sandbox-qradle :: no`USER`directive present`
- `GAP-CTR-HEALTH-Dockerfile.sandbox-qradle` — `Dockerfile.sandbox-qradle :: no`HEALTHCHECK`directive`
- `GAP-DEP-PIN-001` — `requirements.txt:1-4 :: pandas / matplotlib / seaborn / numpy (no`==`pin)`

### qnx

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-006` | major | finish-stub | NotImplementedError stub at qnx/backends/cpu.py:36 |
| `GAP-STUB-007` | major | finish-stub | NotImplementedError stub at qnx/backends/gpu.py:36 |
| `GAP-STUB-008` | major | finish-stub | NotImplementedError stub at qnx/backends/qpu.py:36 |

**Evidence:**

- `GAP-STUB-006` — `qnx/backends/cpu.py:36 :: raise NotImplementedError(`
- `GAP-STUB-007` — `qnx/backends/gpu.py:36 :: raise NotImplementedError(`
- `GAP-STUB-008` — `qnx/backends/qpu.py:36 :: raise NotImplementedError(`

### qubic

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-027` | major | finish-stub | NotImplementedError stub at qubic/visualization/adapters/mesh.py:63 |
| `GAP-STUB-028` | major | finish-stub | NotImplementedError stub at qubic/visualization/adapters/timeseries.py:70 |
| `GAP-STUB-029` | major | finish-stub | NotImplementedError stub at qubic/visualization/adapters/tire.py:79 |

**Evidence:**

- `GAP-STUB-027` — `qubic/visualization/adapters/mesh.py:63 :: raise NotImplementedError("File-based loading not yet implemented. Use dictionary input.")`
- `GAP-STUB-028` — `qubic/visualization/adapters/timeseries.py:70 :: raise NotImplementedError("File-based loading not yet implemented. Use list input.")`
- `GAP-STUB-029` — `qubic/visualization/adapters/tire.py:79 :: raise NotImplementedError("File-based loading not yet implemented. Use dictionary input.")`

### sdk

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-030` | major | finish-stub | NotImplementedError stub at sdk/ansys/quasim_ansys_adapter.py:594 |
| `GAP-STUB-031` | major | finish-stub | NotImplementedError stub at sdk/ansys/quasim_ansys_adapter.py:813 |
| `GAP-STUB-032` | major | finish-stub | NotImplementedError stub at sdk/ansys/quasim_ansys_adapter.py:860 |

**Evidence:**

- `GAP-STUB-030` — `sdk/ansys/quasim_ansys_adapter.py:594 :: raise NotImplementedError("CDB file import not yet implemented")`
- `GAP-STUB-031` — `sdk/ansys/quasim_ansys_adapter.py:813 :: raise NotImplementedError(f"Export format '{format}' not yet implemented")`
- `GAP-STUB-032` — `sdk/ansys/quasim_ansys_adapter.py:860 :: raise NotImplementedError("Preconditioner registration not yet implemented")`

### theory

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-DOC-MONO-001` | block | write-proof | CIIR monograph chapters 9-22 are missing from repository |
| `GAP-DOC-MONO-002` | block | write-proof | Monograph main file 'ciir_monograph.tex' (referenced by prior task notes) does not exist; actual file is 'ciir_quasim_formalization.tex' which has no \input/\in |
| `GAP-THEORY-001` | major | write-proof | 397 formal claims (theorem/lemma/proposition/corollary/definition/proof/algorithm) extracted from .tex with no automated linkage to code or tests |

**Evidence:**

- `GAP-DOC-MONO-001` — `manuscripts/ciir_monograph/chapters/ :: only ch03..ch08 present; no ch21_multi_qubit_controller.tex, no ch22_unified_extensions.tex`
- `GAP-DOC-MONO-002` — `manuscripts/ciir_monograph/ciir_quasim_formalization.tex :: standalone, no chapter inclusion`
- `GAP-THEORY-001` — `audit/theory/claims_matrix.csv`

### code

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-CODE-PRINT-001` | info | refactor | 5945 `print(...)` calls outside tests/demos suggest unmigrated debug logging |
| `GAP-CODE-TODO-001` | minor | finish-stub | 196 TODO markers across the codebase |
| `GAP-CODE-EXCEPT-001` | minor | refactor | 13 bare-except + 116 broad-Exception handlers — risk of swallowed errors |

**Evidence:**

- `GAP-CODE-PRINT-001` — `audit/code/findings.jsonl :: rule_id=DBG-PRINT`
- `GAP-CODE-TODO-001` — `audit/code/findings.jsonl :: rule_id=STUB-TODO`
- `GAP-CODE-EXCEPT-001` — `audit/code/findings.jsonl :: rule_id IN (BARE-EXCEPT, BROAD-EXCEPT)`

### qagents

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-002` | major | finish-stub | NotImplementedError stub at qagents/strategy.py:23 |
| `GAP-CIIR-RIC-V2-001` | major | write-test | qagents/reality_interface_v2.py exists but no companion test_ric_v2.py is enumerated under tests/ (despite stored memory of 18 tests); verify regression coverag |

**Evidence:**

- `GAP-STUB-002` — `qagents/strategy.py:23 :: raise NotImplementedError`
- `GAP-CIIR-RIC-V2-001` — `tests/ :: contains test_ric_v2.py — confirm symbol mapping in audit/tests/coverage_gap.csv`

### qcore

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-003` | major | finish-stub | NotImplementedError stub at qcore/hamiltonian.py:105 |
| `GAP-STUB-004` | major | finish-stub | NotImplementedError stub at qcore/hamiltonian.py:125 |

**Evidence:**

- `GAP-STUB-003` — `qcore/hamiltonian.py:105 :: raise NotImplementedError(`
- `GAP-STUB-004` — `qcore/hamiltonian.py:125 :: raise NotImplementedError(`

### qos

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-009` | major | finish-stub | NotImplementedError stub at qos/envelopes.py:49 |
| `GAP-STUB-010` | major | finish-stub | NotImplementedError stub at qos/policy.py:47 |

**Evidence:**

- `GAP-STUB-009` — `qos/envelopes.py:49 :: raise NotImplementedError(`
- `GAP-STUB-010` — `qos/policy.py:47 :: raise NotImplementedError(`

### qreal

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-012` | major | finish-stub | NotImplementedError stub at qreal/base_adapter.py:55 |
| `GAP-STUB-013` | major | finish-stub | NotImplementedError stub at qreal/base_adapter.py:60 |

**Evidence:**

- `GAP-STUB-012` — `qreal/base_adapter.py:55 :: raise NotImplementedError`
- `GAP-STUB-013` — `qreal/base_adapter.py:60 :: raise NotImplementedError`

### cli

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-CIIR-RUN-001` | major | wire-seam | Top-level CLI runner run_multi_qubit_ciir.py is missing |
| `GAP-CIIR-RUN-002` | major | wire-seam | Top-level CLI runner run_falsification.py is missing |

**Evidence:**

- `GAP-CIIR-RUN-001` — `ROOT :: file does not exist (referenced by prior implementation notes & problem statement)`
- `GAP-CIIR-RUN-002` — `ROOT :: file does not exist (referenced by problem statement § CLI surface)`

### production_workflow_checkpointing.py

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-001` | major | finish-stub | NotImplementedError stub at production_workflow_checkpointing.py:519 |

**Evidence:**

- `GAP-STUB-001` — `production_workflow_checkpointing.py:519 :: raise NotImplementedError("Subclass must implement _execute_stages")`

### qdl

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-005` | major | finish-stub | NotImplementedError stub at qdl/ast.py:11 |

**Evidence:**

- `GAP-STUB-005` — `qdl/ast.py:11 :: raise NotImplementedError`

### qratum-rust

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-011` | major | finish-stub | NotImplementedError stub at qratum-rust/src/q_substrate.rs:670 |

**Evidence:**

- `GAP-STUB-011` — `qratum-rust/src/q_substrate.rs:670 :: raise NotImplementedError("Generated stub for: {}")"#, desc, desc),`

### qstack

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-STUB-014` | major | finish-stub | NotImplementedError stub at qstack/alignment/policies.py:25 |

**Evidence:**

- `GAP-STUB-014` — `qstack/alignment/policies.py:25 :: raise NotImplementedError`

### quasim/ciir

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-CIIR-MQ-001` | block | finish-stub | Promised quasim/ciir/multi_qubit/ subsystem (controller, density_matrix, fractal_qec, hybrid_sim, stability, plots, non_markovian, geometric_control, hardware_p |

**Evidence:**

- `GAP-CIIR-MQ-001` — `quasim/ciir/ :: only plasma/, simulation/, integration/, ciir.py present; no multi_qubit/ directory exists`

### tests

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-TEST-PUB-001` | major | write-test | 2218 public Python symbols out of 5977 (37%) have no textual reference in any test file |

**Evidence:**

- `GAP-TEST-PUB-001` — `audit/tests/coverage_gap.csv :: column referenced_by_test='N'`

### inventory

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-ORPH-001` | minor | delete-orphan | 179 code files appear to have no inbound import or doc reference |

**Evidence:**

- `GAP-ORPH-001` — `audit/inventory/orphans.txt`

### artifacts

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-ART-001` | minor | document | 24 `.zip` artifacts checked into repo with no producer/consumer manifest |

**Evidence:**

- `GAP-ART-001` — `audit/artifacts/manifest.csv`

### documentation

| ID | Sev | Fix class | Title |
|----|-----|-----------|-------|
| `GAP-DOC-README-OLD` | minor | delete-orphan | README_OLD.md retained alongside README.md — duplication risk; placeholder secret on line 186 |

**Evidence:**

- `GAP-DOC-README-OLD` — `README_OLD.md:186 :: ibmq_token="YOUR_API_TOKEN_HERE"`
