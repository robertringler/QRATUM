# 20 High-Power Expert Prompts for QRATUM Development

## Automation Prompts for Claude Opus 4.5 / GPT-4

**Created:** December 19, 2025  
**Purpose:** Automate deep technical tasks for QRATUM platform development

---

## Core Analysis Prompts

### 1. Full Repository Manifest and Evidence Extraction

```
Clone robertringler/QRATUM, enumerate all branches/tags/PRs/issues, and produce a CSV manifest of every file with: path, size, last commit hash, author, test coverage indicator. For each file add a one-line summary of purpose and a risk rating (LOW/MEDIUM/HIGH/CRITICAL). Include package.json/pyproject.toml dependencies with versions and licenses. Output: manifest.csv with 1000+ entries.
```

### 2. Claim Validation Extractor

```
Parse all markdown documentation in robertringler/QRATUM (README.md, docs/*, *.md files), extract every factual claim about performance, compliance, capabilities, or features. For each claim, locate supporting code/tests/commits. Output: validation-manifest.csv with columns: claim_text, source_file, evidence_file, line_range, validation_status (VALIDATED/UNVALIDATED/CONTRADICTED), commit_hash.
```

### 3. Dependency and License Audit

```
Generate a complete dependency graph for robertringler/QRATUM including Python (requirements.txt, pyproject.toml), JavaScript (package.json), Rust (Cargo.toml), and system packages. For each dependency: name, version, license, known CVEs, SBOM entry, and whether used at runtime or dev-only. Prioritize packages with GPL/AGPL licenses or security issues. Output: dependency-audit.json and license-conflicts.txt.
```

### 4. Open/Closed PR Triage

```
Analyze all open and closed PRs in robertringler/QRATUM. For each PR output: PR number, title, status, mergeability, file changes count, test status, conflicts (if any), required rebase steps, age in days, and suggested action (MERGE/CLOSE/REWORK). Provide exact git commands for each action. Output: pr-triage.md with actionable commands.
```

### 5. Automated Refactor Plan Generator

```
For each module in robertringler/QRATUM/quasim, propose a refactor plan: file moves (if needed), API changes, tests to add, performance targets, and a risk assessment. Produce patch files for the top 3 highest-impact modules. Include before/after code snippets. Output: refactor-plan.md + patches/module-*.patch.
```

---

## Documentation and Compliance

### 6. Whitepaper Consolidation

```
Collect all existing whitepapers and executive summaries in robertringler/QRATUM (search for *.md with "whitepaper" or "executive" in name/content). Reconcile contradictions, merge duplicates, and produce a single 50-page technical whitepaper with: architecture diagrams (mermaid), threat model, data flow, API surface, compliance matrix, performance benchmarks, and migration plan. Cite evidence from repo files. Output: technical-whitepaper.md (12,000+ words).
```

### 7. Executive Summary Rewrite

```
Produce a one-page executive summary for robertringler/QRATUM suitable for C-suite and investors. Include: product value proposition, market opportunity ($12B+ TAM), technical differentiation (CQCC category), competitive analysis table, traction metrics, 12-month roadmap, financial projections (ARR), and investment ask. Use data from repository commits, test counts, and compliance docs. Output: executive-summary.md (1-2 pages).
```

### 8. Compliance Gap Analysis

```
Audit robertringler/QRATUM against DO-178C Level A, NIST 800-53 Rev 5 (HIGH), CMMC 2.0 Level 2, and DFARS requirements. For each control, determine: IMPLEMENTED/PARTIAL/MISSING, evidence file, gap description, remediation steps. Prioritize safety-critical gaps. Output: compliance-gap-analysis.csv + remediation-plan.md with exact file changes and tests needed.
```

---

## CI/CD and DevOps

### 9. CI/CD and Release Plan

```
Audit all CI pipelines in robertringler/QRATUM (.github/workflows/*). Produce a hardened CI/CD plan with: build matrix (Python 3.10-3.12, Ubuntu/Mac/Windows), test stages (unit/integration/e2e), artifact signing (cosign), release tagging (semantic versioning), SBOM generation, and desktop packaging steps (Electron/Tauri). Output: ci-cd-plan.md + .github/workflows/release.yml.
```

### 10. Security Hotfix PR Template

```
When a vulnerability is found in robertringler/QRATUM, generate a complete hotfix package: branch name (hotfix/CVE-YYYY-NNNN), patch diff with exact file changes, unit tests to validate fix, regression tests, and PR template with CVE references, CVSS score, affected versions, mitigation steps, and rollback plan. Output: hotfix-template.md + patches/cve-fix.patch.
```

---

## Testing and Quality

### 11. Automated Test Generation

```
For each module in robertringler/QRATUM lacking test coverage (<80%), generate pytest test skeletons with: test fixtures, mocks for external dependencies, assertions for happy path and edge cases, parametrized tests for input variations. Aim for 90%+ coverage. Output: tests/test_<module>.py files (one per module).
```

### 12. Performance Profiling Plan

```
Identify performance hotspots in robertringler/QRATUM by analyzing code for: nested loops, large data structures, I/O operations, and quantum circuit depth. Propose profiling steps (cProfile, line_profiler), generate benchmark scripts with target KPIs (latency <100ms, throughput >1000 ops/sec), and optimization recommendations. Output: performance-profiling-plan.md + benchmarks/*.py.
```

### 13. Package Cleanup and Orphan Detection

```
List all orphaned packages and unused modules in robertringler/QRATUM by analyzing: import statements, dependency graphs, test references, and CI usage. For each orphan, propose deletion or consolidation with exact git commands and deprecation notices. Output: package-cleanup.md with git rm commands and impact analysis.
```

---

## Deployment and Operations

### 14. Desktop Edition Build and Packaging

```
Produce a reproducible desktop build plan for robertringler/QRATUM using Electron or Tauri. Include: package.json setup, main process code, renderer integration, native module binding, code signing (Windows/Mac), installer generation (NSIS/DMG), auto-update mechanism (Squirrel), and CI job for multi-platform builds. Output: desktop-build-plan.md + desktop/ directory structure.
```

### 15. Regulatory Compliance Checklist Generator

```
For each of 12 market verticals (aerospace, defense, healthcare, finance, energy, telecom, government, transportation, education, media, agriculture, sustainability), produce a compliance checklist mapping QRATUM components to relevant regulations (HIPAA, PCI DSS, GDPR, ITAR, FDA 21 CFR Part 11). Include: regulation name, requirement, QRATUM feature, evidence file, compliance status. Output: compliance/verticals/<vertical>-checklist.md.
```

---

## Code Quality and Integration

### 16. Merge Conflict Resolver

```
For each conflicting file in robertringler/QRATUM (identify via git status on feature branches), produce a three-way merge plan with: base version, branch A changes, branch B changes, conflict markers, suggested resolution diff, and conflict resolution strategy (manual review, accept ours, accept theirs, hybrid). Output: conflicts/<file>-resolution.md + resolved.patch.
```

### 17. Automated Changelog and Release Notes

```
Generate release notes for robertringler/QRATUM from merged PRs and commits since last tag. Group by: Features (✨), Fixes (🐛), Security (🔒), Performance (⚡), Documentation (📚), Refactor (♻️). Include PR numbers, commit hashes, and contributors. Recommend semantic version (MAJOR.MINOR.PATCH). Output: CHANGELOG.md + RELEASE_NOTES.md.
```

### 18. Backlog Prioritization Engine

```
Rank all open issues and TODOs in robertringler/QRATUM by: impact (HIGH/MEDIUM/LOW), effort (story points 1-13), risk (technical debt, security, compliance), and dependencies. Generate a 12-week sprint plan with milestones, assignees, and acceptance criteria. Output: backlog-prioritized.md with Gantt chart (mermaid).
```

---

## End-to-End Validation

### 19. Integration Test Harness

```
Design an end-to-end integration test harness for robertringler/QRATUM that exercises: web dashboards, REST APIs, model server, orchestrator, RAG pipeline, and desktop edition (if available). Use sample data (mock molecules, tire models), validate outputs against expected values, measure latencies, and generate test report. Output: tests/integration/e2e_test_harness.py + test-report.html.
```

### 20. Final Audit Report Generator

```
Produce a final audit report for robertringler/QRATUM certifying: every file analyzed (manifest verification), every claim validated (validation manifest), all compliance controls mapped, all tests passing, all security scans clean, and listing unresolved items with owners and deadlines. Include executive summary, risk matrix, and sign-off checklist. Output: final-audit-report.md (20+ pages).
```

---

## Usage Instructions

1. **Copy prompt text** from code blocks above
2. **Paste into Claude Opus 4.5 or GPT-4** with repository access
3. **Review outputs** for accuracy and completeness
4. **Execute recommended actions** (git commands, code changes, PRs)
5. **Iterate** if needed with follow-up prompts

## Best Practices

- Run prompts in sequence (1 → 20) for full repository analysis
- Use output files as inputs to subsequent prompts
- Validate all generated code with tests before committing
- Review git commands before execution (dry-run first)
- Store outputs in `/deliverables/expert-prompts/outputs/`

## Output Organization

```
deliverables/
  expert-prompts/
    outputs/
      manifest.csv
      validation-manifest.csv
      dependency-audit.json
      pr-triage.md
      refactor-plan.md
      ...
```

---

**Prompt Set Version:** 1.0  
**Compatible With:** Claude Opus 4.5, GPT-4, GitHub Copilot with MCP  
**Repository:** <https://github.com/robertringler/QRATUM>
