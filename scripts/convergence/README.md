# QRATUM Convergence Evidence Scripts

## Overview

This directory contains scripts for capturing, validating, and managing evidence artifacts required for QRATUM convergence verification. These scripts implement the epistemic verification gate that ensures merge readiness is based on measured ground truth rather than model inference.

## Scripts

### `capture_ground_truth.sh`

**Purpose:** Step V-1 — Repository Ground Truth Capture

Captures the current state of the QRATUM repository including:
- Current HEAD SHA of origin/main
- Full list of open PRs with metadata
- Open issues for cross-reference
- Execution timestamp

**Usage:**
```bash
./scripts/convergence/capture_ground_truth.sh
```

**Output:**
- `docs/convergence_evidence/main_head_sha.txt`
- `docs/convergence_evidence/open_pr_snapshot.json`
- `docs/convergence_evidence/open_pr_snapshot.txt`
- `docs/convergence_evidence/open_issue_snapshot.json`
- `docs/convergence_evidence/verification_timestamp.txt`

### `capture_pr_evidence.sh`

**Purpose:** Step V-2 — Per-PR Evidence Collection

Captures detailed evidence for a specific PR including:
- Branch HEAD SHA and recent commits
- PR metadata (mergeable status, CI checks, review decision)
- CI check status and conclusions
- Diff statistics and changed files

**Usage:**
```bash
./scripts/convergence/capture_pr_evidence.sh <PR_NUMBER>
```

**Example:**
```bash
./scripts/convergence/capture_pr_evidence.sh 370
```

**Output:**
- `docs/convergence_evidence/pr_<NUM>_head.log`
- `docs/convergence_evidence/pr_<NUM>_head_sha.txt`
- `docs/convergence_evidence/pr_<NUM>_meta.json`
- `docs/convergence_evidence/pr_<NUM>_checks.txt`
- `docs/convergence_evidence/pr_<NUM>_checks.json`
- `docs/convergence_evidence/pr_<NUM>_diffstat.txt`
- `docs/convergence_evidence/pr_<NUM>_changed_files.txt`

### `validate_evidence.sh`

**Purpose:** Step V-3 — Evidence Artifact Validation

Validates the integrity of collected evidence artifacts:
- Verifies JSON structure validity
- Validates SHA format
- Generates evidence manifest

**Usage:**
```bash
./scripts/convergence/validate_evidence.sh
```

**Output:**
- `docs/convergence_evidence/evidence_manifest.json`
- Validation messages to stdout

### `collect_tier1_evidence.sh`

**Purpose:** Orchestration script for Tier-1 PRs

Executes the complete evidence collection workflow for all Tier-1 PRs (370, 387, 378, 197, 149):
1. Captures repository ground truth
2. Collects evidence for each Tier-1 PR
3. Validates all evidence artifacts

**Usage:**
```bash
./scripts/convergence/collect_tier1_evidence.sh
```

**Output:**
All evidence artifacts for repository and Tier-1 PRs in `docs/convergence_evidence/`

## Requirements

- GitHub CLI (`gh`) must be installed and authenticated
- `jq` must be installed for JSON validation
- Git repository must be clean (no uncommitted changes)
- Network access to fetch from GitHub

## Verification Gate

After collecting evidence, verify the gate passes:

```bash
for pr in 370 387 378 197 149; do
    if [ ! -f "docs/convergence_evidence/pr_${pr}_meta.json" ]; then
        echo "GATE FAILURE: Missing evidence for PR #$pr"
        exit 1
    fi
done
echo "GATE PASSED: All evidence artifacts present"
```

## System Invariant

**QRATUM does not execute merges based on inference.**

**The state of the system must be measured before it is transformed.**

All convergence actions are conditional on evidence artifact existence.
