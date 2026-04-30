#!/usr/bin/env bash
set -euo pipefail

# Step V-1: Repository Ground Truth Capture
echo "=== QRATUM CONVERGENCE — GROUND TRUTH CAPTURE ==="

EVIDENCE_DIR="docs/convergence_evidence"
mkdir -p "$EVIDENCE_DIR"

# Capture current main branch HEAD
git fetch origin
git rev-parse origin/main > "$EVIDENCE_DIR/main_head_sha.txt"

# Capture full PR list
gh pr list --state open --limit 50 --json number,title,state,isDraft,baseRefName,headRefName,mergeable,reviewDecision > "$EVIDENCE_DIR/open_pr_snapshot.json"
gh pr list --state open --limit 50 > "$EVIDENCE_DIR/open_pr_snapshot.txt"

# Capture issue list for cross-reference
gh issue list --state open --limit 100 --json number,title,state,labels > "$EVIDENCE_DIR/open_issue_snapshot.json"

# Log execution timestamp
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EVIDENCE_DIR/verification_timestamp.txt"

echo "Step V-1 complete. Evidence artifacts in $EVIDENCE_DIR/"
