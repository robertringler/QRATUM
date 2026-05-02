#!/usr/bin/env bash
set -euo pipefail

# Step V-2: Tier-1 PR Baseline Evidence
PR_NUM=${1:?Usage: $0 <PR_NUMBER>}

EVIDENCE_DIR="docs/convergence_evidence"
mkdir -p "$EVIDENCE_DIR"

echo "=== Capturing evidence for PR #$PR_NUM ==="

# Checkout PR branch (read-only)
gh pr checkout "$PR_NUM"

# Capture branch HEAD
git log --oneline --max-count=5 > "$EVIDENCE_DIR/pr_${PR_NUM}_head.log"
git rev-parse HEAD > "$EVIDENCE_DIR/pr_${PR_NUM}_head_sha.txt"

# Capture PR metadata
gh pr view "$PR_NUM" --json number,title,state,isDraft,mergeable,mergeStateStatus,baseRefName,headRefName,commits,additions,deletions,changedFiles,reviewDecision,labels > "$EVIDENCE_DIR/pr_${PR_NUM}_meta.json"

# Capture CI check status (both formats consistently)
if gh pr checks "$PR_NUM" > "$EVIDENCE_DIR/pr_${PR_NUM}_checks.txt" 2>/dev/null; then
    # If text format succeeded, try JSON format
    if ! gh pr checks "$PR_NUM" --json name,status,conclusion,detailsUrl > "$EVIDENCE_DIR/pr_${PR_NUM}_checks.json" 2>/dev/null; then
        # If JSON failed but text succeeded, create empty JSON array
        echo "[]" > "$EVIDENCE_DIR/pr_${PR_NUM}_checks.json"
    fi
else
    # No CI checks available, write consistent empty states
    echo "No CI checks" > "$EVIDENCE_DIR/pr_${PR_NUM}_checks.txt"
    echo "[]" > "$EVIDENCE_DIR/pr_${PR_NUM}_checks.json"
fi

# Capture diff stat
git diff --stat origin/main...HEAD > "$EVIDENCE_DIR/pr_${PR_NUM}_diffstat.txt"
git diff --name-only origin/main...HEAD > "$EVIDENCE_DIR/pr_${PR_NUM}_changed_files.txt"

# Return to main
git checkout main

echo "Evidence bundle for PR #$PR_NUM complete"
