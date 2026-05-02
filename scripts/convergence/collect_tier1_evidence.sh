#!/bin/bash
# Collect evidence for Tier-1 PR convergence

set -e

EVIDENCE_DIR="docs/convergence_evidence"
mkdir -p "$EVIDENCE_DIR"

echo "📊 Collecting Tier-1 PR Evidence..."

# PR list (excluding merged #378 and blocked #149)
PRS=(370 387 197)

for pr in "${PRS[@]}"; do
    echo "Processing PR #$pr..."
    
    # Get PR metadata
    gh pr view "$pr" --json number,title,state,isDraft,mergeable,reviews,additions,deletions > "$EVIDENCE_DIR/pr_${pr}_metadata.json"
    
    # Get conflict status
    gh pr view "$pr" --json mergeStateStatus,mergeable > "$EVIDENCE_DIR/pr_${pr}_conflicts.json"
    
    # Get CI status
    gh pr checks "$pr" --json name,status,conclusion > "$EVIDENCE_DIR/pr_${pr}_ci.json" 2>/dev/null || echo "[]" > "$EVIDENCE_DIR/pr_${pr}_ci.json"
    
    # Get review status
    gh pr view "$pr" --json reviews --jq '.reviews | length' > "$EVIDENCE_DIR/pr_${pr}_review_count.txt"
done

echo "✅ Evidence collection complete in $EVIDENCE_DIR"
ls -lh "$EVIDENCE_DIR"
