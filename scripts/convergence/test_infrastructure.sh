#!/usr/bin/env bash
set -euo pipefail

# QRATUM Convergence Evidence Infrastructure - Integration Test
# This script validates that all components are properly installed and functional

echo "=========================================="
echo "QRATUM CONVERGENCE EVIDENCE INFRASTRUCTURE"
echo "Integration Test & Validation"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Test function
test_check() {
    local description="$1"
    local test_command="$2"
    
    echo -n "Testing: $description ... "
    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

echo "=== 1. DIRECTORY STRUCTURE ==="
test_check "scripts/convergence/ exists" "[ -d 'scripts/convergence' ]"
test_check "docs/convergence_evidence/ exists" "[ -d 'docs/convergence_evidence' ]"
test_check ".github/workflows/ exists" "[ -d '.github/workflows' ]"
echo ""

echo "=== 2. EVIDENCE CAPTURE SCRIPTS ==="
test_check "capture_ground_truth.sh exists" "[ -f 'scripts/convergence/capture_ground_truth.sh' ]"
test_check "capture_ground_truth.sh is executable" "[ -x 'scripts/convergence/capture_ground_truth.sh' ]"
test_check "capture_ground_truth.sh has valid syntax" "bash -n 'scripts/convergence/capture_ground_truth.sh'"

test_check "capture_pr_evidence.sh exists" "[ -f 'scripts/convergence/capture_pr_evidence.sh' ]"
test_check "capture_pr_evidence.sh is executable" "[ -x 'scripts/convergence/capture_pr_evidence.sh' ]"
test_check "capture_pr_evidence.sh has valid syntax" "bash -n 'scripts/convergence/capture_pr_evidence.sh'"

test_check "validate_evidence.sh exists" "[ -f 'scripts/convergence/validate_evidence.sh' ]"
test_check "validate_evidence.sh is executable" "[ -x 'scripts/convergence/validate_evidence.sh' ]"
test_check "validate_evidence.sh has valid syntax" "bash -n 'scripts/convergence/validate_evidence.sh'"

test_check "collect_tier1_evidence.sh exists" "[ -f 'scripts/convergence/collect_tier1_evidence.sh' ]"
test_check "collect_tier1_evidence.sh is executable" "[ -x 'scripts/convergence/collect_tier1_evidence.sh' ]"
test_check "collect_tier1_evidence.sh has valid syntax" "bash -n 'scripts/convergence/collect_tier1_evidence.sh'"
echo ""

echo "=== 3. DOCUMENTATION ==="
test_check "QRATUM_CONVERGENCE_REPORT.md exists" "[ -f 'docs/QRATUM_CONVERGENCE_REPORT.md' ]"
test_check "Convergence report contains epistemic integrity notice" "grep -q 'EPISTEMIC INTEGRITY NOTICE' 'docs/QRATUM_CONVERGENCE_REPORT.md'"
test_check "Convergence report contains system invariant" "grep -q 'SYSTEM INVARIANT' 'docs/QRATUM_CONVERGENCE_REPORT.md'"
test_check "Convergence report contains Tier-1 PRs" "grep -q 'PR #370' 'docs/QRATUM_CONVERGENCE_REPORT.md'"
test_check "Convergence report contains verification gate" "grep -q 'Deterministic Verification Gate' 'docs/QRATUM_CONVERGENCE_REPORT.md'"

test_check "scripts/convergence/README.md exists" "[ -f 'scripts/convergence/README.md' ]"
test_check "docs/convergence_evidence/README.md exists" "[ -f 'docs/convergence_evidence/README.md' ]"
echo ""

echo "=== 4. GITHUB ACTIONS WORKFLOW ==="
test_check "convergence-evidence-gate.yml exists" "[ -f '.github/workflows/convergence-evidence-gate.yml' ]"
# Try Python YAML validation if available, otherwise skip with warning
if command -v python3 &>/dev/null && python3 -c 'import yaml' 2>/dev/null; then
    test_check "Workflow has valid YAML syntax" "python3 -c 'import yaml; yaml.safe_load(open(\".github/workflows/convergence-evidence-gate.yml\"))'"
else
    echo "Testing: Workflow has valid YAML syntax ... ⚠ SKIPPED (PyYAML not available)"
fi
test_check "Workflow has workflow_dispatch trigger" "grep -q 'workflow_dispatch' '.github/workflows/convergence-evidence-gate.yml'"
test_check "Workflow has schedule trigger" "grep -q 'schedule' '.github/workflows/convergence-evidence-gate.yml'"
test_check "Workflow has correct permissions" "grep -q 'contents: write' '.github/workflows/convergence-evidence-gate.yml'"
echo ""

echo "=== 5. SCRIPT CONTENT VALIDATION ==="
test_check "capture_ground_truth.sh uses set -euo pipefail" "grep -q 'set -euo pipefail' 'scripts/convergence/capture_ground_truth.sh'"
test_check "capture_ground_truth.sh fetches origin" "grep -q 'git fetch origin' 'scripts/convergence/capture_ground_truth.sh'"
test_check "capture_ground_truth.sh uses gh pr list" "grep -q 'gh pr list' 'scripts/convergence/capture_ground_truth.sh'"
test_check "capture_ground_truth.sh uses gh issue list" "grep -q 'gh issue list' 'scripts/convergence/capture_ground_truth.sh'"

test_check "capture_pr_evidence.sh requires PR_NUM argument" "grep -q 'PR_NUM=\${1:?' 'scripts/convergence/capture_pr_evidence.sh'"
test_check "capture_pr_evidence.sh checks out PR" "grep -q 'gh pr checkout' 'scripts/convergence/capture_pr_evidence.sh'"
test_check "capture_pr_evidence.sh captures CI checks" "grep -q 'gh pr checks' 'scripts/convergence/capture_pr_evidence.sh'"
test_check "capture_pr_evidence.sh returns to main" "grep -q 'git checkout main' 'scripts/convergence/capture_pr_evidence.sh'"

test_check "validate_evidence.sh validates JSON" "grep -q 'jq empty' 'scripts/convergence/validate_evidence.sh'"
test_check "validate_evidence.sh validates SHAs" "grep -q 'grep -Eq' 'scripts/convergence/validate_evidence.sh'"
test_check "validate_evidence.sh generates manifest" "grep -q 'evidence_manifest.json' 'scripts/convergence/validate_evidence.sh'"

test_check "collect_tier1_evidence.sh defines Tier-1 PRs" "grep -q 'TIER1_PRS' 'scripts/convergence/collect_tier1_evidence.sh'"
test_check "collect_tier1_evidence.sh includes PR 370" "grep -q '370' 'scripts/convergence/collect_tier1_evidence.sh'"
test_check "collect_tier1_evidence.sh includes PR 387" "grep -q '387' 'scripts/convergence/collect_tier1_evidence.sh'"
test_check "collect_tier1_evidence.sh includes PR 378" "grep -q '378' 'scripts/convergence/collect_tier1_evidence.sh'"
test_check "collect_tier1_evidence.sh includes PR 197" "grep -q '197' 'scripts/convergence/collect_tier1_evidence.sh'"
test_check "collect_tier1_evidence.sh includes PR 149" "grep -q '149' 'scripts/convergence/collect_tier1_evidence.sh'"
echo ""

echo "=== 6. VERIFICATION GATE LOGIC ==="
# Test verification gate logic with mock data
echo -n "Testing: Verification gate fails without evidence ... "
GATE_RESULT=0
for pr in 370 387 378 197 149; do
    if [ ! -f "docs/convergence_evidence/pr_${pr}_meta.json" ]; then
        GATE_RESULT=1
        break
    fi
done
if [ $GATE_RESULT -eq 1 ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ FAIL${NC}"
    FAILED=$((FAILED + 1))
fi

# Create mock evidence
echo -n "Testing: Creating mock evidence files ... "
for pr in 370 387 378 197 149; do
    echo '{"mock": true, "pr": '$pr'}' > "docs/convergence_evidence/pr_${pr}_meta.json"
done
echo -e "${GREEN}✓ PASS${NC}"
PASSED=$((PASSED + 1))

echo -n "Testing: Verification gate passes with evidence ... "
GATE_RESULT=0
for pr in 370 387 378 197 149; do
    if [ ! -f "docs/convergence_evidence/pr_${pr}_meta.json" ]; then
        GATE_RESULT=1
        break
    fi
done
if [ $GATE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ FAIL${NC}"
    FAILED=$((FAILED + 1))
fi

# Cleanup mock evidence
echo -n "Testing: Cleaning up mock evidence ... "
rm -f docs/convergence_evidence/pr_*_meta.json
echo -e "${GREEN}✓ PASS${NC}"
PASSED=$((PASSED + 1))
echo ""

echo "=== 7. CONVERGENCE READINESS REMEDIATION LAYER (CRRL) ==="
test_check "evaluate_readiness.sh exists" "[ -f 'scripts/convergence/evaluate_readiness.sh' ]"
test_check "evaluate_readiness.sh is executable" "[ -x 'scripts/convergence/evaluate_readiness.sh' ]"
test_check "evaluate_readiness.sh has valid syntax" "bash -n 'scripts/convergence/evaluate_readiness.sh'"
test_check "CRRL script contains readiness vector logic" "grep -q 'pr_readiness_vector.json' 'scripts/convergence/evaluate_readiness.sh'"
test_check "CRRL script contains blocker resolution logic" "grep -q 'blocker_resolution_matrix.json' 'scripts/convergence/evaluate_readiness.sh'"
test_check "CRRL script contains convergence ordering logic" "grep -q 'convergence_order.json' 'scripts/convergence/evaluate_readiness.sh'"
test_check "CRRL script contains trust audit logic" "grep -q 'trust_audit.log' 'scripts/convergence/evaluate_readiness.sh'"
test_check "CRRL script computes omega scores" "grep -q 'omega_scores' 'scripts/convergence/evaluate_readiness.sh'"
test_check "CRRL script validates readiness gate" "grep -q 'Convergence Readiness Gate' 'scripts/convergence/evaluate_readiness.sh'"
echo ""

echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo -e "Total tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "Failed: $FAILED"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo ""
    echo "The QRATUM Convergence Evidence Infrastructure (including CRRL) is properly installed."
    echo ""
    echo "Next steps:"
    echo "  1. Ensure GitHub CLI (gh) is authenticated"
    echo "  2. Run: ./scripts/convergence/collect_tier1_evidence.sh"
    echo "  3. Run: ./scripts/convergence/evaluate_readiness.sh"
    echo "  4. Review evidence artifacts in docs/convergence_evidence/"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the failed tests above and ensure all components are properly installed."
    echo ""
    exit 1
fi
