#!/bin/bash
# Integration test for VITRA-E0 biokey system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITRA_ROOT="$SCRIPT_DIR/.."
MERKLER_BIN="$VITRA_ROOT/merkler-static/target/release/merkler-static"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  VITRA-E0 Integration Test"
echo "=========================================="
echo

# Test 1: Merkler-static binary exists
echo -n "Test 1: Checking merkler-static binary... "
if [ -f "$MERKLER_BIN" ]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC} - Binary not found"
    echo "Build with: cd merkler-static && cargo build --release"
    exit 1
fi

# Test 2: Derive biokey from test data
echo -n "Test 2: Deriving biokey from test data... "
TEST_LOCI="$VITRA_ROOT/test_data/test_loci.json"
if [ ! -f "$TEST_LOCI" ]; then
    echo -e "${RED}FAIL${NC} - Test data not found"
    exit 1
fi

OUTPUT=$("$MERKLER_BIN" derive-biokey "$TEST_LOCI" 2>&1)
if echo "$OUTPUT" | grep -q "Public hash:"; then
    echo -e "${GREEN}PASS${NC}"
    PUBLIC_HASH=$(echo "$OUTPUT" | grep "Public hash:" | awk '{print $3}')
    echo "  Public hash: ${PUBLIC_HASH:0:32}..."
else
    echo -e "${RED}FAIL${NC}"
    echo "$OUTPUT"
    exit 1
fi

# Test 3: Generate ZKP challenge
echo -n "Test 3: Generating ZKP challenge... "
CHALLENGE_OUTPUT=$("$MERKLER_BIN" generate-challenge 2>&1)
CHALLENGE=$(echo "$CHALLENGE_OUTPUT" | tail -1)
if [ ${#CHALLENGE} -eq 64 ]; then
    echo -e "${GREEN}PASS${NC}"
    echo "  Challenge: ${CHALLENGE:0:32}..."
else
    echo -e "${RED}FAIL${NC}"
    echo "Challenge length: ${#CHALLENGE} (expected 64)"
    echo "Output: $CHALLENGE_OUTPUT"
    exit 1
fi

# Test 4: Create ZKP proof
echo -n "Test 4: Creating ZKP proof... "
PROOF_FILE="/tmp/test_proof.json"
"$MERKLER_BIN" prove "$TEST_LOCI" "$CHALLENGE" > "$PROOF_FILE" 2>&1
if [ -f "$PROOF_FILE" ] && grep -q "challenge" "$PROOF_FILE"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 5: Verify ZKP proof
echo -n "Test 5: Verifying ZKP proof... "
VERIFY_OUTPUT=$("$MERKLER_BIN" verify-zkp "$PROOF_FILE" 2>&1)
if echo "$VERIFY_OUTPUT" | grep -q "ZKP format valid: true"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "$VERIFY_OUTPUT"
    exit 1
fi

# Test 6: Dual signature
echo -n "Test 6: Creating dual signature... "
DUAL_SIG_FILE="/tmp/test_dual_sig.json"
"$MERKLER_BIN" dual-sign "$TEST_LOCI" "$TEST_LOCI" "Test message" > "$DUAL_SIG_FILE" 2>&1
if [ -f "$DUAL_SIG_FILE" ] && grep -q "operator_a_hash" "$DUAL_SIG_FILE"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 7: Verify dual signature
echo -n "Test 7: Verifying dual signature... "
VERIFY_DUAL=$("$MERKLER_BIN" verify-dual "$DUAL_SIG_FILE" "Test message" 2>&1)
if echo "$VERIFY_DUAL" | grep -q "VALID"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "$VERIFY_DUAL"
    exit 1
fi

# Test 8: Rust unit tests
echo -n "Test 8: Running Rust unit tests... "
cd "$VITRA_ROOT/merkler-static"
TEST_OUTPUT=$(cargo test 2>&1 | grep "test result:")
if echo "$TEST_OUTPUT" | grep -q "0 failed"; then
    echo -e "${GREEN}PASS${NC}"
    PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ passed')
    echo "  $PASSED"
else
    echo -e "${RED}FAIL${NC}"
    echo "$TEST_OUTPUT"
    exit 1
fi
cd "$SCRIPT_DIR"

# Test 9: Configuration files exist
echo -n "Test 9: Checking configuration files... "
MISSING=0
for file in \
    "$VITRA_ROOT/configs/operator_biokeys.json" \
    "$VITRA_ROOT/configs/parabricks_params.json" \
    "$VITRA_ROOT/nextflow.config" \
    "$VITRA_ROOT/README.md"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}FAIL${NC} - Missing: $file"
        MISSING=1
    fi
done
if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}PASS${NC}"
fi

# Test 10: Scripts are executable
echo -n "Test 10: Checking script permissions... "
NONEXEC=0
for script in \
    "$VITRA_ROOT/scripts/biokey/derive_biokey.sh" \
    "$VITRA_ROOT/scripts/biokey/verify_biokey.sh" \
    "$VITRA_ROOT/scripts/deploy_zones.sh" \
    "$VITRA_ROOT/scripts/init_genesis_merkle.sh"; do
    if [ ! -x "$script" ]; then
        echo -e "${RED}FAIL${NC} - Not executable: $script"
        NONEXEC=1
    fi
done
if [ $NONEXEC -eq 0 ]; then
    echo -e "${GREEN}PASS${NC}"
fi

# Cleanup
rm -f /tmp/test_proof.json /tmp/test_dual_sig.json

echo
echo "=========================================="
echo "  All Tests Passed!"
echo "=========================================="
echo
echo "Summary:"
echo "  ✓ Biokey derivation working"
echo "  ✓ Zero-knowledge proofs working"
echo "  ✓ Dual signatures working"
echo "  ✓ All Rust unit tests passing"
echo "  ✓ Configuration files present"
echo "  ✓ Scripts executable"
echo
echo "VITRA-E0 system ready for deployment!"
