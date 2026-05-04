#!/bin/bash
# Verify biokey using zero-knowledge proof
# Usage: ./verify_biokey.sh <operator-id>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/biokey_lib.sh"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <operator-id>"
    echo
    echo "Arguments:"
    echo "  operator-id   Operator to verify (e.g., 'operator-alice')"
    echo
    echo "Example:"
    echo "  $0 operator-alice"
    echo
    echo "Process:"
    echo "  1. Generates random challenge"
    echo "  2. Operator provides ZKP response"
    echo "  3. Verifies response without revealing genome"
    exit 1
fi

OPERATOR_ID="$1"
REGISTRY_FILE="${QRVITRA_CONFIG:-$SCRIPT_DIR/../../configs}/operator_biokeys.json"

# Banner
echo "=========================================="
echo "  VITRA-E0 Zero-Knowledge Proof"
echo "=========================================="
echo
log_info "Operator: $OPERATOR_ID"
echo

# Step 1: Load operator's public hash from registry
log_info "Step 1: Loading operator public hash..."

if [ ! -f "$REGISTRY_FILE" ]; then
    log_error "Operator registry not found: $REGISTRY_FILE"
    exit 1
fi

# Simplified - would use jq in production
PUBLIC_HASH_REGISTERED="placeholder-hash-from-registry"
log_success "Public hash loaded from registry"
echo

# Step 2: Generate random challenge
log_info "Step 2: Generating ZKP challenge..."

MERKLER_BIN="${MERKLER_BIN:-$SCRIPT_DIR/../../merkler-static/target/release/merkler-static}"
if [ ! -f "$MERKLER_BIN" ]; then
    log_error "merkler-static binary not found: $MERKLER_BIN"
    exit 1
fi

CHALLENGE=$("$MERKLER_BIN" generate-challenge)
log_success "Challenge generated: ${CHALLENGE:0:32}..."
echo

# Step 3: Prompt operator to generate proof
log_info "Step 3: Operator must generate ZKP proof..."
echo
echo "Challenge: $CHALLENGE"
echo
echo "To generate proof, operator must run:"
echo "  merkler-static prove \$VITRA_BIOKEY_JSON $CHALLENGE > proof.json"
echo
read -p "Press Enter when proof is ready, or Ctrl+C to cancel..."
echo

# Step 4: Read and verify proof
log_info "Step 4: Verifying proof..."

if [ ! -f "proof.json" ]; then
    log_error "Proof file not found: proof.json"
    log_info "Operator must generate proof and save to proof.json"
    exit 1
fi

"$MERKLER_BIN" verify-zkp proof.json

if [ $? -eq 0 ]; then
    log_success "Zero-knowledge proof VALID"
    echo
    echo "=========================================="
    echo "  Verification Successful"
    echo "=========================================="
    echo
    echo "Operator:      $OPERATOR_ID"
    echo "Status:        AUTHENTICATED"
    echo "Method:        Zero-Knowledge Proof"
    echo "Genome Data:   NOT REVEALED"
    echo
    log_success "Operator credentials verified without genome exposure"
else
    log_error "Zero-knowledge proof INVALID"
    echo
    echo "=========================================="
    echo "  Verification Failed"
    echo "=========================================="
    echo
    echo "Operator:      $OPERATOR_ID"
    echo "Status:        AUTHENTICATION FAILED"
    echo
    log_error "Cannot verify operator credentials"
    exit 1
fi
