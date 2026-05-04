#!/bin/bash
# Derive ephemeral biokey from operator VCF file
# Usage: ./derive_biokey.sh <operator-id> <vcf-file>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/biokey_lib.sh"

# Configuration
TMPFS_PATH="/dev/shm/vitra-biokey-$$"
MIN_QUALITY=30
MIN_DEPTH=10
NUM_LOCI=128

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <operator-id> <vcf-file> [num-loci]"
    echo
    echo "Arguments:"
    echo "  operator-id   Unique operator identifier (e.g., 'operator-alice')"
    echo "  vcf-file      Path to operator's VCF file (gzipped or uncompressed)"
    echo "  num-loci      Number of SNP loci to use (default: 128, range: 128-256)"
    echo
    echo "Example:"
    echo "  $0 operator-alice /secure/alice.vcf.gz 192"
    echo
    echo "Security:"
    echo "  - VCF file loaded into RAM-only tmpfs"
    echo "  - Private key never written to disk"
    echo "  - Automatic cleanup on exit"
    echo "  - Session timeout: $SESSION_TIMEOUT_MINUTES minutes"
    exit 1
fi

OPERATOR_ID="$1"
VCF_FILE="$2"
NUM_LOCI="${3:-$NUM_LOCI}"

# Validate inputs
if [ ! -f "$VCF_FILE" ]; then
    log_error "VCF file not found: $VCF_FILE"
    exit 1
fi

if [ "$NUM_LOCI" -lt "$MIN_LOCI" ]; then
    log_error "Number of loci ($NUM_LOCI) below minimum ($MIN_LOCI)"
    exit 1
fi

if [ "$NUM_LOCI" -gt "$MAX_LOCI" ]; then
    log_error "Number of loci ($NUM_LOCI) above maximum ($MAX_LOCI)"
    exit 1
fi

# Banner
echo "=========================================="
echo "  VITRA-E0 Ephemeral Biokey Derivation"
echo "=========================================="
echo
log_info "Operator: $OPERATOR_ID"
log_info "VCF file: $VCF_FILE"
log_info "Target loci: $NUM_LOCI"
echo

# Step 1: Create secure tmpfs
log_info "Step 1: Creating secure tmpfs..."
create_secure_tmpfs "$TMPFS_PATH" 200

# Setup automatic cleanup
setup_auto_cleanup "$TMPFS_PATH"

# Step 2: Copy VCF to tmpfs
log_info "Step 2: Loading VCF into RAM..."
VCF_TMPFS="$TMPFS_PATH/operator.vcf.gz"
cp "$VCF_FILE" "$VCF_TMPFS"
log_success "VCF loaded into secure RAM ($(du -h "$VCF_TMPFS" | cut -f1))"

# Step 3: Extract SNPs
log_info "Step 3: Extracting high-quality SNPs..."
LOCI_JSON="$TMPFS_PATH/loci.json"
extract_snps_from_vcf "$VCF_TMPFS" "$LOCI_JSON" "$MIN_QUALITY" "$MIN_DEPTH" "$NUM_LOCI"

# Step 4: Derive biokey
log_info "Step 4: Deriving ephemeral biokey..."
BIOKEY_OUTPUT="$TMPFS_PATH/biokey_output.txt"

# Check if merkler-static is available
MERKLER_BIN="${MERKLER_BIN:-$SCRIPT_DIR/../../merkler-static/target/release/merkler-static}"
if [ ! -f "$MERKLER_BIN" ]; then
    log_error "merkler-static binary not found: $MERKLER_BIN"
    log_info "Build it with: cd merkler-static && cargo build --release"
    exit 1
fi

"$MERKLER_BIN" derive-biokey "$LOCI_JSON" > "$BIOKEY_OUTPUT"

if [ $? -ne 0 ]; then
    log_error "Biokey derivation failed"
    exit 1
fi

# Extract results
PUBLIC_HASH=$(grep "Public hash:" "$BIOKEY_OUTPUT" | awk '{print $3}')
LOCI_COUNT=$(grep "Loci count:" "$BIOKEY_OUTPUT" | awk '{print $3}')

if [ -z "$PUBLIC_HASH" ] || [ -z "$LOCI_COUNT" ]; then
    log_error "Failed to extract biokey information"
    exit 1
fi

log_success "Biokey derived successfully!"
echo

# Step 5: Export to environment
log_info "Step 5: Exporting biokey to environment..."
export_biokey_to_env "$PUBLIC_HASH" "$LOCI_COUNT" "$OPERATOR_ID"

# Save loci JSON for session (needed for signing)
export VITRA_BIOKEY_JSON="$LOCI_JSON"

# Step 6: Register in operator registry
log_info "Step 6: Registering biokey..."
REGISTRY_FILE="${QRVITRA_CONFIG:-$SCRIPT_DIR/../../configs}/operator_biokeys.json"
mkdir -p "$(dirname "$REGISTRY_FILE")"
register_biokey "$OPERATOR_ID" "Operator" "$PUBLIC_HASH" "$LOCI_COUNT" "ELEVATED" "$REGISTRY_FILE"

echo
echo "=========================================="
echo "  Biokey Session Active"
echo "=========================================="
echo
echo "Operator:      $OPERATOR_ID"
echo "Public Hash:   $PUBLIC_HASH"
echo "Loci Count:    $LOCI_COUNT"
echo "Session Start: $VITRA_BIOKEY_SESSION_START"
echo "Expires:       In $SESSION_TIMEOUT_MINUTES minutes"
echo
echo "Environment variables exported:"
echo "  \$VITRA_BIOKEY_PUBLIC_HASH"
echo "  \$VITRA_BIOKEY_LOCI_COUNT"
echo "  \$VITRA_BIOKEY_OPERATOR"
echo "  \$VITRA_BIOKEY_JSON"
echo
echo "Security:"
echo "  ✓ VCF file in RAM only (tmpfs)"
echo "  ✓ Private key in process memory only"
echo "  ✓ Auto-cleanup on shell exit"
echo "  ✓ Session timeout enforced"
echo
log_warning "Keep this shell session open for biokey operations"
log_warning "Closing this shell will destroy the biokey"
echo

# Keep shell alive with exported environment
exec $SHELL
