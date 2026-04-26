#!/usr/bin/env bash
# SquashFS container packing script for VITRA-E0
# Creates relocatable, air-gapped deployment containers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITRA_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-$VITRA_ROOT/dist}"
CONTAINER_NAME="vitra-e0-v1.0.squashfs"
GUIX_PROFILE="${GUIX_PROFILE:-/gnu/store}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v guix &> /dev/null; then
        log_error "Guix not found. Install from https://guix.gnu.org/"
        exit 1
    fi
    
    if ! command -v mksquashfs &> /dev/null; then
        log_error "mksquashfs not found. Install squashfs-tools."
        exit 1
    fi
    
    log_info "Dependencies satisfied."
}

# Build merkler-static with Guix
build_with_guix() {
    log_info "Building merkler-static with Guix..."
    
    cd "$SCRIPT_DIR"
    
    # Build the package
    guix build -f merkler-static.scm
    
    local store_path
    store_path=$(guix build -f merkler-static.scm)
    
    if [[ -z "$store_path" ]]; then
        log_error "Guix build failed - no store path returned"
        exit 1
    fi
    
    log_info "Built successfully: $store_path"
    echo "$store_path"
}

# Pack into SquashFS container
pack_squashfs() {
    local store_path="$1"
    
    log_info "Packing into SquashFS container..."
    
    mkdir -p "$OUTPUT_DIR"
    
    # Use guix pack to create relocatable container
    guix pack -f squashfs \
        -S /bin=bin \
        -S /lib=lib \
        -S /share=share \
        merkler-static \
        > "$OUTPUT_DIR/$CONTAINER_NAME"
    
    if [[ ! -f "$OUTPUT_DIR/$CONTAINER_NAME" ]]; then
        log_error "SquashFS packing failed"
        exit 1
    fi
    
    log_info "SquashFS container created: $OUTPUT_DIR/$CONTAINER_NAME"
    ls -lh "$OUTPUT_DIR/$CONTAINER_NAME"
}

# Verify container integrity
verify_container() {
    local container="$OUTPUT_DIR/$CONTAINER_NAME"
    
    log_info "Verifying container integrity..."
    
    # Check if we can list contents
    if command -v unsquashfs &> /dev/null; then
        unsquashfs -l "$container" | head -20
    fi
    
    # Compute container hash
    local container_hash
    if command -v sha256sum &> /dev/null; then
        container_hash=$(sha256sum "$container" | awk '{print $1}')
    else
        container_hash=$(shasum -a 256 "$container" | awk '{print $1}')
    fi
    
    log_info "Container SHA-256: $container_hash"
    
    # Save hash manifest
    cat > "$OUTPUT_DIR/vitra-e0-v1.0.manifest" <<EOF
VITRA-E0 Container Manifest
============================
Version: 1.0.0
Created: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Container: $CONTAINER_NAME
SHA-256: $container_hash

Components:
- merkler-static: Self-hashing Merkle provenance binary
- CUDA epoch: Anchored to NVIDIA 535.x driver
- Guix profile: Deterministic build closure

Deployment:
1. Transfer container to air-gapped zone
2. Mount with: mount -t squashfs -o loop $CONTAINER_NAME /mnt/vitra
3. Run: /mnt/vitra/bin/merkler-static
4. Verify signatures with FIDO2 keys

Reproducibility:
To rebuild bit-identical container:
  cd qrVITRA/guix
  ./pack.sh

Expected SHA-256: $container_hash
EOF
    
    log_info "Manifest created: $OUTPUT_DIR/vitra-e0-v1.0.manifest"
}

# Create deployment archive
create_deployment_archive() {
    log_info "Creating deployment archive..."
    
    cd "$OUTPUT_DIR"
    
    tar czf vitra-e0-v1.0-deployment.tar.gz \
        "$CONTAINER_NAME" \
        vitra-e0-v1.0.manifest
    
    log_info "Deployment archive: $OUTPUT_DIR/vitra-e0-v1.0-deployment.tar.gz"
    ls -lh vitra-e0-v1.0-deployment.tar.gz
}

# Main execution
main() {
    log_info "VITRA-E0 SquashFS Packing Script"
    log_info "================================="
    
    check_dependencies
    
    # Build with Guix
    store_path=$(build_with_guix)
    
    # Pack into SquashFS
    pack_squashfs "$store_path"
    
    # Verify
    verify_container
    
    # Create deployment archive
    create_deployment_archive
    
    log_info "Packing complete!"
    log_info "Output directory: $OUTPUT_DIR"
    
    echo ""
    log_info "Next steps:"
    echo "  1. Transfer vitra-e0-v1.0-deployment.tar.gz to deployment zone"
    echo "  2. Extract and mount SquashFS container"
    echo "  3. Verify manifest SHA-256"
    echo "  4. Run pipeline with mounted container"
    echo ""
    echo "Air-gapped deployment:"
    echo "  scp vitra-e0-v1.0-deployment.tar.gz airgap-host:/secure/"
    echo "  ssh airgap-host 'cd /secure && tar xzf vitra-e0-v1.0-deployment.tar.gz'"
}

main "$@"
