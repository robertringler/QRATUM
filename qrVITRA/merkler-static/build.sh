#!/usr/bin/env bash
# Deterministic build script for merkler-static binary
# Injects hashes post-build for self-verification and CUDA anchoring

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check for required tools
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v rustc &> /dev/null; then
        log_error "rustc not found. Install Rust toolchain."
        exit 1
    fi
    
    if ! command -v cargo &> /dev/null; then
        log_error "cargo not found. Install Rust toolchain."
        exit 1
    fi
    
    if ! command -v sha256sum &> /dev/null && ! command -v shasum &> /dev/null; then
        log_error "sha256sum or shasum not found."
        exit 1
    fi
    
    log_info "All dependencies satisfied."
}

# Build static binary with musl
build_static_binary() {
    log_info "Building static musl binary..."
    
    # Install musl target if not present
    rustup target add x86_64-unknown-linux-musl 2>/dev/null || true
    
    # Build with release optimizations
    cargo build --release --target x86_64-unknown-linux-musl
    
    local binary_path="target/x86_64-unknown-linux-musl/release/merkler-static"
    
    if [[ ! -f "$binary_path" ]]; then
        log_error "Build failed - binary not found at $binary_path"
        exit 1
    fi
    
    log_info "Static binary built successfully: $binary_path"
    echo "$binary_path"
}

# Compute SHA3-256 hash (using sha256sum as placeholder)
compute_hash() {
    local file="$1"
    
    if command -v sha256sum &> /dev/null; then
        sha256sum "$file" | awk '{print $1}'
    else
        shasum -a 256 "$file" | awk '{print $1}'
    fi
}

# Extract placeholder hashes from injected/ directory
extract_injected_hashes() {
    log_info "Extracting injected hashes from placeholder files..."
    
    local injected_dir="$SCRIPT_DIR/injected"
    mkdir -p "$injected_dir"
    
    # Create placeholder files if they don't exist
    for file in cuda_ptx_hash.bin driver_manifest.bin epoch_pubkey_a.bin epoch_pubkey_b.bin; do
        if [[ ! -f "$injected_dir/$file" ]]; then
            log_warn "Creating placeholder: $injected_dir/$file"
            dd if=/dev/zero of="$injected_dir/$file" bs=32 count=1 2>/dev/null
        fi
    done
    
    log_info "Placeholder hashes ready."
}

# Inject hashes into binary (simplified - just for demonstration)
inject_self_hash() {
    local binary="$1"
    local temp_binary="${binary}.tmp"
    
    log_info "Computing self-hash..."
    local self_hash
    self_hash=$(compute_hash "$binary")
    
    log_info "Self-hash: $self_hash"
    
    # In production, this would patch the binary at MERKLER_SELF_HASH location
    # For now, just create a marker file
    echo "$self_hash" > "$SCRIPT_DIR/injected/merkler_self.bin"
    
    log_info "Self-hash injection complete (marker file created)."
}

# Main build pipeline
main() {
    log_info "Starting deterministic build for merkler-static..."
    
    check_dependencies
    extract_injected_hashes
    
    # Clean previous builds
    log_info "Cleaning previous builds..."
    cargo clean
    
    # Build binary
    binary_path=$(build_static_binary)
    
    # Inject self-hash
    inject_self_hash "$binary_path"
    
    # Verification
    log_info "Verifying binary..."
    file "$binary_path"
    ls -lh "$binary_path"
    
    log_info "Build complete!"
    log_info "Binary: $binary_path"
    log_info "Injected hashes: $SCRIPT_DIR/injected/"
    
    echo ""
    log_info "Next steps:"
    echo "  1. Extract CUDA PTX hashes from Parabricks container"
    echo "  2. Update injected/cuda_ptx_hash.bin"
    echo "  3. Update injected/driver_manifest.bin with NVIDIA driver info"
    echo "  4. Generate FIDO2 keys and update epoch_pubkey_*.bin"
    echo "  5. Rebuild and verify reproducibility"
}

main "$@"
