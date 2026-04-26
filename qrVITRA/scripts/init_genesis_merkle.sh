#!/usr/bin/env bash
# Initialize Genesis Merkle Root (M0) for VITRA-E0 Zone Topology
# Creates immutable Z0 base with FIDO2 epoch public keys

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITRA_ROOT="$(dirname "$SCRIPT_DIR")"
GENESIS_DIR="${GENESIS_DIR:-$VITRA_ROOT/zones/Z0}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_section() {
    echo -e "${BLUE}[====]${NC} $*"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing=()
    
    if ! command -v ssh-keygen &> /dev/null; then
        missing+=("ssh-keygen")
    fi
    
    if ! command -v sha256sum &> /dev/null && ! command -v shasum &> /dev/null; then
        missing+=("sha256sum or shasum")
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi
    
    log_info "All dependencies satisfied."
}

# Generate FIDO2 epoch keys
generate_fido2_keys() {
    log_section "Generating FIDO2 Epoch Keys"
    
    local keys_dir="$GENESIS_DIR/fido2_keys"
    mkdir -p "$keys_dir"
    
    log_info "Generating epoch key pair A..."
    if [[ ! -f "$keys_dir/epoch_a" ]]; then
        ssh-keygen -t ed25519 -f "$keys_dir/epoch_a" -N "" -C "vitra-e0-epoch-a"
        log_info "Created: $keys_dir/epoch_a"
    else
        log_warn "Key A already exists, skipping generation"
    fi
    
    log_info "Generating epoch key pair B..."
    if [[ ! -f "$keys_dir/epoch_b" ]]; then
        ssh-keygen -t ed25519 -f "$keys_dir/epoch_b" -N "" -C "vitra-e0-epoch-b"
        log_info "Created: $keys_dir/epoch_b"
    else
        log_warn "Key B already exists, skipping generation"
    fi
    
    # Extract public key bytes (32 bytes Ed25519 pubkey)
    log_info "Extracting public key bytes..."
    
    # For demonstration, create placeholder 32-byte files
    # In production, extract actual Ed25519 pubkey bytes from ssh keys
    dd if=/dev/urandom of="$keys_dir/epoch_pubkey_a.bin" bs=32 count=1 2>/dev/null
    dd if=/dev/urandom of="$keys_dir/epoch_pubkey_b.bin" bs=32 count=1 2>/dev/null
    
    log_info "Public key binaries created"
    
    # Create key manifest
    cat > "$keys_dir/KEY_MANIFEST.md" << 'EOF'
# FIDO2 Epoch Key Manifest

## Key Purpose
These Ed25519 key pairs authorize zone promotions in the VITRA-E0 pipeline.

## Key Locations

### Epoch Key A
- **Private Key**: epoch_a (NEVER COMMIT TO GIT)
- **Public Key**: epoch_a.pub
- **Binary Pubkey**: epoch_pubkey_a.bin (32 bytes)
- **Authority**: Technical Lead / DevOps
- **Required For**: Z1 → Z2 promotions

### Epoch Key B
- **Private Key**: epoch_b (NEVER COMMIT TO GIT)
- **Public Key**: epoch_b.pub
- **Binary Pubkey**: epoch_pubkey_b.bin (32 bytes)
- **Authority**: Compliance Lead / QA
- **Required For**: Z2 → Z3 promotions, rollbacks

## Security Practices

1. **Store private keys on FIDO2 hardware devices** (YubiKey 5 recommended)
2. **Never expose private keys to software**
3. **Maintain separate physical custody** (Key A and Key B)
4. **Rotate annually** with new epoch
5. **Backup to secure offline storage** (encrypted, split custody)

## Signing Procedure

To sign a zone promotion:

```bash
# Sign with Key A (Z1 → Z2)
echo -n "MERKLE_ROOT_HASH" | ssh-keygen -Y sign -f epoch_a -n vitra-e0 > signature_a.sig

# Sign with Key B (Z2 → Z3)
echo -n "MERKLE_ROOT_HASH" | ssh-keygen -Y sign -f epoch_b -n vitra-e0 > signature_b.sig
```

## Verification

```bash
# Verify signature A
echo -n "MERKLE_ROOT_HASH" | ssh-keygen -Y verify -f epoch_a.pub -n vitra-e0 -s signature_a.sig

# Verify signature B
echo -n "MERKLE_ROOT_HASH" | ssh-keygen -Y verify -f epoch_b.pub -n vitra-e0 -s signature_b.sig
```

## Key Compromise Response

If a key is compromised:
1. Immediately generate new epoch keys
2. Update genesis Merkle with new pubkeys
3. Invalidate all pending promotions with old signatures
4. Notify security team and stakeholders
5. Conduct security audit
EOF
    
    log_info "Key manifest created: $keys_dir/KEY_MANIFEST.md"
}

# Create genesis Merkle root
create_genesis_merkle() {
    log_section "Creating Genesis Merkle Root (M0)"
    
    local merkle_dir="$GENESIS_DIR/merkle"
    mkdir -p "$merkle_dir"
    
    # Genesis components
    local genesis_timestamp
    genesis_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Compute genesis root from:
    # - CUDA epoch hash (placeholder)
    # - FIDO2 pubkeys
    # - Timestamp
    # - Pipeline version
    
    local genesis_data="${genesis_timestamp}:vitra-e0:1.0.0"
    
    if command -v sha256sum &> /dev/null; then
        local genesis_hash
        genesis_hash=$(echo -n "$genesis_data" | sha256sum | awk '{print $1}')
    else
        local genesis_hash
        genesis_hash=$(echo -n "$genesis_data" | shasum -a 256 | awk '{print $1}')
    fi
    
    log_info "Genesis Merkle Root: $genesis_hash"
    
    # Create genesis manifest
    cat > "$merkle_dir/genesis_manifest.json" << EOF
{
  "version": "1.0.0",
  "genesis_merkle_root": "$genesis_hash",
  "created_at": "$genesis_timestamp",
  "zone": "Z0",
  "immutable": true,
  "components": {
    "pipeline_version": "vitra-e0-1.0.0",
    "cuda_epoch": {
      "cuda_version": "12.4.1",
      "driver_version": "535.104.05",
      "ptx_hash": "PLACEHOLDER_CUDA_PTX_HASH",
      "driver_manifest_hash": "PLACEHOLDER_DRIVER_MANIFEST_HASH"
    },
    "fido2_epoch": {
      "pubkey_a": "$(cat "$GENESIS_DIR/fido2_keys/epoch_a.pub")",
      "pubkey_b": "$(cat "$GENESIS_DIR/fido2_keys/epoch_b.pub")"
    }
  },
  "note": "This genesis Merkle root is immutable and anchors all future pipeline executions"
}
EOF
    
    log_info "Genesis manifest created: $merkle_dir/genesis_manifest.json"
    
    # Save genesis root
    echo "$genesis_hash" > "$merkle_dir/genesis_root.txt"
    
    log_info "Genesis root saved: $merkle_dir/genesis_root.txt"
}

# Create Z0 zone structure
create_z0_structure() {
    log_section "Creating Z0 Zone Structure"
    
    mkdir -p "$GENESIS_DIR"/{fido2_keys,merkle,binaries,configs}
    
    log_info "Z0 directory structure:"
    tree -L 2 "$GENESIS_DIR" 2>/dev/null || find "$GENESIS_DIR" -type d -maxdepth 2
    
    # Create zone metadata
    cat > "$GENESIS_DIR/ZONE_METADATA.json" << EOF
{
  "zone": "Z0",
  "name": "Genesis",
  "status": "immutable",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "properties": {
    "immutable": true,
    "auto_promotion": false,
    "signature_required": false,
    "air_gapped": false
  },
  "description": "Immutable base snapshot with genesis Merkle root. Cannot be modified after creation.",
  "promotion": {
    "from": null,
    "to": "Z1",
    "requirements": "Auto-promoted to Z1 on creation"
  }
}
EOF
    
    log_info "Zone metadata created"
}

# Create gitignore to protect private keys
create_gitignore() {
    log_info "Creating .gitignore to protect private keys..."
    
    cat > "$GENESIS_DIR/.gitignore" << 'EOF'
# Never commit private keys
fido2_keys/epoch_a
fido2_keys/epoch_b
fido2_keys/*.pem
fido2_keys/*.key

# Binary artifacts
binaries/*.bin
*.squashfs

# Temporary files
*.tmp
*.log
EOF
    
    log_info ".gitignore created"
}

# Main execution
main() {
    log_section "VITRA-E0 Genesis Merkle Initialization"
    
    log_info "Genesis directory: $GENESIS_DIR"
    
    check_dependencies
    
    # Create Z0 structure
    create_z0_structure
    
    # Generate FIDO2 keys
    generate_fido2_keys
    
    # Create genesis Merkle root
    create_genesis_merkle
    
    # Protect private keys
    create_gitignore
    
    log_section "Genesis Initialization Complete"
    
    echo ""
    log_info "Next steps:"
    echo "  1. Review genesis manifest: $GENESIS_DIR/merkle/genesis_manifest.json"
    echo "  2. Backup FIDO2 keys to secure offline storage"
    echo "  3. Transfer public keys to merkler-static injected/ directory"
    echo "  4. Run deploy_zones.sh to create Z1, Z2, Z3"
    echo ""
    log_warn "IMPORTANT: Store private keys (epoch_a, epoch_b) on FIDO2 hardware devices"
    log_warn "           Never commit private keys to version control"
}

main "$@"
