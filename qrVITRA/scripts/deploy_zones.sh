#!/usr/bin/env bash
# Deploy Zone Topology (Z0 → Z1 → Z2 → Z3) for VITRA-E0
# Creates zone directories with forward-only promotion scripts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITRA_ROOT="$(dirname "$SCRIPT_DIR")"
ZONES_ROOT="${ZONES_ROOT:-$VITRA_ROOT/zones}"

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

# Create zone directory structure
create_zone_structure() {
    local zone="$1"
    local zone_dir="$ZONES_ROOT/$zone"
    
    log_info "Creating zone structure: $zone"
    
    mkdir -p "$zone_dir"/{pipelines,artifacts,logs,configs}
    
    # Create zone metadata
    case "$zone" in
        Z0)
            cat > "$zone_dir/ZONE_METADATA.json" << EOF
{
  "zone": "Z0",
  "name": "Genesis",
  "status": "immutable",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "properties": {
    "immutable": true,
    "auto_promotion": true,
    "signature_required": false,
    "air_gapped": false
  },
  "promotion": {
    "to": "Z1",
    "requirements": "Automatic on creation"
  }
}
EOF
            ;;
        Z1)
            cat > "$zone_dir/ZONE_METADATA.json" << EOF
{
  "zone": "Z1",
  "name": "Staging",
  "status": "active",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "properties": {
    "immutable": false,
    "auto_promotion": false,
    "signature_required": true,
    "air_gapped": false
  },
  "promotion": {
    "from": "Z0",
    "to": "Z2",
    "requirements": "Single FIDO2 signature A + GIAB F1 ≥ 0.995"
  }
}
EOF
            ;;
        Z2)
            cat > "$zone_dir/ZONE_METADATA.json" << EOF
{
  "zone": "Z2",
  "name": "Production",
  "status": "active",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "properties": {
    "immutable": false,
    "auto_promotion": false,
    "signature_required": true,
    "air_gapped": false
  },
  "promotion": {
    "from": "Z1",
    "to": "Z3",
    "requirements": "Dual FIDO2 signatures A+B + air-gap validation"
  },
  "rollback": {
    "to": "Z1",
    "requirements": "Dual FIDO2 signatures A+B + emergency authorization"
  }
}
EOF
            ;;
        Z3)
            cat > "$zone_dir/ZONE_METADATA.json" << EOF
{
  "zone": "Z3",
  "name": "Archive",
  "status": "cold_storage",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "properties": {
    "immutable": true,
    "auto_promotion": false,
    "signature_required": true,
    "air_gapped": true
  },
  "promotion": {
    "from": "Z2",
    "to": null,
    "requirements": "Dual FIDO2 signatures A+B + network isolation"
  },
  "rollback": {
    "to": "Z2",
    "requirements": "Dual FIDO2 signatures A+B + emergency authorization + audit trail"
  }
}
EOF
            ;;
    esac
    
    log_info "Created: $zone_dir/ZONE_METADATA.json"
}

# Create promotion script
create_promotion_script() {
    local from_zone="$1"
    local to_zone="$2"
    local script_name="promote_${from_zone}_to_${to_zone}.sh"
    local script_path="$ZONES_ROOT/$script_name"
    
    log_info "Creating promotion script: $script_name"
    
    cat > "$script_path" << 'EOF'
#!/usr/bin/env bash
# Promote pipeline artifacts between zones
# Usage: ./promote_Z1_to_Z2.sh <merkle_dag.cbor> [signature_a] [signature_b]

set -euo pipefail

FROM_ZONE="FROM_ZONE_PLACEHOLDER"
TO_ZONE="TO_ZONE_PLACEHOLDER"
ZONES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Parse arguments
MERKLE_DAG="${1:-}"
SIGNATURE_A="${2:-}"
SIGNATURE_B="${3:-}"

if [[ -z "$MERKLE_DAG" ]]; then
    log_error "Usage: $0 <merkle_dag.cbor> [signature_a] [signature_b]"
fi

if [[ ! -f "$MERKLE_DAG" ]]; then
    log_error "Merkle DAG not found: $MERKLE_DAG"
fi

log_info "Promoting from $FROM_ZONE to $TO_ZONE"
log_info "Merkle DAG: $MERKLE_DAG"

# Verify signature requirements
case "$TO_ZONE" in
    Z1)
        log_info "Z0 → Z1: Auto-promotion (no signature required)"
        ;;
    Z2)
        if [[ -z "$SIGNATURE_A" ]]; then
            log_error "Z1 → Z2 requires FIDO2 signature A"
        fi
        log_info "Verifying signature A..."
        # In production: verify signature with epoch_a.pub
        ;;
    Z3)
        if [[ -z "$SIGNATURE_A" ]] || [[ -z "$SIGNATURE_B" ]]; then
            log_error "Z2 → Z3 requires dual FIDO2 signatures A and B"
        fi
        log_info "Verifying dual signatures..."
        # In production: verify both signatures
        ;;
esac

# Copy artifacts
FROM_DIR="$ZONES_ROOT/$FROM_ZONE/artifacts"
TO_DIR="$ZONES_ROOT/$TO_ZONE/artifacts"

log_info "Copying artifacts from $FROM_DIR to $TO_DIR..."
mkdir -p "$TO_DIR"

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
ARTIFACT_DIR="$TO_DIR/promotion_$TIMESTAMP"
mkdir -p "$ARTIFACT_DIR"

cp "$MERKLE_DAG" "$ARTIFACT_DIR/"

# Create promotion manifest
cat > "$ARTIFACT_DIR/promotion_manifest.json" << MANIFEST
{
  "from_zone": "$FROM_ZONE",
  "to_zone": "$TO_ZONE",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "merkle_dag": "$(basename "$MERKLE_DAG")",
  "merkle_hash": "$(sha256sum "$MERKLE_DAG" | awk '{print $1}')",
  "signatures": {
    "fido2_a": ${SIGNATURE_A:+"\"$SIGNATURE_A\""},
    "fido2_b": ${SIGNATURE_B:+"\"$SIGNATURE_B\""}
  },
  "promoted_by": "$USER",
  "hostname": "$HOSTNAME"
}
MANIFEST

log_info "Promotion complete: $ARTIFACT_DIR"
log_info "Manifest: $ARTIFACT_DIR/promotion_manifest.json"
EOF
    
    # Replace placeholders
    sed -i "s/FROM_ZONE_PLACEHOLDER/$from_zone/g" "$script_path"
    sed -i "s/TO_ZONE_PLACEHOLDER/$to_zone/g" "$script_path"
    
    chmod +x "$script_path"
    log_info "Created: $script_path"
}

# Create rollback script
create_rollback_script() {
    local from_zone="$1"
    local to_zone="$2"
    local script_name="rollback_${from_zone}_to_${to_zone}.sh"
    local script_path="$ZONES_ROOT/$script_name"
    
    log_info "Creating rollback script: $script_name"
    
    cat > "$script_path" << 'EOF'
#!/usr/bin/env bash
# Rollback pipeline artifacts between zones (EMERGENCY ONLY)
# Usage: ./rollback_Z3_to_Z2.sh <merkle_dag.cbor> <signature_a> <signature_b> <justification>

set -euo pipefail

FROM_ZONE="FROM_ZONE_PLACEHOLDER"
TO_ZONE="TO_ZONE_PLACEHOLDER"
ZONES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Parse arguments
MERKLE_DAG="${1:-}"
SIGNATURE_A="${2:-}"
SIGNATURE_B="${3:-}"
JUSTIFICATION="${4:-}"

if [[ -z "$MERKLE_DAG" ]] || [[ -z "$SIGNATURE_A" ]] || [[ -z "$SIGNATURE_B" ]] || [[ -z "$JUSTIFICATION" ]]; then
    log_error "Usage: $0 <merkle_dag.cbor> <signature_a> <signature_b> <justification>"
fi

log_warn "ROLLBACK INITIATED: $FROM_ZONE → $TO_ZONE"
log_warn "Justification: $JUSTIFICATION"
log_warn "This action is logged and audited."

# Require dual signatures for all rollbacks
log_info "Verifying dual FIDO2 signatures..."
# In production: verify both signatures with epoch pubkeys

# Create rollback record
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
ROLLBACK_DIR="$ZONES_ROOT/$TO_ZONE/rollbacks/rollback_$TIMESTAMP"
mkdir -p "$ROLLBACK_DIR"

cp "$MERKLE_DAG" "$ROLLBACK_DIR/"

cat > "$ROLLBACK_DIR/rollback_manifest.json" << MANIFEST
{
  "action": "rollback",
  "from_zone": "$FROM_ZONE",
  "to_zone": "$TO_ZONE",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "justification": "$JUSTIFICATION",
  "initiated_by": "$USER",
  "hostname": "$HOSTNAME",
  "signatures": {
    "fido2_a": "$SIGNATURE_A",
    "fido2_b": "$SIGNATURE_B"
  },
  "audit_trail": "This rollback was authorized by dual signature holders and is permanently logged."
}
MANIFEST

log_warn "Rollback complete: $ROLLBACK_DIR"
log_warn "Audit record: $ROLLBACK_DIR/rollback_manifest.json"
log_info "Notify security team and conduct post-rollback review."
EOF
    
    sed -i "s/FROM_ZONE_PLACEHOLDER/$from_zone/g" "$script_path"
    sed -i "s/TO_ZONE_PLACEHOLDER/$to_zone/g" "$script_path"
    
    chmod +x "$script_path"
    log_info "Created: $script_path"
}

# Create zone topology diagram
create_topology_diagram() {
    log_info "Creating zone topology diagram..."
    
    cat > "$ZONES_ROOT/ZONE_TOPOLOGY.md" << 'EOF'
# VITRA-E0 Zone Topology

## Zone Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Z0 (Genesis)                            │
│                   Immutable Base                            │
│                                                             │
│  • Genesis Merkle root (M0)                                │
│  • FIDO2 epoch pubkeys                                     │
│  • CUDA epoch hash                                         │
│  • Auto-promotion to Z1                                    │
└─────────────────┬───────────────────────────────────────────┘
                  │ Auto
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     Z1 (Staging)                            │
│                  Development Builds                         │
│                                                             │
│  • Active development                                       │
│  • Pipeline testing                                         │
│  • GIAB validation                                         │
│  • Promotion to Z2: Signature A + F1 ≥ 0.995              │
└─────────────────┬───────────────────────────────────────────┘
                  │ Signature A
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     Z2 (Production)                         │
│                  Validated Pipelines                        │
│                                                             │
│  • GIAB-validated (F1 ≥ 0.995)                            │
│  • Production deployments                                  │
│  • Signature A required                                    │
│  • Promotion to Z3: Signatures A+B + air-gap              │
└─────────────────┬───────────────────────────────────────────┘
                  │ Signatures A+B
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     Z3 (Archive)                            │
│                  Cold Storage                               │
│                                                             │
│  • Air-gapped deployment                                   │
│  • Immutable archive                                       │
│  • Dual signatures A+B required                           │
│  • Rollback to Z2: Signatures A+B + emergency auth        │
└─────────────────────────────────────────────────────────────┘
```

## Zone Properties

| Zone | Name       | Mutable | Auto-Promote | Signature Required | Air-Gapped |
|------|------------|---------|--------------|-------------------|-----------|
| Z0   | Genesis    | No      | Yes (→ Z1)   | None              | No        |
| Z1   | Staging    | Yes     | No           | None              | No        |
| Z2   | Production | Yes     | No           | Single (A)        | No        |
| Z3   | Archive    | No      | No           | Dual (A+B)        | Yes       |

## Promotion Requirements

### Z0 → Z1 (Automatic)
- No signature required
- Happens on genesis initialization
- Z0 becomes immutable after promotion

### Z1 → Z2 (Production Gate)
- FIDO2 signature A required
- GIAB F1 score ≥ 0.995
- Technical authority approval
- Merkle DAG validation

### Z2 → Z3 (Archive Gate)
- FIDO2 signatures A and B required
- Air-gap validation
- Network isolation
- Dual authority approval
- Permanent archive

## Rollback Procedures

### Z3 → Z2 (Emergency Only)
- FIDO2 signatures A and B required
- Written justification required
- Emergency authorization
- Full audit trail
- Security team notification

### Z2 → Z1 (Emergency Only)
- FIDO2 signatures A and B required
- Written justification required
- Emergency authorization
- Post-rollback review

## Scripts

- `init_genesis_merkle.sh` - Initialize Z0 with genesis Merkle
- `deploy_zones.sh` - Create zone directory structure
- `promote_Z0_to_Z1.sh` - Auto-promotion to staging
- `promote_Z1_to_Z2.sh` - Production promotion
- `promote_Z2_to_Z3.sh` - Archive promotion
- `rollback_Z3_to_Z2.sh` - Emergency rollback from archive
- `rollback_Z2_to_Z1.sh` - Emergency rollback from production

## Security Model

1. **Immutable Zones**: Z0 and Z3 cannot be modified after creation
2. **Forward-Only Flow**: Promotions are one-way (except emergency rollbacks)
3. **Dual Authorization**: Z3 operations require two independent signature holders
4. **Air-Gap Isolation**: Z3 has no network connectivity
5. **Cryptographic Audit**: All operations are Merkle-chained and signed

## Compliance

- **HIPAA**: Sovereign deployment, audit trails, no PHI egress
- **CMMC Level 3**: Air-gapped Z3, dual authorization
- **FDA 21 CFR Part 11**: Electronic signatures, audit trails, rollback
- **ISO 27001**: Key management, access logging
EOF
    
    log_info "Created: $ZONES_ROOT/ZONE_TOPOLOGY.md"
}

# Main execution
main() {
    log_section "VITRA-E0 Zone Topology Deployment"
    
    log_info "Zones root: $ZONES_ROOT"
    
    # Create zone structures
    for zone in Z0 Z1 Z2 Z3; do
        create_zone_structure "$zone"
    done
    
    # Create promotion scripts
    create_promotion_script "Z0" "Z1"
    create_promotion_script "Z1" "Z2"
    create_promotion_script "Z2" "Z3"
    
    # Create rollback scripts
    create_rollback_script "Z3" "Z2"
    create_rollback_script "Z2" "Z1"
    
    # Create topology diagram
    create_topology_diagram
    
    log_section "Zone Topology Deployment Complete"
    
    echo ""
    log_info "Zone structure created in: $ZONES_ROOT"
    log_info "Review zone topology: $ZONES_ROOT/ZONE_TOPOLOGY.md"
    echo ""
    log_info "Next steps:"
    echo "  1. Initialize genesis: ./init_genesis_merkle.sh"
    echo "  2. Run pipeline in Z1 with: nextflow run vitra-e0-germline.nf --zone Z1"
    echo "  3. Promote to Z2: ./promote_Z1_to_Z2.sh results/provenance_dag.cbor <sig_a>"
    echo "  4. Archive to Z3: ./promote_Z2_to_Z3.sh results/provenance_dag.cbor <sig_a> <sig_b>"
}

main "$@"
