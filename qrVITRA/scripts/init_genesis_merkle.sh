#!/bin/bash
# Initialize genesis Merkle chain for VITRA-E0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENESIS_DIR="${1:-$SCRIPT_DIR/../zones/Z0}"

echo "=========================================="
echo "  VITRA-E0 Genesis Merkle Initialization"
echo "=========================================="
echo

# Create genesis zone directory
mkdir -p "$GENESIS_DIR"

# Create genesis block
GENESIS_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GENESIS_HASH=$(echo -n "VITRA-E0-GENESIS-$GENESIS_TIMESTAMP" | sha256sum | awk '{print $1}')

cat > "$GENESIS_DIR/genesis.json" <<EOF
{
  "zone": "Z0",
  "type": "genesis",
  "timestamp": "$GENESIS_TIMESTAMP",
  "genesis_hash": "$GENESIS_HASH",
  "properties": {
    "immutable": true,
    "biokey_required": false,
    "safety_level": "ROUTINE"
  },
  "metadata": {
    "pipeline": "VITRA-E0",
    "version": "1.0.0",
    "description": "Genesis block for sovereign genomic analysis"
  }
}
EOF

echo "Genesis block created:"
echo "  Zone: Z0"
echo "  Timestamp: $GENESIS_TIMESTAMP"
echo "  Hash: $GENESIS_HASH"
echo "  Path: $GENESIS_DIR/genesis.json"
echo

# Create zone directories
for zone in Z1 Z2 Z3; do
    ZONE_DIR="$SCRIPT_DIR/../zones/$zone"
    mkdir -p "$ZONE_DIR"
    echo "Created zone: $zone ($ZONE_DIR)"
done

echo
echo "Genesis Merkle chain initialized successfully!"
echo
echo "Zone Topology:"
echo "  Z0 (Genesis)   - Immutable, no biokey"
echo "  Z1 (Staging)   - Auto-promoted, single biokey"
echo "  Z2 (Production)- Manual, biokey + FIDO2"
echo "  Z3 (Archive)   - Manual, dual biokey + dual FIDO2 + air-gap"
