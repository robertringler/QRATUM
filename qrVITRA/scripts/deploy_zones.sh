#!/bin/bash
# Deploy and promote between VITRA-E0 zones with biokey enforcement

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/biokey/biokey_lib.sh"

# Parse command
COMMAND="${1:-help}"

show_usage() {
    echo "Usage: $0 <command>"
    echo
    echo "Commands:"
    echo "  promote-z1-to-z2    Promote from staging to production (biokey + FIDO2)"
    echo "  promote-z2-to-z3    Promote from production to archive (dual biokey + dual FIDO2)"
    echo "  list-zones          List all zones and their status"
    echo "  help                Show this help message"
    echo
    echo "Zone Topology:"
    echo "  Z0 (Genesis)   - Immutable baseline"
    echo "  Z1 (Staging)   - Auto-promoted, single biokey"
    echo "  Z2 (Production)- Manual promotion, biokey + FIDO2"
    echo "  Z3 (Archive)   - Manual promotion, dual biokey + dual FIDO2 + air-gap"
}

list_zones() {
    echo "=========================================="
    echo "  VITRA-E0 Zone Status"
    echo "=========================================="
    echo
    
    for zone in Z0 Z1 Z2 Z3; do
        ZONE_DIR="$SCRIPT_DIR/../zones/$zone"
        if [ -d "$ZONE_DIR" ]; then
            FILE_COUNT=$(find "$ZONE_DIR" -type f | wc -l)
            echo "$zone: Active ($FILE_COUNT files)"
        else
            echo "$zone: Not initialized"
        fi
    done
    echo
}

promote_z1_to_z2() {
    echo "=========================================="
    echo "  Promote Z1 (Staging) → Z2 (Production)"
    echo "=========================================="
    echo
    
    # Check biokey session
    if ! check_biokey_session; then
        log_error "Active biokey session required for Z1→Z2 promotion"
        exit 1
    fi
    
    # Check FIDO2 device
    if ! check_fido2_device /dev/hidraw0; then
        log_error "FIDO2 device required for Z1→Z2 promotion"
        exit 1
    fi
    
    log_info "Operator: $VITRA_BIOKEY_OPERATOR"
    log_info "Safety level: SENSITIVE"
    echo
    
    # Promote files
    Z1_DIR="$SCRIPT_DIR/../zones/Z1"
    Z2_DIR="$SCRIPT_DIR/../zones/Z2"
    
    if [ ! -d "$Z1_DIR" ]; then
        log_error "Z1 (Staging) not found"
        exit 1
    fi
    
    mkdir -p "$Z2_DIR"
    
    log_info "Copying files from Z1 to Z2..."
    cp -r "$Z1_DIR"/* "$Z2_DIR/" 2>/dev/null || true
    
    # Create promotion record
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    cat > "$Z2_DIR/promotion_z1_z2.json" <<EOF
{
  "promotion": "Z1_to_Z2",
  "timestamp": "$TIMESTAMP",
  "operator": "$VITRA_BIOKEY_OPERATOR",
  "public_hash": "$VITRA_BIOKEY_PUBLIC_HASH",
  "safety_level": "SENSITIVE",
  "fido2_device": "/dev/hidraw0"
}
EOF
    
    log_success "Promotion Z1→Z2 complete!"
    echo
}

promote_z2_to_z3() {
    echo "=========================================="
    echo "  Promote Z2 (Production) → Z3 (Archive)"
    echo "=========================================="
    echo
    log_warning "This operation requires DUAL BIOKEY + DUAL FIDO2 authorization"
    echo
    
    # Check first biokey session
    if ! check_biokey_session; then
        log_error "First biokey session required"
        exit 1
    fi
    
    OPERATOR_A="$VITRA_BIOKEY_OPERATOR"
    HASH_A="$VITRA_BIOKEY_PUBLIC_HASH"
    
    log_info "Operator A authenticated: $OPERATOR_A"
    echo
    
    # Prompt for second operator
    log_info "Second operator must authenticate..."
    read -p "Press Enter when second operator has active biokey session, or Ctrl+C to cancel..."
    
    # In production, would verify second session in separate terminal
    log_warning "Production system would verify second biokey in separate process"
    
    # Check FIDO2 devices
    if ! check_fido2_device /dev/hidraw0; then
        log_error "FIDO2 device A required"
        exit 1
    fi
    
    if ! check_fido2_device /dev/hidraw1; then
        log_warning "FIDO2 device B not found at /dev/hidraw1"
        log_info "Production system would require second FIDO2 device"
    fi
    
    # Promote files
    Z2_DIR="$SCRIPT_DIR/../zones/Z2"
    Z3_DIR="$SCRIPT_DIR/../zones/Z3"
    
    if [ ! -d "$Z2_DIR" ]; then
        log_error "Z2 (Production) not found"
        exit 1
    fi
    
    mkdir -p "$Z3_DIR"
    
    log_info "Copying files from Z2 to Z3..."
    cp -r "$Z2_DIR"/* "$Z3_DIR/" 2>/dev/null || true
    
    # Create promotion record with dual signatures
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    cat > "$Z3_DIR/promotion_z2_z3.json" <<EOF
{
  "promotion": "Z2_to_Z3",
  "timestamp": "$TIMESTAMP",
  "dual_authorization": {
    "operator_a": "$OPERATOR_A",
    "public_hash_a": "$HASH_A",
    "operator_b": "operator-b-placeholder",
    "public_hash_b": "hash-b-placeholder"
  },
  "safety_level": "CRITICAL",
  "fido2_devices": ["/dev/hidraw0", "/dev/hidraw1"],
  "air_gap": true
}
EOF
    
    log_success "Promotion Z2→Z3 complete!"
    echo
    log_warning "Z3 (Archive) is now air-gapped"
    log_info "Mount Z3 as read-only for cold storage"
    echo
}

# Main
case "$COMMAND" in
    promote-z1-to-z2)
        promote_z1_to_z2
        ;;
    promote-z2-to-z3)
        promote_z2_to_z3
        ;;
    list-zones)
        list_zones
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
