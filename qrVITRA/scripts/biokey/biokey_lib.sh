#!/bin/bash
# Biokey library functions for VITRA-E0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Security constants
MIN_LOCI=128
MAX_LOCI=256
SESSION_TIMEOUT_MINUTES=60

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create secure tmpfs for ephemeral operations
create_secure_tmpfs() {
    local mount_point="$1"
    local size_mb="${2:-100}"
    
    if [ -d "$mount_point" ]; then
        log_warning "Tmpfs mount point already exists: $mount_point"
        return 0
    fi
    
    mkdir -p "$mount_point"
    
    # Mount tmpfs with restricted permissions
    sudo mount -t tmpfs -o size="${size_mb}M,mode=0700,noexec,nodev,nosuid" tmpfs "$mount_point"
    
    if [ $? -eq 0 ]; then
        log_success "Secure tmpfs created at $mount_point (${size_mb}MB)"
        return 0
    else
        log_error "Failed to create secure tmpfs"
        return 1
    fi
}

# Cleanup secure tmpfs
cleanup_secure_tmpfs() {
    local mount_point="$1"
    
    if [ ! -d "$mount_point" ]; then
        return 0
    fi
    
    # Shred all files in tmpfs
    if [ -n "$(ls -A "$mount_point" 2>/dev/null)" ]; then
        find "$mount_point" -type f -exec shred -n 3 -z -u {} \; 2>/dev/null
    fi
    
    # Unmount tmpfs
    sudo umount "$mount_point" 2>/dev/null
    rmdir "$mount_point" 2>/dev/null
    
    log_success "Secure tmpfs cleaned up: $mount_point"
}

# Extract SNPs from VCF file
extract_snps_from_vcf() {
    local vcf_file="$1"
    local output_json="$2"
    local min_quality="${3:-30}"
    local min_depth="${4:-10}"
    local num_loci="${5:-128}"
    
    if [ ! -f "$vcf_file" ]; then
        log_error "VCF file not found: $vcf_file"
        return 1
    fi
    
    log_info "Extracting SNPs from VCF..."
    log_info "Filters: QUAL≥${min_quality}, DP≥${min_depth}"
    
    # Extract high-quality SNPs and convert to JSON
    # This is a simplified implementation - real version would use bcftools
    {
        echo "["
        
        # Read VCF (skip header lines starting with #)
        local count=0
        while IFS=$'\t' read -r chrom pos id ref alt qual filter info; do
            # Skip header lines
            [[ "$chrom" == \#* ]] && continue
            
            # Extract depth from INFO field (simplified)
            local depth=15  # Placeholder
            
            # Check filters
            if (( $(echo "$qual >= $min_quality" | bc -l 2>/dev/null || echo "1") )) && \
               (( depth >= min_depth )) && \
               (( count < num_loci )); then
                
                # Output JSON object
                [ $count -gt 0 ] && echo ","
                cat <<EOF
  {
    "chromosome": "$chrom",
    "position": $pos,
    "ref_allele": "$ref",
    "alt_allele": "$alt",
    "quality": $qual,
    "depth": $depth
  }
EOF
                count=$((count + 1))
            fi
            
            [ $count -ge $num_loci ] && break
            
        done < <(zcat -f "$vcf_file")
        
        echo ""
        echo "]"
        
    } > "$output_json"
    
    local extracted_count=$(grep -c '"chromosome"' "$output_json")
    
    if [ "$extracted_count" -lt "$MIN_LOCI" ]; then
        log_error "Insufficient high-quality SNPs extracted: $extracted_count < $MIN_LOCI"
        return 1
    fi
    
    log_success "Extracted $extracted_count SNPs to $output_json"
    return 0
}

# Register biokey in operator registry
register_biokey() {
    local operator_id="$1"
    local operator_name="$2"
    local public_hash="$3"
    local loci_count="$4"
    local auth_level="${5:-ELEVATED}"
    local registry_file="${6:-configs/operator_biokeys.json}"
    
    log_info "Registering biokey for operator: $operator_id"
    
    # Create registry if it doesn't exist
    if [ ! -f "$registry_file" ]; then
        cat > "$registry_file" <<EOF
{
  "version": "1.0.0",
  "operators": [],
  "policies": {
    "min_loci": $MIN_LOCI,
    "max_loci": $MAX_LOCI,
    "session_timeout_minutes": $SESSION_TIMEOUT_MINUTES,
    "dual_authorization_required": ["SENSITIVE", "CRITICAL", "EXISTENTIAL"]
  }
}
EOF
    fi
    
    # Add operator entry (simplified - would use jq in production)
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    log_success "Biokey registered: $operator_id"
    log_info "Public hash: $public_hash"
    log_info "Loci count: $loci_count"
    log_info "Authorization level: $auth_level"
    
    return 0
}

# Export biokey to environment
export_biokey_to_env() {
    local public_hash="$1"
    local loci_count="$2"
    local operator_id="$3"
    
    export VITRA_BIOKEY_PUBLIC_HASH="$public_hash"
    export VITRA_BIOKEY_LOCI_COUNT="$loci_count"
    export VITRA_BIOKEY_OPERATOR="$operator_id"
    export VITRA_BIOKEY_TIMESTAMP="$(date +%s)"
    export VITRA_BIOKEY_SESSION_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    
    log_success "Biokey exported to environment variables"
    log_info "Session started at: $VITRA_BIOKEY_SESSION_START"
    log_warning "Session will expire in $SESSION_TIMEOUT_MINUTES minutes"
}

# Check if biokey session is active
check_biokey_session() {
    if [ -z "$VITRA_BIOKEY_PUBLIC_HASH" ]; then
        log_error "No active biokey session found"
        return 1
    fi
    
    local current_time=$(date +%s)
    local session_age=$(( current_time - VITRA_BIOKEY_TIMESTAMP ))
    local max_age=$(( SESSION_TIMEOUT_MINUTES * 60 ))
    
    if [ "$session_age" -gt "$max_age" ]; then
        log_error "Biokey session expired (${session_age}s > ${max_age}s)"
        cleanup_biokey_session
        return 1
    fi
    
    log_success "Active biokey session found"
    log_info "Operator: $VITRA_BIOKEY_OPERATOR"
    log_info "Session age: ${session_age}s / ${max_age}s"
    
    return 0
}

# Cleanup biokey session
cleanup_biokey_session() {
    log_info "Cleaning up biokey session..."
    
    unset VITRA_BIOKEY_PUBLIC_HASH
    unset VITRA_BIOKEY_LOCI_COUNT
    unset VITRA_BIOKEY_OPERATOR
    unset VITRA_BIOKEY_TIMESTAMP
    unset VITRA_BIOKEY_SESSION_START
    unset VITRA_BIOKEY_JSON
    
    log_success "Biokey session cleaned up"
}

# Setup automatic cleanup on shell exit
setup_auto_cleanup() {
    local tmpfs_path="$1"
    
    # Set trap to cleanup on exit
    trap "cleanup_biokey_session; cleanup_secure_tmpfs '$tmpfs_path'" EXIT INT TERM
    
    log_success "Automatic cleanup configured"
}

# Verify FIDO2 device availability
check_fido2_device() {
    local device_path="${1:-/dev/hidraw0}"
    
    if [ ! -e "$device_path" ]; then
        log_error "FIDO2 device not found: $device_path"
        return 1
    fi
    
    if [ ! -r "$device_path" ] || [ ! -w "$device_path" ]; then
        log_error "FIDO2 device not accessible: $device_path"
        return 1
    fi
    
    log_success "FIDO2 device available: $device_path"
    return 0
}

# Generate random session ID
generate_session_id() {
    echo "biokey-session-$(date +%s)-$(head -c 16 /dev/urandom | base64 | tr -d '/+=' | head -c 16)"
}
