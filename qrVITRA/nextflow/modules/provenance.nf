// Provenance module with biokey enforcement

process PROVENANCE {
    tag "Provenance Chain"
    publishDir "${params.outdir}/provenance", mode: 'copy'
    
    input:
    path(vcf)
    val(biokey_required)
    
    output:
    path("provenance.json"), emit: provenance
    path("merkle_chain.txt"), emit: merkle
    
    beforeScript:
    """
    #!/bin/bash
    
    # Check biokey session if required
    if [ "${biokey_required}" = "true" ]; then
        if [ -z "\$VITRA_BIOKEY_PUBLIC_HASH" ]; then
            echo "ERROR: Biokey required but session not active"
            exit 1
        fi
        
        echo "Biokey session verified:"
        echo "  Operator: \$VITRA_BIOKEY_OPERATOR"
        echo "  Public hash: \$VITRA_BIOKEY_PUBLIC_HASH"
        echo "  Safety level: ${params.safety_level}"
    fi
    """
    
    script:
    """
    #!/bin/bash
    set -e
    
    echo "Creating provenance chain"
    echo "VCF: ${vcf}"
    echo "Biokey required: ${biokey_required}"
    
    # Compute VCF hash
    VCF_HASH=\$(sha256sum ${vcf} | awk '{print \$1}')
    
    # Get biokey info if available
    OPERATOR="\${VITRA_BIOKEY_OPERATOR:-none}"
    PUBLIC_HASH="\${VITRA_BIOKEY_PUBLIC_HASH:-none}"
    SAFETY_LEVEL="${params.safety_level}"
    
    # Create provenance JSON
    cat > provenance.json <<EOF
{
  "pipeline": "VITRA-E0",
  "version": "1.0.0",
  "timestamp": "\$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "vcf_hash": "\$VCF_HASH",
  "biokey": {
    "enabled": ${biokey_required},
    "operator": "\$OPERATOR",
    "public_hash": "\$PUBLIC_HASH",
    "safety_level": "\$SAFETY_LEVEL"
  },
  "workflow": {
    "session_id": "${workflow.sessionId}",
    "run_name": "${workflow.runName}",
    "start": "${workflow.start}",
    "profile": "${workflow.profile}"
  },
  "parameters": {
    "fastq_r1": "${params.fastq_r1}",
    "fastq_r2": "${params.fastq_r2}",
    "reference": "${params.ref}",
    "validate_giab": ${params.validate_giab}
  }
}
EOF
    
    # Create Merkle chain entry
    cat > merkle_chain.txt <<EOF
Merkle Chain Entry
==================

Timestamp:   \$(date -u +%Y-%m-%dT%H:%M:%SZ)
VCF Hash:    \$VCF_HASH
Operator:    \$OPERATOR
Public Hash: \$PUBLIC_HASH
Safety:      \$SAFETY_LEVEL

Workflow ID: ${workflow.sessionId}
Profile:     ${workflow.profile}

Signatures:
-----------
EOF
    
    # If biokey enabled, add signature
    if [ "${biokey_required}" = "true" ] && [ -n "\$VITRA_BIOKEY_JSON" ]; then
        echo "Adding biokey signature to Merkle chain..."
        
        # In production, this would call merkler-static to sign
        echo "Biokey signature: placeholder-signature" >> merkle_chain.txt
    fi
    
    echo "Provenance chain created"
    """
}
