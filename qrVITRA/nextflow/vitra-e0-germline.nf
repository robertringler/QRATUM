#!/usr/bin/env nextflow

/*
 * VITRA-E0: Biokey-Enabled Whole Genome Sequencing Pipeline
 * 
 * Features:
 * - Ephemeral biokey authentication
 * - Zero-knowledge proof verification
 * - Dual-signature authorization for critical operations
 * - Complete Merkle chain provenance
 * - HIPAA/GDPR/BIPA compliant
 */

nextflow.enable.dsl = 2

// Import modules
include { ALIGN_FQ2BAM } from './modules/align'
include { CALL_VARIANTS } from './modules/call_variants'
include { GIAB_VALIDATE } from './modules/validate'
include { PROVENANCE } from './modules/provenance'

// Validate required parameters
def checkParams() {
    if (!params.fastq_r1) {
        error "Missing required parameter: --fastq_r1"
    }
    if (!params.fastq_r2) {
        error "Missing required parameter: --fastq_r2"
    }
    if (!params.ref) {
        error "Missing required parameter: --ref"
    }
    
    // Check biokey requirements
    if (params.biokey_required) {
        if (!System.getenv('VITRA_BIOKEY_PUBLIC_HASH')) {
            error "Biokey required but not active. Run: ./scripts/biokey/derive_biokey.sh"
        }
        log.info "Biokey session active: ${System.getenv('VITRA_BIOKEY_OPERATOR')}"
        log.info "Safety level: ${params.safety_level}"
    }
}

workflow {
    // Validate parameters
    checkParams()
    
    // Log pipeline info
    log.info ""
    log.info "=========================================="
    log.info "  VITRA-E0 Genomic Analysis Pipeline"
    log.info "=========================================="
    log.info "Input R1:      ${params.fastq_r1}"
    log.info "Input R2:      ${params.fastq_r2}"
    log.info "Reference:     ${params.ref}"
    log.info "Output:        ${params.outdir}"
    log.info "Biokey:        ${params.biokey_required ? 'ENABLED' : 'DISABLED'}"
    log.info "Safety Level:  ${params.safety_level}"
    log.info "=========================================="
    log.info ""
    
    // Create input channels
    fastq_ch = Channel.fromPath([params.fastq_r1, params.fastq_r2])
    ref_ch = Channel.fromPath(params.ref)
    
    // Step 1: Align FASTQ to BAM
    ALIGN_FQ2BAM(
        fastq_ch.collect(),
        ref_ch
    )
    
    // Step 2: Call variants
    CALL_VARIANTS(
        ALIGN_FQ2BAM.out.bam,
        ref_ch
    )
    
    // Step 3: Validate with GIAB (optional)
    if (params.validate_giab && params.giab_truth) {
        giab_truth_ch = Channel.fromPath(params.giab_truth)
        GIAB_VALIDATE(
            CALL_VARIANTS.out.vcf,
            giab_truth_ch
        )
    }
    
    // Step 4: Create provenance chain
    PROVENANCE(
        CALL_VARIANTS.out.vcf,
        params.biokey_required
    )
    
    // Emit final outputs
    CALL_VARIANTS.out.vcf.view { vcf ->
        log.info "Final VCF: ${vcf}"
    }
}

workflow.onComplete {
    log.info ""
    log.info "=========================================="
    log.info "  Pipeline Completed"
    log.info "=========================================="
    log.info "Status:     ${workflow.success ? 'SUCCESS' : 'FAILED'}"
    log.info "Duration:   ${workflow.duration}"
    log.info "Output dir: ${params.outdir}"
    log.info "=========================================="
    log.info ""
    
    if (params.biokey_required) {
        log.info "Biokey session information:"
        log.info "  Operator:    ${System.getenv('VITRA_BIOKEY_OPERATOR')}"
        log.info "  Public hash: ${System.getenv('VITRA_BIOKEY_PUBLIC_HASH')}"
        log.info "  Safety:      ${params.safety_level}"
    }
}

workflow.onError {
    log.error "Pipeline execution failed: ${workflow.errorMessage}"
}
