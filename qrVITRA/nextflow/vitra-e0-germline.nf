#!/usr/bin/env nextflow

/*
 * VITRA-E0 Germline WGS Pipeline
 * ===============================
 * 
 * GPU-accelerated whole genome sequencing pipeline with Merkle-chained provenance.
 * Deterministic variant calling with GIAB validation and cryptographic audit trails.
 *
 * Pipeline Stages:
 *   1. ALIGN_FQ2BAM    - FASTQ → BAM alignment (Parabricks fq2bam)
 *   2. CALL_VARIANTS   - BAM → VCF variant calling (DeepVariant)
 *   3. GIAB_VALIDATE   - VCF validation against truth set (vcfeval)
 *   4. PROVENANCE      - Merkle DAG generation (merkler-static)
 *
 * Usage:
 *   nextflow run vitra-e0-germline.nf \
 *     --fastq_r1 HG001_R1.fastq.gz \
 *     --fastq_r2 HG001_R2.fastq.gz \
 *     --ref GRCh38.fa \
 *     --giab_truth HG001_truth.vcf.gz \
 *     --outdir ./results \
 *     -profile guix,gpu
 *
 * Requirements:
 *   - NVIDIA GPU (A100 recommended)
 *   - CUDA 12.4.x
 *   - Parabricks 4.2.1+
 *   - 64GB+ RAM
 *   - 1TB+ storage
 */

nextflow.enable.dsl = 2

// Import modules
include { ALIGN_FQ2BAM } from './modules/align.nf'
include { CALL_VARIANTS } from './modules/call_variants.nf'
include { GIAB_VALIDATE } from './modules/validate.nf'
include { PROVENANCE } from './modules/provenance.nf'

// Pipeline parameters with defaults
params.fastq_r1 = null
params.fastq_r2 = null
params.ref = null
params.giab_truth = null
params.giab_bed = null
params.outdir = './results'
params.sample_id = 'sample'

// Parabricks parameters (locked for determinism)
params.pb_fq2bam_opts = '--low-memory --tmp-dir /tmp'
params.pb_deepvariant_seed = 42

// Zone topology
params.zone = 'Z0'  // Z0, Z1, Z2, or Z3
params.fido2_sig_a = null
params.fido2_sig_b = null

// Validate required parameters
def check_params() {
    if (!params.fastq_r1) {
        error "Missing required parameter: --fastq_r1"
    }
    if (!params.fastq_r2) {
        error "Missing required parameter: --fastq_r2"
    }
    if (!params.ref) {
        error "Missing required parameter: --ref"
    }
    if (!params.giab_truth && params.zone in ['Z2', 'Z3']) {
        error "GIAB truth set required for Z2/Z3 promotion: --giab_truth"
    }
}

workflow {
    check_params()
    
    // Log pipeline start
    log.info """
    =========================================
    VITRA-E0 Germline WGS Pipeline
    =========================================
    Sample ID    : ${params.sample_id}
    FASTQ R1     : ${params.fastq_r1}
    FASTQ R2     : ${params.fastq_r2}
    Reference    : ${params.ref}
    GIAB Truth   : ${params.giab_truth ?: 'N/A'}
    Output Dir   : ${params.outdir}
    Zone         : ${params.zone}
    =========================================
    """.stripIndent()
    
    // Stage 1: Alignment (FASTQ → BAM)
    ALIGN_FQ2BAM(
        params.fastq_r1,
        params.fastq_r2,
        params.ref,
        params.sample_id
    )
    
    // Stage 2: Variant Calling (BAM → VCF)
    CALL_VARIANTS(
        ALIGN_FQ2BAM.out.bam,
        ALIGN_FQ2BAM.out.bai,
        params.ref,
        params.sample_id
    )
    
    // Stage 3: GIAB Validation (if truth set provided)
    if (params.giab_truth) {
        GIAB_VALIDATE(
            CALL_VARIANTS.out.vcf,
            CALL_VARIANTS.out.vcf_idx,
            params.giab_truth,
            params.giab_bed,
            params.ref,
            params.sample_id
        )
        validation_report = GIAB_VALIDATE.out.report
    } else {
        validation_report = Channel.empty()
    }
    
    // Stage 4: Merkle Provenance
    PROVENANCE(
        ALIGN_FQ2BAM.out.bam,
        CALL_VARIANTS.out.vcf,
        validation_report,
        params.zone,
        params.fido2_sig_a,
        params.fido2_sig_b
    )
    
    // Collect all outputs
    PROVENANCE.out.merkle_dag.subscribe { dag ->
        log.info "Merkle DAG created: ${dag}"
    }
    
    log.info "Pipeline complete. Results in: ${params.outdir}"
}

workflow.onComplete {
    log.info """
    =========================================
    Pipeline Execution Summary
    =========================================
    Status       : ${workflow.success ? 'SUCCESS' : 'FAILED'}
    Duration     : ${workflow.duration}
    Exit Status  : ${workflow.exitStatus}
    Error Report : ${workflow.errorReport ?: 'N/A'}
    Merkle DAG   : ${params.outdir}/provenance_dag.cbor
    =========================================
    """.stripIndent()
}

workflow.onError {
    log.error "Pipeline failed: ${workflow.errorReport}"
}
