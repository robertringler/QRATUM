#!/usr/bin/env nextflow

/*
 * CALL_VARIANTS Module
 * ====================
 * 
 * GPU-accelerated variant calling using NVIDIA Parabricks DeepVariant.
 * Deterministic variant calling with fixed random seed.
 *
 * Input:
 *   - BAM file (sorted)
 *   - BAM index (.bai)
 *   - Reference genome (FASTA with indices)
 *   - Sample ID
 *
 * Output:
 *   - VCF file (gzipped)
 *   - VCF index (.tbi)
 *
 * Determinism:
 *   - Fixed random seed (42)
 *   - Locked DeepVariant model version
 *   - Deterministic GPU kernel selection
 *
 * Performance: ~30 minutes for 30x WGS on A100 GPU
 */

nextflow.enable.dsl = 2

process CALL_VARIANTS {
    tag "${sample_id}"
    label 'gpu'
    publishDir "${params.outdir}/vcf", mode: 'copy'
    
    container 'nvcr.io/nvidia/clara/clara-parabricks:4.2.1-1'
    
    input:
    path bam
    path bai
    path ref
    val sample_id
    
    output:
    path "${sample_id}.vcf.gz", emit: vcf
    path "${sample_id}.vcf.gz.tbi", emit: vcf_idx
    path "${sample_id}_variants.log", emit: log
    
    script:
    def seed = params.pb_deepvariant_seed ?: 42
    """
    # Set deterministic environment
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_CACHE_DISABLE=1
    export TF_DETERMINISTIC_OPS=1
    export TF_CUDNN_DETERMINISTIC=1
    
    # Log GPU info
    nvidia-smi > ${sample_id}_variants.log
    echo "---" >> ${sample_id}_variants.log
    echo "DeepVariant seed: ${seed}" >> ${sample_id}_variants.log
    
    # Run Parabricks DeepVariant
    pbrun deepvariant \\
        --ref ${ref} \\
        --in-bam ${bam} \\
        --out-variants ${sample_id}.vcf.gz \\
        --tmp-dir /tmp/pb_tmp_${sample_id} \\
        --num-gpus 1 \\
        --gpus 0 \\
        --seed ${seed} \\
        2>&1 | tee -a ${sample_id}_variants.log
    
    # Index VCF
    tabix -p vcf ${sample_id}.vcf.gz
    
    # Log completion
    echo "Variant calling complete: \$(date)" >> ${sample_id}_variants.log
    echo "VCF size: \$(stat -c%s ${sample_id}.vcf.gz) bytes" >> ${sample_id}_variants.log
    
    # Count variants
    echo "Total variants: \$(zcat ${sample_id}.vcf.gz | grep -v '^#' | wc -l)" >> ${sample_id}_variants.log
    """
    
    stub:
    """
    touch ${sample_id}.vcf.gz
    touch ${sample_id}.vcf.gz.tbi
    echo "STUB: Variant calling skipped" > ${sample_id}_variants.log
    """
}

workflow test_variants {
    CALL_VARIANTS(
        file("test.bam"),
        file("test.bam.bai"),
        file("GRCh38.fa"),
        "test_sample"
    )
}
