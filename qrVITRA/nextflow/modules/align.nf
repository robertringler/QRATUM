#!/usr/bin/env nextflow

/*
 * ALIGN_FQ2BAM Module
 * ===================
 * 
 * GPU-accelerated FASTQ to BAM alignment using NVIDIA Parabricks fq2bam.
 * Performs BWA-MEM alignment with deterministic sorting and duplicate marking.
 *
 * Input:
 *   - FASTQ R1 (gzipped)
 *   - FASTQ R2 (gzipped)
 *   - Reference genome (FASTA with indices)
 *   - Sample ID
 *
 * Output:
 *   - BAM file (sorted, duplicates marked)
 *   - BAM index (.bai)
 *
 * Performance: ~45 minutes for 30x WGS on A100 GPU
 */

nextflow.enable.dsl = 2

process ALIGN_FQ2BAM {
    tag "${sample_id}"
    label 'gpu'
    publishDir "${params.outdir}/bam", mode: 'copy'
    
    container 'nvcr.io/nvidia/clara/clara-parabricks:4.2.1-1'
    
    input:
    path fastq_r1
    path fastq_r2
    path ref
    val sample_id
    
    output:
    path "${sample_id}.bam", emit: bam
    path "${sample_id}.bam.bai", emit: bai
    path "${sample_id}_align.log", emit: log
    
    script:
    """
    # Set deterministic environment
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_CACHE_DISABLE=1
    
    # Log GPU info
    nvidia-smi > ${sample_id}_align.log
    echo "---" >> ${sample_id}_align.log
    
    # Run Parabricks fq2bam
    pbrun fq2bam \\
        --ref ${ref} \\
        --in-fq ${fastq_r1} ${fastq_r2} \\
        --out-bam ${sample_id}.bam \\
        --tmp-dir /tmp/pb_tmp_${sample_id} \\
        --num-gpus 1 \\
        --gpus 0 \\
        --low-memory \\
        2>&1 | tee -a ${sample_id}_align.log
    
    # Verify BAM integrity
    samtools quickcheck ${sample_id}.bam
    
    # Log completion
    echo "Alignment complete: \$(date)" >> ${sample_id}_align.log
    echo "BAM size: \$(stat -c%s ${sample_id}.bam) bytes" >> ${sample_id}_align.log
    """
    
    stub:
    """
    touch ${sample_id}.bam
    touch ${sample_id}.bam.bai
    echo "STUB: Alignment skipped" > ${sample_id}_align.log
    """
}

workflow test_align {
    ALIGN_FQ2BAM(
        file("test_R1.fastq.gz"),
        file("test_R2.fastq.gz"),
        file("GRCh38.fa"),
        "test_sample"
    )
}
