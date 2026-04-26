#!/usr/bin/env nextflow

/*
 * GIAB_VALIDATE Module
 * ====================
 * 
 * Validate variant calls against GIAB (Genome in a Bottle) truth sets.
 * Computes precision, recall, and F1 scores for SNPs and indels.
 *
 * Input:
 *   - VCF file (query, gzipped)
 *   - VCF index (.tbi)
 *   - GIAB truth VCF (gzipped)
 *   - GIAB high-confidence BED (optional)
 *   - Reference genome
 *   - Sample ID
 *
 * Output:
 *   - Validation report (JSON)
 *   - vcfeval summary
 *
 * Requirements:
 *   - F1 score ≥ 0.995 for Z2 promotion
 *   - Precision ≥ 0.998
 *   - Recall ≥ 0.992
 *
 * Reference:
 *   - HG001 (NA12878) GRCh38 truth set from NIST GIAB
 *
 * Performance: ~5 minutes
 */

nextflow.enable.dsl = 2

process GIAB_VALIDATE {
    tag "${sample_id}"
    label 'cpu'
    publishDir "${params.outdir}/validation", mode: 'copy'
    
    container 'quay.io/biocontainers/rtg-tools:3.12.1--hdfd78af_0'
    
    input:
    path query_vcf
    path query_vcf_idx
    path truth_vcf
    path truth_bed
    path ref
    val sample_id
    
    output:
    path "${sample_id}_validation.json", emit: report
    path "${sample_id}_vcfeval/", emit: vcfeval_dir
    path "${sample_id}_validation.log", emit: log
    
    script:
    def bed_option = truth_bed ? "--bed-regions=${truth_bed}" : ""
    """
    # Log validation start
    echo "Starting GIAB validation: \$(date)" > ${sample_id}_validation.log
    
    # Create SDF reference (required by rtg-tools)
    rtg format -o ref.sdf ${ref}
    
    # Run vcfeval
    rtg vcfeval \\
        -b ${truth_vcf} \\
        -c ${query_vcf} \\
        -o ${sample_id}_vcfeval \\
        -t ref.sdf \\
        ${bed_option} \\
        --all-records \\
        2>&1 | tee -a ${sample_id}_validation.log
    
    # Parse results into JSON
    python3 << 'EOF' > ${sample_id}_validation.json
import json
import os

# Read vcfeval summary
summary_file = "${sample_id}_vcfeval/summary.txt"
metrics = {}

if os.path.exists(summary_file):
    with open(summary_file) as f:
        for line in f:
            if "SNP" in line or "Indel" in line:
                parts = line.strip().split()
                variant_type = parts[0]
                
                # Parse metrics (simplified)
                metrics[variant_type.lower()] = {
                    "true_positives": int(parts[1]) if len(parts) > 1 else 0,
                    "false_positives": int(parts[2]) if len(parts) > 2 else 0,
                    "false_negatives": int(parts[3]) if len(parts) > 3 else 0
                }

# Compute precision, recall, F1
results = {}
for vtype, m in metrics.items():
    tp = m["true_positives"]
    fp = m["false_positives"]
    fn = m["false_negatives"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results[vtype] = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

# Overall metrics (combined SNP + Indel)
total_tp = sum(m["true_positives"] for m in metrics.values())
total_fp = sum(m["false_positives"] for m in metrics.values())
total_fn = sum(m["false_negatives"] for m in metrics.values())

overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

validation_report = {
    "sample_id": "${sample_id}",
    "truth_set": "GIAB_HG001_GRCh38",
    "by_variant_type": results,
    "overall": {
        "precision": round(overall_precision, 6),
        "recall": round(overall_recall, 6),
        "f1_score": round(overall_f1, 6)
    },
    "zone_promotion": {
        "z2_eligible": overall_f1 >= 0.995,
        "threshold": 0.995,
        "message": "PASS" if overall_f1 >= 0.995 else "FAIL - F1 score below threshold"
    }
}

print(json.dumps(validation_report, indent=2))
EOF
    
    # Log completion
    echo "Validation complete: \$(date)" >> ${sample_id}_validation.log
    cat ${sample_id}_validation.json >> ${sample_id}_validation.log
    """
    
    stub:
    """
    echo '{"stub": true}' > ${sample_id}_validation.json
    mkdir -p ${sample_id}_vcfeval
    echo "STUB: Validation skipped" > ${sample_id}_validation.log
    """
}

workflow test_validate {
    GIAB_VALIDATE(
        file("test.vcf.gz"),
        file("test.vcf.gz.tbi"),
        file("HG001_truth.vcf.gz"),
        file("HG001_highconf.bed"),
        file("GRCh38.fa"),
        "test_sample"
    )
}
