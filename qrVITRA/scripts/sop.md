# VITRA-E0 Standard Operating Procedures (SOP)

**Document Version**: 1.0.0  
**Last Updated**: 2024-12-24  
**Classification**: Internal Use Only  
**Review Frequency**: Quarterly

---

## Table of Contents

1. [Introduction](#introduction)
2. [Scope and Purpose](#scope-and-purpose)
3. [Roles and Responsibilities](#roles-and-responsibilities)
4. [FIDO2 Key Management](#fido2-key-management)
5. [Zone Promotion Procedures](#zone-promotion-procedures)
6. [Rollback Procedures](#rollback-procedures)
7. [Air-Gap Deployment](#air-gap-deployment)
8. [Pipeline Execution](#pipeline-execution)
9. [Reproducibility Validation](#reproducibility-validation)
10. [Incident Response](#incident-response)
11. [Audit and Compliance](#audit-and-compliance)
12. [Appendices](#appendices)

---

## 1. Introduction

This document defines standard operating procedures for the VITRA-E0 sovereign entropy anchor for deterministic genomics. It covers:

- FIDO2 key management and custody
- Zone topology operations (Z0 → Z1 → Z2 → Z3)
- Pipeline execution and validation
- Emergency rollback procedures
- Compliance and audit requirements

### 1.1 Document Conventions

- 🔴 **CRITICAL**: Mandatory security controls
- 🟡 **IMPORTANT**: Strongly recommended practices
- 🟢 **OPTIONAL**: Best practices for enhanced security

---

## 2. Scope and Purpose

### 2.1 Scope

This SOP applies to:

- All VITRA-E0 pipeline executions
- Zone promotion and rollback operations
- FIDO2 signature holders (Authorities A and B)
- System administrators managing VITRA-E0 infrastructure
- QA and compliance personnel validating genomics outputs

### 2.2 Purpose

To ensure:

- **Determinism**: Bit-identical VCF outputs across runs
- **Auditability**: Cryptographic provenance for all operations
- **Security**: Dual authorization for critical promotions
- **Compliance**: HIPAA, CMMC, FDA 21 CFR Part 11, ISO 27001

---

## 3. Roles and Responsibilities

### 3.1 Technical Authority (Signature A Holder)

**Responsibilities**:

- Execute and validate pipeline runs in Z1
- Sign Z1 → Z2 promotions (single signature)
- Co-sign Z2 → Z3 promotions (dual signature)
- Monitor pipeline performance and GIAB metrics
- Maintain FIDO2 hardware device (Key A)

**Required Skills**:

- Nextflow pipeline execution
- GPU computing (NVIDIA Parabricks)
- GIAB validation interpretation
- Merkle provenance verification

### 3.2 Compliance Authority (Signature B Holder)

**Responsibilities**:

- Review GIAB validation reports (F1 ≥ 0.995)
- Co-sign Z2 → Z3 promotions (dual signature)
- Authorize emergency rollbacks
- Audit Merkle provenance chains
- Maintain FIDO2 hardware device (Key B)

**Required Skills**:

- Genomics quality control
- Regulatory compliance (HIPAA, FDA)
- Audit trail analysis
- CBOR/Merkle chain validation

### 3.3 System Administrator

**Responsibilities**:

- Deploy and maintain zone infrastructure
- Manage Guix container deployments
- Configure air-gapped Z3 environment
- Monitor system resources (GPU, storage)
- Backup genesis Merkle and FIDO2 pubkeys

---

## 4. FIDO2 Key Management

### 4.1 Key Generation 🔴 CRITICAL

**Procedure**:

1. **Obtain FIDO2 Hardware Devices**
   - YubiKey 5 (or equivalent FIDO2 device)
   - Separate devices for Key A and Key B
   - Purchase from authorized distributors

2. **Generate Ed25519 Key Pairs**

   ```bash
   # Key A (Technical Authority)
   ssh-keygen -t ed25519 -f epoch_a -N "" -C "vitra-e0-epoch-a"
   
   # Key B (Compliance Authority)
   ssh-keygen -t ed25519 -f epoch_b -N "" -C "vitra-e0-epoch-b"
   ```

3. **Extract Public Key Binaries**

   ```bash
   # Convert to 32-byte binary format (implementation-specific)
   # Store in qrVITRA/merkler-static/injected/epoch_pubkey_*.bin
   ```

4. **Store Private Keys on FIDO2 Devices**
   - Transfer private keys to YubiKey PIV slots
   - Delete private keys from filesystem
   - Verify: `ls epoch_a` should fail after transfer

### 4.2 Key Custody 🔴 CRITICAL

**Physical Security**:

- Store Key A and Key B in separate secure locations
- Use tamper-evident containers
- Maintain access logs

**Backup**:

- Create encrypted offline backups of private keys
- Split backup using Shamir's Secret Sharing (3-of-5 threshold)
- Store shares in geographically separated vaults
- Test recovery annually

### 4.3 Key Rotation 🟡 IMPORTANT

**Annual Rotation Procedure**:

1. Generate new epoch key pairs (epoch_a_v2, epoch_b_v2)
2. Update genesis Merkle with new pubkeys
3. Sign final promotion with old keys
4. Activate new keys for subsequent promotions
5. Archive old keys in secure cold storage
6. Document rotation in audit log

**Rotation Triggers** (immediate):

- Key compromise suspected
- FIDO2 device loss or theft
- Authority role change
- Security audit recommendation

---

## 5. Zone Promotion Procedures

### 5.1 Z0 → Z1 (Auto-Promotion)

**Trigger**: Genesis initialization  
**Authorization**: Automatic (no signature required)

**Procedure**:

```bash
# Initialize genesis
cd qrVITRA/scripts
./init_genesis_merkle.sh

# Deploy zone topology
./deploy_zones.sh

# Verify Z0 immutability
cat zones/Z0/ZONE_METADATA.json | jq '.properties.immutable'
# Expected: true
```

**Verification**:

- Z0 contains genesis Merkle root
- FIDO2 pubkeys injected into merkler-static
- Z1 staging directory created

### 5.2 Z1 → Z2 (Production Promotion) 🔴 CRITICAL

**Trigger**: Successful GIAB validation (F1 ≥ 0.995)  
**Authorization**: Single FIDO2 signature A  
**Authority**: Technical Lead

**Prerequisites**:

- Pipeline execution in Z1 complete
- GIAB F1 score ≥ 0.995 verified
- Precision ≥ 0.998, Recall ≥ 0.992
- Merkle DAG generated and validated

**Procedure**:

```bash
# 1. Review GIAB validation
cat results/validation/sample_validation.json | jq '.overall.f1_score'
# Expected: ≥ 0.995

# 2. Extract Merkle DAG hash
MERKLE_HASH=$(sha256sum results/provenance/provenance_dag.cbor | awk '{print $1}')

# 3. Sign with FIDO2 Key A (Technical Authority)
echo -n "$MERKLE_HASH" | ssh-keygen -Y sign -f /path/to/yubikey/epoch_a -n vitra-e0 > signature_a.sig

# 4. Execute promotion
cd zones
./promote_Z1_to_Z2.sh results/provenance/provenance_dag.cbor signature_a.sig

# 5. Verify promotion
cat Z2/artifacts/promotion_*/promotion_manifest.json | jq '.signatures.fido2_a'
```

**Verification Checklist**:

- [ ] GIAB F1 ≥ 0.995
- [ ] Signature A valid
- [ ] Merkle DAG copied to Z2
- [ ] Promotion manifest created
- [ ] Audit log updated

### 5.3 Z2 → Z3 (Archive Promotion) 🔴 CRITICAL

**Trigger**: Long-term archival requirement  
**Authorization**: Dual FIDO2 signatures A + B  
**Authorities**: Technical Lead + Compliance Lead

**Prerequisites**:

- Pipeline validated in Z2
- Air-gap environment prepared
- Network isolation verified
- Dual signature holders available

**Procedure**:

```bash
# 1. Prepare air-gapped Z3 environment
# - Disconnect network
# - Verify: ping -c 1 8.8.8.8 (should fail)

# 2. Extract Merkle DAG hash
MERKLE_HASH=$(sha256sum Z2/artifacts/promotion_*/provenance_dag.cbor | awk '{print $1}')

# 3. Sign with FIDO2 Key A (Technical Authority)
echo -n "$MERKLE_HASH" | ssh-keygen -Y sign -f /yubikey/epoch_a -n vitra-e0 > sig_a.sig

# 4. Sign with FIDO2 Key B (Compliance Authority)
echo -n "$MERKLE_HASH" | ssh-keygen -Y sign -f /yubikey/epoch_b -n vitra-e0 > sig_b.sig

# 5. Execute promotion
./promote_Z2_to_Z3.sh Z2/artifacts/promotion_*/provenance_dag.cbor sig_a.sig sig_b.sig

# 6. Verify air-gap isolation
ip link show | grep -i "state up"  # No active interfaces
```

**Verification Checklist**:

- [ ] Network isolation verified
- [ ] Dual signatures valid
- [ ] Merkle DAG copied to Z3
- [ ] Z3 marked immutable
- [ ] Physical access logged

---

## 6. Rollback Procedures

### 6.1 Emergency Rollback Authorization 🔴 CRITICAL

**Triggers** (emergency only):

- Critical pipeline bug discovered
- Data integrity compromised
- Security vulnerability in pipeline
- Regulatory compliance failure

**Authorization**: Dual FIDO2 signatures A + B + written justification

### 6.2 Z3 → Z2 Rollback

**Prerequisites**:

- Written justification (incident report)
- Dual signature holders present
- Emergency authorization from management
- Security team notified

**Procedure**:

```bash
# 1. Document justification
JUSTIFICATION="Critical bug in DeepVariant 1.5.0 affects variant calling accuracy. Rollback to validated Z2 snapshot pending fix."

# 2. Extract artifact hash
ARTIFACT_HASH=$(sha256sum Z3/artifacts/promotion_*/provenance_dag.cbor | awk '{print $1}')

# 3. Dual signatures required
echo -n "$ARTIFACT_HASH" | ssh-keygen -Y sign -f /yubikey/epoch_a -n vitra-e0 > sig_a.sig
echo -n "$ARTIFACT_HASH" | ssh-keygen -Y sign -f /yubikey/epoch_b -n vitra-e0 > sig_b.sig

# 4. Execute rollback
./rollback_Z3_to_Z2.sh Z3/artifacts/promotion_*/provenance_dag.cbor sig_a.sig sig_b.sig "$JUSTIFICATION"

# 5. Notify stakeholders
echo "Z3 → Z2 rollback executed. Incident: $JUSTIFICATION" | mail -s "VITRA-E0 Rollback" security@org.com
```

**Post-Rollback Actions**:

1. Conduct security review
2. Update pipeline to fix root cause
3. Re-validate with GIAB
4. Document in compliance audit
5. Schedule retrospective

---

## 7. Air-Gap Deployment

### 7.1 Z3 Air-Gap Configuration 🔴 CRITICAL

**Network Isolation**:

```bash
# Disable all network interfaces
sudo ip link set eth0 down
sudo ip link set wlan0 down

# Verify isolation
ping -c 1 8.8.8.8  # Should fail
curl -I https://google.com  # Should fail

# Physical verification
# - Remove ethernet cable
# - Disable WiFi hardware switch
# - Tape over ethernet port
```

**Container Deployment**:

```bash
# Transfer SquashFS container via USB
mount /dev/sdb1 /mnt/usb
cp /mnt/usb/vitra-e0-v1.0.squashfs /opt/vitra/containers/

# Verify hash
sha256sum /opt/vitra/containers/vitra-e0-v1.0.squashfs
# Compare with manifest from online system

# Mount container
sudo mount -t squashfs -o loop,ro /opt/vitra/containers/vitra-e0-v1.0.squashfs /opt/vitra/runtime
```

**Access Control**:

- Two-person rule (witnesses required)
- Physical access log
- Video surveillance
- Badge access only

---

## 8. Pipeline Execution

### 8.1 Standard Pipeline Run

**Prerequisites**:

- FASTQ files validated (MD5 checksums)
- Reference genome indexed (GRCh38)
- GIAB truth set available (if Z2+ promotion)
- GPU available (NVIDIA A100 recommended)

**Execution**:

```bash
# Load Nextflow
module load nextflow/23.04.0

# Run pipeline in Z1 (staging)
nextflow run qrVITRA/nextflow/vitra-e0-germline.nf \
  --fastq_r1 /data/HG001_R1.fastq.gz \
  --fastq_r2 /data/HG001_R2.fastq.gz \
  --ref /data/GRCh38.fa \
  --giab_truth /data/HG001_truth.vcf.gz \
  --giab_bed /data/HG001_highconf.bed \
  --outdir ./results_Z1 \
  --sample_id HG001 \
  --zone Z1 \
  -profile guix,gpu \
  -resume

# Monitor execution
tail -f .nextflow.log
```

**Performance Expectations**:

- ALIGN_FQ2BAM: 30-45 minutes (30x WGS on A100)
- CALL_VARIANTS: 25-30 minutes
- GIAB_VALIDATE: 3-5 minutes
- PROVENANCE: <5 seconds
- **Total**: ~1 hour for 30x WGS

### 8.2 Reproducibility Testing

```bash
# Run pipeline 3 times
for i in {1..3}; do
  nextflow run qrVITRA/nextflow/vitra-e0-germline.nf \
    --fastq_r1 /data/HG001_R1.fastq.gz \
    --fastq_r2 /data/HG001_R2.fastq.gz \
    --ref /data/GRCh38.fa \
    --outdir ./run_$i \
    --sample_id HG001_run_$i \
    -profile guix,gpu
done

# Verify bit-identical VCFs
cd qrVITRA/scripts
./verify_reproducibility.sh \
  --vcf run_1/vcf/HG001.vcf.gz \
  --merkle-chain run_1/provenance/provenance_dag.cbor \
  --giab-truth /data/HG001_truth.vcf.gz \
  --num-runs 3 \
  --output-dir ./reproducibility_report
```

---

## 9. Reproducibility Validation

### 9.1 VCF Hash Verification

```bash
# Extract variants (skip headers)
zcat run_1/vcf/HG001.vcf.gz | grep -v '^#' | sha256sum > hash_1.txt
zcat run_2/vcf/HG001.vcf.gz | grep -v '^#' | sha256sum > hash_2.txt
zcat run_3/vcf/HG001.vcf.gz | grep -v '^#' | sha256sum > hash_3.txt

# Compare
diff hash_1.txt hash_2.txt  # Should be identical
diff hash_2.txt hash_3.txt  # Should be identical
```

**Expected Result**: All hashes identical (bit-for-bit reproducibility)

### 9.2 GIAB F1 Score Threshold

**Z2 Promotion Criteria**:

- Overall F1 ≥ 0.995
- SNP F1 ≥ 0.995
- Indel F1 ≥ 0.990
- Precision ≥ 0.998
- Recall ≥ 0.992

**Validation**:

```bash
cat results/validation/sample_validation.json | jq '
{
  "Overall F1": .overall.f1_score,
  "SNP F1": .by_variant_type.snp.f1_score,
  "Indel F1": .by_variant_type.indel.f1_score,
  "Z2 Eligible": .zone_promotion.z2_eligible
}'
```

---

## 10. Incident Response

### 10.1 Key Compromise

**Immediate Actions**:

1. Revoke compromised key
2. Generate new epoch key pair
3. Update genesis Merkle with new pubkey
4. Notify all pipeline users
5. Invalidate pending promotions
6. Conduct forensic analysis

**Escalation**:

- Security team (immediate)
- Legal/compliance (within 24h)
- Regulatory bodies (if PHI affected)

### 10.2 Pipeline Failure

**Triage**:

```bash
# Check Nextflow logs
tail -100 .nextflow.log

# Check GPU status
nvidia-smi

# Check disk space
df -h

# Check Merkle chain
cat results/provenance/provenance.log
```

**Resolution Paths**:

- GPU OOM: Reduce batch size or use `--low-memory`
- Disk full: Clean work directory (`nextflow clean -f`)
- GIAB fail: Review validation metrics, adjust thresholds

---

## 11. Audit and Compliance

### 11.1 Monthly Audit Checklist

- [ ] Review all Z1 → Z2 promotions
- [ ] Verify FIDO2 signature validity
- [ ] Check GIAB F1 scores (≥0.995)
- [ ] Validate Merkle chain continuity
- [ ] Review air-gap access logs (Z3)
- [ ] Test FIDO2 key backups
- [ ] Verify reproducibility (3-run test)
- [ ] Document any rollbacks with justification

### 11.2 Compliance Mapping

| Requirement | VITRA-E0 Control | Evidence |
|-------------|------------------|----------|
| HIPAA § 164.312(a)(1) | Unique user identification | FIDO2 signatures |
| HIPAA § 164.312(b) | Audit controls | Merkle provenance chain |
| FDA 21 CFR 11.10(e) | Audit trail | CBOR Merkle DAG |
| FDA 21 CFR 11.50 | Signature manifestations | Dual FIDO2 signatures |
| CMMC Level 3 AC.L3-3.1.5 | Dual authorization | Z3 promotions |
| ISO 27001 A.9.4.2 | Secure log-on | FIDO2 hardware keys |

---

## 12. Appendices

### Appendix A: Glossary

- **Merkle DAG**: Directed Acyclic Graph with cryptographic hashing
- **FIDO2**: Fast Identity Online 2.0 authentication standard
- **GIAB**: Genome in a Bottle (NIST reference materials)
- **CBOR**: Concise Binary Object Representation
- **VCF**: Variant Call Format

### Appendix B: References

- NIST GIAB: <https://www.nist.gov/programs-projects/genome-bottle>
- NVIDIA Parabricks: <https://docs.nvidia.com/clara/parabricks/>
- FIDO Alliance: <https://fidoalliance.org/fido2/>
- FDA 21 CFR Part 11: <https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application>

### Appendix C: Contact Information

- **Technical Authority (Key A)**: <tech-lead@org.com>
- **Compliance Authority (Key B)**: <compliance@org.com>
- **Security Team**: <security@org.com>
- **Emergency Hotline**: +1-555-VITRA-E0

---

**Document Control**

- **Approved By**: [Name, Title]
- **Approval Date**: [YYYY-MM-DD]
- **Next Review**: [YYYY-MM-DD]
- **Revision History**:
  - v1.0.0 (2024-12-24): Initial release
