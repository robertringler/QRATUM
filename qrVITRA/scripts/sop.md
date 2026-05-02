# VITRA-E0 Standard Operating Procedures (SOP)
# Biokey-Enabled Sovereign Genomic Operations

**Version**: 1.0.0  
**Effective Date**: 2024-12-24  
**Classification**: CONTROLLED UNCLASSIFIED  
**Review Cycle**: Quarterly  

---

## Table of Contents

1. [Biokey Overview](#1-biokey-overview)
2. [Ephemeral Biokey Derivation](#2-ephemeral-biokey-derivation)
3. [Dual Biokey + FIDO2 Workflow](#3-dual-biokey--fido2-workflow)
4. [Zero-Knowledge Proof Verification](#4-zero-knowledge-proof-verification)
5. [Zone Promotions with Biokey](#5-zone-promotions-with-biokey)
6. [Air-Gapped Biokey Operations](#6-air-gapped-biokey-operations)
7. [Incident Response](#7-incident-response)
8. [Compliance & Legal](#8-compliance--legal)

---

## 1. Biokey Overview

### 1.1 Purpose

Biokeys provide ephemeral biometric authentication for sovereign genomic operations without storing plaintext genetic data.

### 1.2 Security Properties

- **Ephemeral**: Keys exist only in RAM (never written to disk)
- **Zero-Knowledge**: Public hash doesn't reveal DNA sequence
- **Dual-Control**: Critical operations require 2 operators
- **Compliant**: HIPAA/GDPR/BIPA regulations met

### 1.3 Key Terminology

| Term | Definition |
|------|-----------|
| **Biokey** | Ephemeral cryptographic key derived from genomic SNPs |
| **Public Hash** | SHA3-256 hash of private key (safe to store) |
| **ZKP** | Zero-Knowledge Proof (verify without revealing genome) |
| **Dual-Sig** | Dual signature requiring 2 operators + 2 FIDO2 keys |
| **tmpfs** | RAM-only filesystem (no disk writes) |

---

## 2. Ephemeral Biokey Derivation

### 2.1 Prerequisites

- Operator VCF file (WGS data, QUAL≥30, DP≥10)
- Access to secure workstation
- merkler-static binary installed
- Sufficient RAM (≥4GB for tmpfs)

### 2.2 Procedure

**Step 1: Verify VCF File Integrity**

```bash
# Check file exists and is readable
ls -lh /path/to/operator.vcf.gz

# Verify checksum (if available)
sha256sum /path/to/operator.vcf.gz
# Compare with registered checksum
```

**Step 2: Derive Biokey**

```bash
# Navigate to VITRA-E0 directory
cd /path/to/qrVITRA

# Derive biokey (opens new shell with active session)
./scripts/biokey/derive_biokey.sh operator-id /path/to/operator.vcf.gz 192

# Output should show:
# - Public hash
# - Loci count
# - Session start time
# - Environment variables
```

**Step 3: Verify Session Active**

```bash
# Check environment variables
echo $VITRA_BIOKEY_PUBLIC_HASH
echo $VITRA_BIOKEY_OPERATOR
echo $VITRA_BIOKEY_SESSION_START

# Verify tmpfs mounted
mount | grep vitra-biokey
```

**Step 4: Register Biokey**

Biokey is automatically registered in `configs/operator_biokeys.json`.

Verify entry:

```bash
cat configs/operator_biokeys.json | grep operator-id
```

### 2.3 Session Management

**Session Timeout**: 60 minutes (default)

**Manual Session Extension**:

```bash
# Re-export session start time
export VITRA_BIOKEY_TIMESTAMP=$(date +%s)
```

**Session Cleanup**:

```bash
# Exit shell to trigger automatic cleanup
exit

# Verify cleanup
mount | grep vitra-biokey  # Should be empty
```

### 2.4 Troubleshooting

| Issue | Resolution |
|-------|-----------|
| "VCF file not found" | Verify path, check permissions |
| "Insufficient high-quality SNPs" | Use different VCF or lower quality threshold |
| "tmpfs mount failed" | Check sudo access, verify RAM available |
| "Session timeout" | Re-derive biokey (session expired) |

---

## 3. Dual Biokey + FIDO2 Workflow

### 3.1 When Required

Dual authorization required for:

- **SENSITIVE**: System configuration (biokey + FIDO2)
- **CRITICAL**: Zone promotions, self-improvement (dual biokey + dual FIDO2)
- **EXISTENTIAL**: Architecture changes (dual biokey + dual FIDO2 + board)

### 3.2 Procedure

**Step 1: Operator A Derives Biokey**

```bash
# Terminal 1: Operator A
./scripts/biokey/derive_biokey.sh operator-alice /secure/alice.vcf.gz

# Keep this terminal open
```

**Step 2: Operator B Derives Biokey**

```bash
# Terminal 2: Operator B (separate workstation)
./scripts/biokey/derive_biokey.sh operator-bob /secure/bob.vcf.gz

# Keep this terminal open
```

**Step 3: Verify FIDO2 Devices**

```bash
# Check device A (Operator A workstation)
ls -l /dev/hidraw0

# Check device B (Operator B workstation)
ls -l /dev/hidraw1
```

**Step 4: Execute Critical Operation**

```bash
# From Operator A terminal
./scripts/deploy_zones.sh promote-z2-to-z3

# System prompts for:
# 1. Operator A biokey verification
# 2. Operator B biokey verification
# 3. FIDO2 device A authentication
# 4. FIDO2 device B authentication
```

**Step 5: Verify Dual Signature**

```bash
# Check promotion record
cat zones/Z3/promotion_z2_z3.json

# Should contain both operator hashes
```

### 3.3 Geographic Separation

For high-security operations, enforce geographic separation:

- Operator A: Site 1
- Operator B: Site 2 (≥100km away)
- FIDO2 devices registered to different physical locations

Verify in `configs/operator_biokeys.json`:

```json
{
  "operator_a": {
    "location": "Site-1-Building-A"
  },
  "operator_b": {
    "location": "Site-2-Building-B"
  }
}
```

---

## 4. Zero-Knowledge Proof Verification

### 4.1 Purpose

Verify operator credentials without revealing genome data.

### 4.2 Procedure

**Step 1: Generate Challenge**

```bash
# Verifier generates random challenge
./scripts/biokey/verify_biokey.sh operator-alice

# Output: 256-bit hex challenge
```

**Step 2: Operator Generates Proof**

```bash
# Operator (with active biokey session) generates proof
merkler-static prove $VITRA_BIOKEY_JSON <challenge-hex> > proof.json
```

**Step 3: Verify Proof**

```bash
# Verifier checks proof
merkler-static verify-zkp proof.json

# Output: VALID or INVALID
```

**Step 4: Document Verification**

```bash
# Log verification result
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) - ZKP verification: operator-alice - VALID" >> audit.log
```

### 4.3 Security Notes

- Challenge must be unique per session (prevent replay)
- Proof doesn't reveal private key or genome data
- Network traffic contains no DNA sequences
- Public hash is safe to transmit over insecure channels

---

## 5. Zone Promotions with Biokey

### 5.1 Zone Topology

| Zone | Description | Authorization | Air-Gap |
|------|-------------|---------------|---------|
| Z0 | Genesis (immutable) | None | No |
| Z1 | Staging | Auto-promoted | No |
| Z2 | Production | Biokey + FIDO2 | No |
| Z3 | Archive | Dual Biokey + Dual FIDO2 | Yes |

### 5.2 Z1 → Z2 Promotion

**Authorization**: Single biokey + FIDO2

```bash
# 1. Derive biokey
./scripts/biokey/derive_biokey.sh operator-director /secure/director.vcf.gz

# 2. Insert FIDO2 key
# (Physical device at /dev/hidraw0)

# 3. Execute promotion
./scripts/deploy_zones.sh promote-z1-to-z2

# 4. Verify promotion
cat zones/Z2/promotion_z1_z2.json
```

### 5.3 Z2 → Z3 Promotion

**Authorization**: Dual biokey + dual FIDO2

```bash
# 1. Operator A derives biokey
./scripts/biokey/derive_biokey.sh operator-director /secure/director.vcf.gz

# 2. Operator B derives biokey (separate terminal/workstation)
./scripts/biokey/derive_biokey.sh operator-cso /secure/cso.vcf.gz

# 3. Insert both FIDO2 keys

# 4. Execute promotion
./scripts/deploy_zones.sh promote-z2-to-z3

# 5. Activate air-gap
sudo ifconfig eth0 down  # Disable network

# 6. Mount Z3 as read-only
sudo mount -o remount,ro zones/Z3
```

### 5.4 Rollback Procedure

If promotion needs to be reversed:

```bash
# 1. Verify biokey session active

# 2. Create rollback record
cat > zones/Z2/rollback_$(date +%s).json <<EOF
{
  "rollback_from": "Z3",
  "rollback_to": "Z2",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "operator": "$VITRA_BIOKEY_OPERATOR",
  "reason": "Detected error in promoted data",
  "authorized_by": "Chief Scientific Officer"
}
EOF

# 3. Restore from Z2 backup
cp -r zones/Z2_backup/* zones/Z2/

# 4. Document in audit log
```

---

## 6. Air-Gapped Biokey Operations

### 6.1 VCF Transport Protocol

**Encrypted USB Transport**:

```bash
# 1. Encrypt VCF file
gpg --encrypt --recipient operator@domain.com operator.vcf.gz

# 2. Copy to encrypted USB
cp operator.vcf.gz.gpg /media/encrypted-usb/

# 3. Physically transport USB to air-gapped system

# 4. On air-gapped system, decrypt
gpg --decrypt operator.vcf.gz.gpg > operator.vcf.gz

# 5. Verify integrity
sha256sum operator.vcf.gz
# Compare with known-good checksum

# 6. Derive biokey
./scripts/biokey/derive_biokey.sh operator-id operator.vcf.gz
```

### 6.2 Air-Gap Verification

```bash
# Verify no network connectivity
ping -c 1 8.8.8.8 || echo "Air-gap confirmed"

# Check interfaces disabled
ip link show | grep "state DOWN"

# Verify no DNS resolution
nslookup example.com || echo "DNS disabled"
```

### 6.3 Air-Gapped Pipeline Execution

```bash
# Ensure all dependencies pre-installed
nextflow run nextflow/vitra-e0-germline.nf \
  --fastq_r1 sample_R1.fastq.gz \
  --fastq_r2 sample_R2.fastq.gz \
  --ref /data/GRCh38.fa \
  --biokey-required true \
  --safety-level CRITICAL \
  -profile airgap,gpu,biokey \
  -offline
```

---

## 7. Incident Response

### 7.1 Biokey Compromise

**Indicators**:
- Unauthorized operations signed with operator biokey
- VCF file accessed without authorization
- Suspicious biokey derivations in audit log

**Response**:

```bash
# IMMEDIATE (within 15 minutes):

# 1. Revoke biokey
./scripts/biokey/revoke_biokey.sh operator-id "COMPROMISED"

# 2. Remove from operator registry
# Edit configs/operator_biokeys.json, set status: "revoked"

# 3. Kill all active sessions
pkill -u operator-id

# 4. Shred all tmpfs
find /dev/shm -name "*biokey*" -exec shred -n 3 -z -u {} \;

# 5. Alert security team
echo "SECURITY INCIDENT: Biokey compromise operator-id $(date)" | \
  mail -s "URGENT: Biokey Compromise" security@domain.com

# SHORT-TERM (within 24 hours):

# 6. Forensic analysis
# - Review audit logs for unauthorized operations
# - Identify all operations signed with compromised biokey

# 7. Rollback compromised operations
# - Use Merkle chain to identify affected blocks
# - Rollback to last known-good state

# 8. Re-sequence operator
# - Obtain new VCF file from fresh sequencing
# - Register new biokey

# LONG-TERM (within 1 week):

# 9. Root cause analysis
# 10. Update security procedures
# 11. Mandatory security training for all operators
```

### 7.2 Memory Dump Attack

**Detection**:

```bash
# Monitor for unauthorized memory access
ps aux | grep gcore  # Check for memory dump tools
dmesg | grep -i "memory dump"  # Check kernel logs
```

**Response**:

```bash
# 1. Terminate biokey session immediately
exit

# 2. Verify memory wiped
# (Automatic with zeroize on drop)

# 3. Report incident
./scripts/report_incident.sh "memory-dump-attempt" "$OPERATOR_ID"

# 4. Re-derive biokey in secure environment
```

### 7.3 Replay Attack

**Detection**:

Merkle chain shows duplicate timestamps or out-of-order operations.

**Response**:

```bash
# 1. Identify replayed signature
# Check Merkle chain for duplicate entries

# 2. Invalidate affected operations
# Rollback to pre-replay state

# 3. Enforce stricter timestamp validation
# Update provenance module with tighter time windows
```

---

## 8. Compliance & Legal

### 8.1 HIPAA Compliance

**Requirements**:
- No PHI (Protected Health Information) on disk
- Audit trail for all genomic operations
- Encryption at rest and in transit
- Access controls (biokey authentication)

**VITRA-E0 Implementation**:

| Requirement | Implementation |
|-------------|----------------|
| PHI Storage | VCF in RAM-only tmpfs |
| Audit Trail | Merkle chain with all operations |
| Encryption | GPG for VCF transport, HTTPS for network |
| Access Control | Biokey + FIDO2 authentication |

**Verification**:

```bash
# 1. Verify no VCF on disk
find / -name "*.vcf" -o -name "*.vcf.gz" 2>/dev/null | grep -v /media/backup

# 2. Check audit trail
cat zones/Z2/provenance.json | jq '.workflow'

# 3. Verify encryption
gpg --list-keys operator@domain.com
```

### 8.2 GDPR Article 9 (Genetic Data)

**Principles**:

1. **Lawfulness**: Explicit consent from operator
2. **Purpose Limitation**: Biokey for authentication only
3. **Data Minimization**: Only public hash stored
4. **Accuracy**: Operator can update VCF
5. **Storage Limitation**: Ephemeral (session-based)
6. **Integrity & Confidentiality**: ZKP + dual-control

**Consent Form**:

```
BIOKEY ENROLLMENT CONSENT FORM

I, [Operator Name], consent to:

1. Derivation of cryptographic keys from my genomic data
2. Storage of public hash (not revealing genome) in operator registry
3. Use of biokey for authentication in genomic operations
4. Ephemeral nature of biokey (destroyed after session)

I understand:
- My genome data is NOT stored on disk
- Public hash does NOT reveal my DNA sequence
- I can revoke consent at any time
- My biokey will be destroyed on session exit

Signature: _______________  Date: __________
```

### 8.3 BIPA (Biometric Information Privacy Act)

**Requirements**:

1. Written policy
2. Informed consent
3. Retention schedule
4. Destruction protocol
5. Security measures
6. No sale of biometric data

**VITRA-E0 Compliance**:

| Requirement | Implementation |
|-------------|----------------|
| Written Policy | This SOP document |
| Informed Consent | Biokey enrollment form |
| Retention | Session-based (ephemeral) |
| Destruction | Auto-wipe on exit (shred -n 3) |
| Security | ZKP, dual-control, air-gap |
| No Sale | Public hash never sold |

### 8.4 21 CFR Part 11 (Electronic Signatures)

**Requirements**:

- Signature uniqueness
- Signature non-repudiation
- Audit trail
- Timestamp accuracy

**VITRA-E0 Implementation**:

```bash
# Biokey signature includes:
# 1. Operator public hash (unique)
# 2. Timestamp (accurate)
# 3. Message hash (non-repudiation)
# 4. Merkle chain entry (audit trail)

# Example signature
{
  "operator": "operator-alice",
  "public_hash": "a1b2c3...",
  "timestamp": "2024-12-24T00:00:00Z",
  "message_hash": "d4e5f6...",
  "merkle_block": 12345
}
```

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **SNP** | Single Nucleotide Polymorphism (genetic variation) |
| **VCF** | Variant Call Format (genomic data file) |
| **tmpfs** | Temporary filesystem (RAM-only, no disk) |
| **SHA3-256** | Secure Hash Algorithm 3 (256-bit, post-quantum) |
| **FIDO2** | Fast Identity Online 2 (hardware authentication) |
| **HSM** | Hardware Security Module (secure key storage) |
| **APT** | Advanced Persistent Threat (nation-state adversary) |

## Appendix B: Contact Information

| Role | Contact |
|------|---------|
| Security Incidents | security@qratum.ai |
| Biokey Support | biokey-support@qratum.ai |
| Compliance Questions | compliance@qratum.ai |
| General Inquiries | contact@qratum.ai |

## Appendix C: Revision History

| Version | Date | Changes | Approved By |
|---------|------|---------|-------------|
| 1.0.0 | 2024-12-24 | Initial release | Chief Scientific Officer |

---

**END OF DOCUMENT**
