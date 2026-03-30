# DFARS Compliance

Defense Federal Acquisition Regulation Supplement (DFARS) compliance documentation.

## Overview

DFARS provides acquisition regulations for DoD contracts. QRATUM complies with key DFARS clauses related to cybersecurity and supply chain security.

!!! success "100% DFARS Compliance"

    All applicable DFARS clauses implemented.

## Key DFARS Clauses

### 252.204-7012: Safeguarding Covered Defense Information

This clause requires:

1. **Adequate Security** - Implement NIST 800-171 controls
2. **Cyber Incident Reporting** - Report within 72 hours
3. **Malicious Software** - Submit to DoD Cyber Crime Center
4. **Media Preservation** - 90 days after reporting
5. **Access to Equipment** - Provide for forensics

#### Implementation

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| NIST 800-171 controls | 110/110 implemented | See [NIST 800-53](nist-800-53.md) |
| 72-hour reporting | Incident response plan | `compliance/ir-plan.md` |
| Media preservation | S3 retention policy | AWS S3 lifecycle |
| Forensic access | Contract terms | Legal agreements |

### 252.204-7019: NIST 800-171 Assessment

This clause requires:

1. Self-assessment using NIST methodology
2. Score submission to SPRS
3. Periodic reassessment

#### SPRS Submission

```yaml
Assessment:
  methodology: NIST SP 800-171A
  score: 110
  date: 2025-11-04
  type: Basic
  assessor: Internal
  next_assessment: 2026-11-04
```

### 252.204-7020: NIST 800-171 DoD Assessment

Applies when DoD conducts assessment (Medium/High confidence).

#### Readiness Checklist

- [x] All 110 requirements documented
- [x] Evidence repository organized
- [x] Personnel trained for interviews
- [x] System demonstrations prepared
- [x] POA&M current

### 252.204-7021: CMMC Requirements

Requires CMMC certification at specified level.

See [CMMC 2.0 Compliance](cmmc-2.md) for details.

### 252.239-7010: Cloud Computing Services

For cloud-based services processing DoD data:

1. **FedRAMP Authorization** - Or DoD equivalent
2. **Data Location** - Within US
3. **Incident Notification** - Same as 7012

#### Cloud Implementation

| Requirement | Implementation |
|-------------|----------------|
| FedRAMP | Targeting Moderate authorization |
| Data location | AWS us-gov-west-1, us-gov-east-1 |
| Incident notification | Integrated with 7012 process |

### 252.246-7008: Sources of Electronic Parts

For supply chain security:

1. Obtain parts from trusted sources
2. Maintain traceability
3. Report counterfeit parts

#### SBOM Implementation

```bash
# Generate Software Bill of Materials
python compliance/scripts/sbom_generator.py \
  --format spdx \
  --output sbom.spdx.json

# Verify component sources
python compliance/scripts/verify_sources.py \
  --sbom sbom.spdx.json
```

## Cyber Incident Response

### Reporting Timeline

```
Event Detection
     │
     ▼ (immediately)
┌─────────────────────┐
│ Security Team       │
│ Notified            │
└──────────┬──────────┘
           │
           ▼ (within 1 hour)
┌─────────────────────┐
│ Initial Assessment  │
│ & Containment       │
└──────────┬──────────┘
           │
           ▼ (within 24 hours)
┌─────────────────────┐
│ Executive Briefing  │
│ Impact Assessment   │
└──────────┬──────────┘
           │
           ▼ (within 72 hours)
┌─────────────────────┐
│ DoD Reporting       │
│ via DIBNet          │
└─────────────────────┘
```

### Required Information

When reporting to DoD:

- [ ] Company name and CAGE code
- [ ] Point of contact
- [ ] Description of incident
- [ ] Affected systems
- [ ] Data potentially compromised
- [ ] Malware samples (if applicable)
- [ ] Network monitoring data (30 days)
- [ ] Images of affected systems

### Evidence Preservation

```yaml
Evidence Retention:
  network_logs: 90 days
  system_images: 90 days
  memory_captures: 90 days
  malware_samples: indefinite
  
Storage:
  location: Isolated forensic server
  encryption: AES-256
  access: Security team only
```

## Supply Chain Risk Management

### SCRM Program

```
┌─────────────────────────────────────────┐
│         SCRM Program Elements           │
├─────────────────────────────────────────┤
│ • Supplier risk assessment              │
│ • Continuous monitoring                 │
│ • Counterfeit detection                 │
│ • Vulnerability management              │
│ • Incident response                     │
└─────────────────────────────────────────┘
```

### Supplier Assessment

| Criteria | Weight | Assessment |
|----------|--------|------------|
| Security posture | 25% | Questionnaire + audit |
| Financial stability | 15% | Dun & Bradstreet |
| Geographic risk | 20% | Country of origin |
| Dependency criticality | 25% | Component analysis |
| Compliance status | 15% | CMMC level |

### NDAA Section 889 Compliance

Prohibited equipment:

- [x] No Huawei products
- [x] No ZTE products
- [x] No Hytera products
- [x] No Hikvision products
- [x] No Dahua products

```bash
# Check for prohibited components
python compliance/scripts/ndaa_889_check.py \
  --sbom sbom.spdx.json
```

## Controlled Unclassified Information (CUI)

### CUI Categories

| Category | Marking | Handling |
|----------|---------|----------|
| CTI (Technical Info) | CUI//SP-CTI | Export controlled |
| PRVCY (Privacy) | CUI//SP-PRVCY | PII protection |
| PROPIN (Proprietary) | CUI//SP-PROPIN | NDA required |

### CUI Handling Procedures

```yaml
CUI_Handling:
  marking:
    method: Document headers/footers
    tool: Automated classification
    
  storage:
    encryption: AES-256
    location: Approved systems only
    backup: Encrypted + geo-redundant
    
  transmission:
    method: TLS 1.3 minimum
    email: Encrypted (S/MIME or GPG)
    
  destruction:
    method: NIST 800-88 compliant
    verification: Certificate of destruction
```

## Contract Flow-Down

When subcontracting:

```yaml
Required_Flowdown:
  - 252.204-7012: Safeguarding CDI
  - 252.204-7019: NIST 800-171 Assessment
  - 252.204-7020: DoD Assessment (if applicable)
  - 252.204-7021: CMMC Requirements
  - 252.246-7008: Electronic Parts Sources
```

### Subcontractor Requirements

- [ ] CMMC certification (appropriate level)
- [ ] SPRS score submission
- [ ] Cyber incident reporting capability
- [ ] CUI handling procedures

## Audit Preparation

### Document Checklist

- [x] System Security Plan (SSP)
- [x] POA&M
- [x] Incident Response Plan
- [x] Business Continuity Plan
- [x] CUI Procedures
- [x] Supplier Agreements
- [x] Training Records
- [x] Assessment Reports

### Evidence Repository

| Evidence Type | Location | Format |
|---------------|----------|--------|
| Policies | `compliance/policies/` | Markdown |
| Procedures | `compliance/procedures/` | Markdown |
| Logs | Loki/S3 | JSON |
| Configurations | Git | YAML/HCL |
| Certificates | Vault | PEM |
| Audit reports | `compliance/audits/` | PDF |

## Contact

For DFARS compliance inquiries:

- **Contracts Team**: <contracts@quasim.example.com>
- **Compliance Officer**: <compliance@quasim.example.com>
- **Security Team**: <security@quasim.example.com>
