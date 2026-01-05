# ITAR Export Control Compliance

## Overview

This document describes ITAR (International Traffic in Arms Regulations) compliance procedures for QuASIM aerospace integrations.

## Export Classification

QuASIM integrations include adapters for aerospace simulation tools that may process export-controlled technical data. Proper classification and handling procedures must be followed.

### Components Classification

| Component | ECCN/USML | Export Controlled | Notes |
|-----------|-----------|-------------------|-------|
| QuASIM Core Runtime | N/A | No | General-purpose tensor operations |
| CFD Adapters | Check USML Cat VIII | Depends | May process controlled aero data |
| FEA Composites | Check USML Cat VIII | Depends | Material properties may be controlled |
| Orbital MC | Check USML Cat IV | Depends | Trajectory data may be controlled |
| API Service | N/A | No | Infrastructure only, data-agnostic |

## ITAR-Clean Build Procedures

### Option 1: Public Artifacts (No Controlled Data)

Build containers and packages without any export-controlled data:

```bash
# Clean build with no controlled data
make pack QUASIM_NO_CONTROLLED_DATA=1

# Verify no controlled files
./scripts/check_export_compliance.sh
```

This produces public artifacts suitable for:

- Open source distribution
- Non-controlled development environments
- CI/CD pipelines
- Public cloud deployments

### Option 2: Controlled Build (Restricted Distribution)

For builds that include controlled technical data:

```bash
# Controlled build (restricted distribution)
make pack QUASIM_CONTROLLED_BUILD=1

# Mark artifacts with export control notice
./scripts/mark_controlled_artifacts.sh
```

Controlled artifacts must:

- Include export control markings
- Be distributed only to authorized recipients
- Be stored in approved secure facilities
- Follow organizational ITAR procedures

## Data Handling Guidelines

### Input Data Classification

Engineers must classify input data before processing:

1. **Unclassified/Public**: No restrictions
2. **Controlled Unclassified**: Follow handling procedures
3. **Export Controlled (ITAR)**: Follow ITAR procedures
4. **Classified**: Follow DoD classification procedures

### Processing Controlled Data

When processing export-controlled data:

1. ✅ Use approved, controlled environments
2. ✅ Limit access to authorized personnel only
3. ✅ Mark all outputs with appropriate control markings
4. ✅ Maintain audit logs of data access
5. ❌ Never mix controlled and public data in same job
6. ❌ Never store controlled data in public cloud without approval

### Output Data Marking

All output artifacts must be marked according to input classification:

```yaml
# Example job config with export control marking
job_config:
  export_control:
    classification: "ITAR"
    category: "USML Category VIII"
    marking: "EXPORT CONTROLLED - ITAR"
    authorized_recipients: ["US_PERSONS_ONLY"]
```

## Compliance Checklist

Before distributing QuASIM integrations:

- [ ] Determine if technical data is export controlled
- [ ] Classify components according to USML/ECCN
- [ ] Apply appropriate export control markings
- [ ] Verify recipient authorization (for controlled data)
- [ ] Document export decision in compliance log
- [ ] Obtain required export licenses if applicable
- [ ] Train users on ITAR handling requirements

## Automated Compliance Checks

Run automated compliance checks:

```bash
# Check for controlled data markers
make check-export-compliance

# Scan for common controlled keywords
./scripts/scan_controlled_keywords.sh

# Verify artifact markings
./scripts/verify_export_markings.sh
```

## Public Release Process

For public release of QuASIM integrations:

1. **Technical Review**: Ensure no controlled data in release
2. **Legal Review**: Obtain approval from export control officer
3. **Documentation**: Update export classification guide
4. **Marking**: Apply appropriate public release markings
5. **Distribution**: Use approved public distribution channels

## Restricted Release Process

For controlled releases to authorized recipients:

1. **Recipient Verification**: Confirm recipient authorization
2. **License Check**: Verify required export licenses
3. **Secure Transfer**: Use approved secure transfer methods
4. **Access Control**: Implement access restrictions
5. **Audit Trail**: Log all transfers and access

## Penalties for Non-Compliance

ITAR violations carry severe penalties:

- Civil penalties up to $1,000,000 per violation
- Criminal penalties including imprisonment
- Debarment from government contracts
- Reputational damage

## Resources

- [ITAR Regulations (22 CFR 120-130)](https://www.pmddtc.state.gov/ddtc_public/ddtc_public?id=ddtc_public_portal_itar_landing)
- [USML Categories](https://www.pmddtc.state.gov/sys_attachment.do?sys_id=24d528fddbfc930044f9ff621f961987)
- [ECCN Classification](https://www.bis.doc.gov/index.php/licensing/commerce-control-list-classification)
- Organizational Export Control Office
- Legal/Compliance Department

## Contact

For export control questions:

- Export Control Office: [contact info]
- Legal Department: [contact info]
- ITAR Compliance Officer: [contact info]

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2025-11-02 | QuASIM Team | Initial version |

---

**IMPORTANT**: This document provides general guidance. Always consult with your organization's export control office and legal counsel for specific compliance requirements.
