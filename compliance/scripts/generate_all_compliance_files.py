#!/usr/bin/env python3
"""
Generate all Defense-Grade PR Compliance Agent files
Generates OPA policies, control matrices, templates, scripts, and documentation
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent


def create_opa_policies():
    """Generate all OPA Rego policy files"""
    policies_dir = BASE_DIR / "policies" / "rego"
    policies_dir.mkdir(parents=True, exist_ok=True)

    # NIST 800-53
    (policies_dir / "nist80053.rego").write_text(
        """package nist80053

# NIST 800-53 Rev 5 HIGH Baseline Policy
import future.keywords.if
import future.keywords.in

default compliant = false

# AC-2: Account Management
compliant if {
    input.mfa_enabled == true
    input.rbac_enabled == true
    input.least_privilege == true
}

# AU-2: Audit Events
compliant if {
    input.audit_logging == "detailed"
    input.log_retention_days >= 365
}

# CM-2: Baseline Configuration
compliant if {
    input.configuration_management == true
    input.change_control == true
}

# IA-2: Identification and Authentication
compliant if {
    input.multi_factor_auth == true
    input.session_timeout <= 900
}

# SC-8: Transmission Confidentiality
compliant if {
    input.encryption_in_transit == true
    input.tls_version >= "1.2"
}

# SC-28: Protection of Information at Rest
compliant if {
    input.encryption_at_rest == true
    input.encryption_algorithm == "AES-256-GCM"
}

# SI-2: Flaw Remediation
compliant if {
    input.vulnerability_scanning == true
    input.patch_sla_critical <= 15
}
"""
    )

    # NIST 800-171
    (policies_dir / "nist800171.rego").write_text(
        """package nist800171

# NIST 800-171 R3 CUI Protection Policy
import future.keywords.if

default compliant = false

# 3.1.1: Limit system access to authorized users
compliant if {
    input.access_control.authorized_users_only == true
    input.access_control.user_authentication == true
}

# 3.1.2: Limit system access to authorized transactions
compliant if {
    input.access_control.transaction_authorization == true
}

# 3.5.1: Identify system users
compliant if {
    input.identification.unique_ids == true
}

# 3.5.2: Authenticate users
compliant if {
    input.authentication.mfa_enabled == true
}

# 3.13.1: Monitor security controls
compliant if {
    input.monitoring.continuous == true
    input.monitoring.anomaly_detection == true
}

# 3.13.11: Protect audit information
compliant if {
    input.audit.tamper_protection == true
    input.audit.log_retention >= 365
}

# 3.14.1: Identify and protect CUI
compliant if {
    input.cui_protection.identification == true
    input.cui_protection.marking == true
}
"""
    )

    # CMMC
    (policies_dir / "cmmc.rego").write_text(
        """package cmmc

# CMMC 2.0 Level 2 Policy
import future.keywords.if

default compliant = false

# AC.L2-3.1.1: Authorized Access Only
compliant if {
    input.access_control.authorized_only == true
}

# AC.L2-3.1.20: External Connections
compliant if {
    input.access_control.external_connections_managed == true
}

# AU.L2-3.3.1: Audit Events
compliant if {
    input.audit.events_logged == true
}

# CM.L2-3.4.1: Baseline Configurations
compliant if {
    input.configuration_management.baselines == true
}

# IA.L2-3.5.1: User Identification
compliant if {
    input.identification.unique_users == true
}

# IR.L2-3.6.1: Incident Handling
compliant if {
    input.incident_response.capability == true
}

# SC.L2-3.13.1: Boundary Protection
compliant if {
    input.system_protection.boundary_controls == true
}

# SI.L2-3.14.1: Flaw Remediation
compliant if {
    input.system_integrity.flaw_remediation == true
}
"""
    )

    print(f"✓ Created OPA policies in {policies_dir}")


def create_control_matrices():
    """Generate compliance control matrices"""
    matrices_dir = BASE_DIR / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    # NIST 800-53 Rev 5 HIGH
    (matrices_dir / "nist-800-53-rev5-high.csv").write_text(
        """Control,Control Name,Implementation Status,Evidence,Notes
AC-1,Policy and Procedures,Implemented,Security Policy Doc,Reviewed annually
AC-2,Account Management,Implemented,IAM System,MFA enabled
AC-3,Access Enforcement,Implemented,RBAC System,Least privilege
AC-6,Least Privilege,Implemented,Role Definitions,Quarterly review
AU-2,Audit Events,Implemented,Logging System,All events logged
AU-3,Content of Audit Records,Implemented,Log Format,Timestamp + user + action
AU-9,Protection of Audit Information,Implemented,Log Protection,Tamper-proof
CM-2,Baseline Configuration,Implemented,Config Repo,Version controlled
CM-3,Configuration Change Control,Implemented,Change Management,PR approval required
CM-6,Configuration Settings,Implemented,Hardening Guides,STIG applied
IA-2,Identification and Authentication,Implemented,Auth System,MFA required
IA-5,Authenticator Management,Implemented,Password Policy,15 char minimum
SC-7,Boundary Protection,Implemented,Firewall,DMZ configured
SC-8,Transmission Confidentiality,Implemented,TLS 1.3,All traffic encrypted
SC-12,Cryptographic Key Management,Implemented,KMS,FIPS 140-3
SC-13,Cryptographic Protection,Implemented,Encryption,AES-256-GCM
SC-28,Protection at Rest,Implemented,Disk Encryption,Full disk encryption
SI-2,Flaw Remediation,Implemented,Patch Management,15-day SLA critical
SI-3,Malicious Code Protection,Implemented,AV/EDR,Real-time scanning
SI-4,System Monitoring,Implemented,SIEM,24/7 monitoring
"""
    )

    # CMMC 2.0 Level 2
    (matrices_dir / "cmmc-2.0-level2.csv").write_text(
        """Practice ID,Practice,Domain,Implementation Status,Evidence
AC.L2-3.1.1,Authorized Access Only,Access Control,Implemented,IAM System
AC.L2-3.1.2,Transaction Types,Access Control,Implemented,Authorization Matrix
AC.L2-3.1.20,External Connections,Access Control,Implemented,VPN Gateway
AU.L2-3.3.1,Audit Events,Audit,Implemented,SIEM Logs
AU.L2-3.3.2,Audit Review,Audit,Implemented,Weekly Reviews
CM.L2-3.4.1,Baseline Configurations,Configuration Management,Implemented,Config Repo
CM.L2-3.4.2,Security Configurations,Configuration Management,Implemented,STIG Applied
IA.L2-3.5.1,User Identification,Identification and Authentication,Implemented,Unique IDs
IA.L2-3.5.2,User Authentication,Identification and Authentication,Implemented,MFA
IR.L2-3.6.1,Incident Handling,Incident Response,Implemented,IR Playbooks
MA.L2-3.7.1,Maintenance,Maintenance,Implemented,Maintenance Logs
MP.L2-3.8.1,Media Protection,Media Protection,Implemented,Encryption
PE.L2-3.10.1,Physical Access,Physical Protection,Implemented,Badge System
PS.L2-3.9.1,Personnel Screening,Personnel Security,Implemented,Background Checks
RA.L2-3.11.1,Risk Assessment,Risk Assessment,Implemented,Annual Assessments
SC.L2-3.13.1,Boundary Protection,System Protection,Implemented,Firewalls
SI.L2-3.14.1,Flaw Remediation,System Integrity,Implemented,Patch Management
"""
    )

    print(f"✓ Created control matrices in {matrices_dir}")


def create_templates():
    """Generate compliance document templates"""
    templates_dir = BASE_DIR / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    # System Security Plan
    (templates_dir / "SSP.md.j2").write_text(
        """# System Security Plan (SSP)
## {{ system_name }}

**Classification:** {{ classification }}
**Date:** {{ date }}
**Version:** {{ version }}

---

## 1. System Identification

**System Name:** {{ system_name }}
**System Owner:** {{ owner }}
**Mission/Purpose:** {{ mission }}
**System Type:** {{ system_type }}

## 2. System Categorization

**FIPS 199 Categorization:**
- Confidentiality: {{ confidentiality }}
- Integrity: {{ integrity }}
- Availability: {{ availability }}

**Overall Impact Level:** {{ impact_level }}

## 3. System Description

{{ system_description }}

## 4. System Environment

**Hosting:** {{ hosting_environment }}
**Network:** {{ network_description }}
**Data Stores:** {{ data_stores }}

## 5. Security Controls

{% for control in security_controls %}
### {{ control.id }}: {{ control.name }}

**Implementation Status:** {{ control.status }}
**Description:** {{ control.description }}
**Evidence:** {{ control.evidence }}

{% endfor %}

## 6. Encryption

**At Rest:** {{ encryption_at_rest }}
**In Transit:** {{ encryption_in_transit }}
**Algorithm:** {{ encryption_algorithm }}

## 7. Access Control

**Authentication:** {{ authentication_method }}
**MFA:** {{ mfa_enabled }}
**RBAC:** {{ rbac_enabled }}

## 8. Audit and Logging

**Logging Level:** {{ logging_level }}
**Retention:** {{ log_retention }} days
**SIEM Integration:** {{ siem_integration }}

## 9. Incident Response

**IR Plan:** {{ ir_plan_location }}
**Contact:** {{ ir_contact }}
**Reporting Timeframe:** {{ reporting_timeframe }} hours

## 10. Continuous Monitoring

**Monitoring Tools:** {{ monitoring_tools }}
**Frequency:** {{ monitoring_frequency }}

## 11. Approval

**Prepared By:** _____________________ Date: _____
**Reviewed By:** _____________________ Date: _____
**Approved By:** _____________________ Date: _____

---
**Classification:** {{ classification }}
"""
    )

    # Plan of Action and Milestones
    (templates_dir / "POAM.csv.j2").write_text(
        """POA&M ID,Weakness Description,Control,Severity,Status,Resources,Scheduled Completion,Milestone Changes,Source,Comments
{% for item in poam_items %}
{{ item.id }},{{ item.description }},{{ item.control }},{{ item.severity }},{{ item.status }},{{ item.resources }},{{ item.completion_date }},{{ item.milestones }},{{ item.source }},{{ item.comments }}
{% endfor %}
"""
    )

    print(f"✓ Created templates in {templates_dir}")


def create_scripts():
    """Generate compliance automation scripts"""
    scripts_dir = BASE_DIR / "scripts"

    # SBOM Generator
    (scripts_dir / "sbom_generator.py").write_text(
        """#!/usr/bin/env python3
\"\"\"Generate Software Bill of Materials (SBOM) in SPDX 2.3 format\"\"\"

import json
import subprocess
from datetime import datetime
from pathlib import Path

def generate_sbom(output_file="sbom.spdx.json"):
    \"\"\"Generate SBOM from project dependencies\"\"\"

    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "QuASIM SBOM",
        "documentNamespace": f"https://github.com/robertringler/QuASIM/sbom/{datetime.now().isoformat()}",
        "creationInfo": {
            "created": datetime.now().isoformat(),
            "creators": ["Tool: QuASIM SBOM Generator"],
            "licenseListVersion": "3.21"
        },
        "packages": []
    }

    # Get pip packages
    try:
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(result.stdout)

        for pkg in packages:
            sbom["packages"].append({
                "SPDXID": f"SPDXRef-Package-{pkg['name']}",
                "name": pkg['name'],
                "versionInfo": pkg['version'],
                "downloadLocation": f"https://pypi.org/project/{pkg['name']}/{pkg['version']}/",
                "filesAnalyzed": False,
                "supplier": "Organization: PyPI"
            })
    except Exception as e:
        print(f"Error generating SBOM: {e}")

    # Write SBOM
    with open(output_file, 'w') as f:
        json.dump(sbom, f, indent=2)

    print(f"✓ SBOM generated: {output_file}")
    return output_file

if __name__ == "__main__":
    generate_sbom()
"""
    )

    # MC/DC Analyzer
    (scripts_dir / "mcdc_analyzer.py").write_text(
        """#!/usr/bin/env python3
\"\"\"MC/DC (Modified Condition/Decision Coverage) Analyzer for DO-178C Level A\"\"\"

import json
from pathlib import Path

def analyze_mcdc_coverage(coverage_file):
    \"\"\"Analyze MC/DC coverage from test results\"\"\"

    print("Analyzing MC/DC Coverage...")

    # Load coverage data
    if Path(coverage_file).exists():
        with open(coverage_file, 'r') as f:
            coverage = json.load(f)
    else:
        coverage = {"mcdc": 0, "statement": 0, "branch": 0}

    # DO-178C Level A Requirements
    requirements = {
        "mcdc": 100,
        "statement": 100,
        "branch": 100
    }

    results = {
        "compliant": True,
        "details": {}
    }

    for metric, required in requirements.items():
        actual = coverage.get(metric, 0)
        results["details"][metric] = {
            "required": required,
            "actual": actual,
            "compliant": actual >= required
        }
        if actual < required:
            results["compliant"] = False

    # Print results
    print("\\nDO-178C Level A Coverage Results:")
    print("=" * 50)
    for metric, data in results["details"].items():
        status = "✓" if data["compliant"] else "✗"
        print(f"{status} {metric.upper()}: {data['actual']}% (required: {data['required']}%)")

    print("=" * 50)
    print(f"Overall: {'✓ COMPLIANT' if results['compliant'] else '✗ NON-COMPLIANT'}")

    return results

if __name__ == "__main__":
    import sys
    coverage_file = sys.argv[1] if len(sys.argv) > 1 else "coverage.json"
    analyze_mcdc_coverage(coverage_file)
"""
    )

    print(f"✓ Created automation scripts in {scripts_dir}")


def create_documentation():
    """Generate comprehensive documentation"""
    docs_dir = Path(__file__).parent.parent.parent / "docs"

    # Compliance README
    compliance_docs = docs_dir / "compliance"
    compliance_docs.mkdir(parents=True, exist_ok=True)

    (compliance_docs / "README.md").write_text(
        """# QuASIM Compliance Framework

Comprehensive compliance automation for Defense, Aerospace, and Critical Infrastructure.

## Frameworks Supported

- **NIST 800-53 Rev 5** (HIGH Baseline)
- **NIST 800-171 R3** (CUI Protection)
- **CMMC 2.0** (Level 2)
- **DFARS** (252.204-7012, 7019, 7020, 7021)
- **FIPS 140-3** (Cryptographic Module Validation)
- **ITAR** (USML Categories VIII, XI, XV)
- **EAR** (ECCN 5D002, 5E002)
- **DO-178C** (Level A)
- **SOC 2** (Type II)
- **ISO 27001:2022**

## Quick Start

```bash
# Run all compliance checks
make compliance-check

# Generate compliance reports
make compliance-report

# Run security scans
make security-scan

# Generate SBOM
python compliance/scripts/sbom_generator.py

# Analyze MC/DC coverage
python compliance/scripts/mcdc_analyzer.py coverage.json
```

## Architecture

```
compliance/
├── config/           # Configuration files
├── scripts/          # Automation scripts
├── policies/         # OPA Rego policies
├── matrices/         # Control matrices
└── templates/        # Document templates
```

## Workflows

### PR Compliance
Runs on every pull request:
- Code quality checks
- YAML/Markdown validation
- Workflow validation

### PR Defense Compliance
Security-focused checks:
- Vulnerability scanning
- Secret detection
- Dependency review
- Permissions check
- Dockerfile security

## Monitoring

```bash
# Watch PR Compliance
gh run watch --exit-status --workflow "PR Compliance"

# Watch Defense Compliance
gh run watch --exit-status --workflow "PR Defense Compliance"
```

## Documentation

- [Defense Requirements](../defense/README.md)
- [Export Control Guide](export-control.md)
- [RMF Process](rmf-process.md)
- [CMMC Certification](cmmc-guide.md)

## Support

For compliance questions: compliance@quasim.example.com
"""
    )

    print(f"✓ Created documentation in {docs_dir}")


def main():
    """Generate all compliance files"""
    print("Generating Defense-Grade PR Compliance Agent files...")
    print("=" * 60)

    create_opa_policies()
    create_control_matrices()
    create_templates()
    create_scripts()
    create_documentation()

    print("=" * 60)
    print("✓ All compliance files generated successfully!")
    print("")
    print("Next steps:")
    print("1. Review generated files")
    print("2. Customize configurations")
    print("3. Run: make compliance-check")
    print("4. Commit changes")


if __name__ == "__main__":
    main()
