package nist80053

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
