package cmmc

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
