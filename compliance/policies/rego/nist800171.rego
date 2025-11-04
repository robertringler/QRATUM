package nist800171

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
