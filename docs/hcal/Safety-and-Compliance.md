# HCAL Safety and Compliance Guidelines

## Overview

This document outlines safety procedures and compliance requirements for using HCAL (Hardware Control Abstraction Layer) in production environments.

## Safety Procedures

### Hardware Interaction

1. **Device Discovery**
   - Always use the DeviceManager API for device discovery
   - Verify device status before performing operations
   - Handle device unavailability gracefully

2. **Resource Management**
   - Respect configured resource limits
   - Monitor resource usage continuously
   - Implement timeout mechanisms for long-running operations

3. **Error Handling**
   - Never suppress hardware errors silently
   - Log all hardware interactions
   - Implement proper fallback mechanisms

### Policy Enforcement

1. **Environment-Specific Policies**
   - DEV: Relaxed policies for development and testing
   - LAB: Moderate restrictions for laboratory environments
   - PROD: Strict policies for production systems

2. **Backend Validation**
   - Validate compute backend availability before use
   - Only use backends listed in allowed_backends
   - Fail safely if requested backend is unavailable

3. **Resource Limits**
   - Enforce max_qubits limits for quantum operations
   - Respect max_circuits limits for batch operations
   - Implement timeout_seconds for all operations

## Compliance Requirements

### Security

1. **Access Control**
   - Implement role-based access control (RBAC)
   - Audit all hardware access attempts
   - Use secure credential storage

2. **Data Protection**
   - Encrypt sensitive configuration data
   - Protect policy files from unauthorized modification
   - Implement secure logging practices

3. **Network Security**
   - Use encrypted communication channels
   - Implement network segmentation
   - Monitor for suspicious activity

### Operational

1. **Change Management**
   - Document all policy changes
   - Review changes before production deployment
   - Maintain version control for policy files

2. **Monitoring and Alerting**
   - Monitor device health continuously
   - Alert on policy violations
   - Track resource utilization

3. **Incident Response**
   - Document incident response procedures
   - Maintain incident logs
   - Conduct post-incident reviews

### Regulatory

1. **Data Governance**
   - Comply with data retention policies
   - Implement data anonymization where required
   - Support data export and deletion requests

2. **Audit Trail**
   - Maintain comprehensive audit logs
   - Support audit log export
   - Retain logs per regulatory requirements

3. **Documentation**
   - Maintain up-to-date documentation
   - Document security controls
   - Provide compliance evidence

## Best Practices

### Development

1. **Testing**
   - Test with mock hardware in development
   - Use the "not requires_hardware" marker for CI tests
   - Validate policies before deployment

2. **Code Review**
   - Review all hardware interaction code
   - Validate error handling paths
   - Ensure proper resource cleanup

3. **Security Scanning**
   - Run Bandit security scans regularly
   - Address security findings promptly
   - Keep dependencies updated

### Deployment

1. **Pre-Deployment**
   - Validate policy configuration
   - Test in staging environment
   - Review deployment checklist

2. **Deployment**
   - Use automated deployment pipelines
   - Implement rollback procedures
   - Monitor deployment progress

3. **Post-Deployment**
   - Verify system functionality
   - Monitor for errors
   - Update documentation

## Emergency Procedures

### Device Failure

1. Mark device as offline in DeviceManager
2. Log incident with full context
3. Notify operations team
4. Initiate failover if available

### Policy Violation

1. Deny operation immediately
2. Log violation with details
3. Alert security team
4. Review and update policies as needed

### System Compromise

1. Isolate affected systems
2. Activate incident response team
3. Preserve evidence
4. Follow incident response plan

## Contact Information

### Support

- **Technical Support**: support@quasim.example.com
- **Security Team**: security@quasim.example.com
- **Operations**: ops@quasim.example.com

### Emergency

- **Emergency Hotline**: +1-555-QUASIM-1
- **On-Call**: Available 24/7

## Compliance Certifications

This system is designed to support compliance with:

- ISO/IEC 27001 (Information Security)
- SOC 2 Type II (Security and Availability)
- NIST Cybersecurity Framework
- Industry-specific regulations as applicable

## Version History

- v0.1.0 (2024): Initial release

## References

- HCAL Documentation: [README.md](./README.md)
- QuASIM Platform Overview: [../platform_overview.md](../platform_overview.md)
- Security Guidelines: [../../SECURITY.md](../../SECURITY.md)
# HCAL Safety & Compliance Guide

**Version:** 0.1.0  
**Last Updated:** 2025-11-05

## Overview

This document describes the safety features, compliance considerations, and best practices for the Hardware Control & Calibration Layer (HCAL).

## Threat Model

### Assets to Protect

1. **Hardware Integrity**: Prevent physical damage to CPUs, GPUs, FPGAs, and other devices
2. **System Availability**: Maintain operational systems without disruption
3. **Data Integrity**: Ensure audit logs are tamper-evident
4. **Operational Safety**: Protect personnel and facilities

### Threat Actors

1. **Accidental Misuse**: Operators making configuration errors
2. **Malicious Actors**: Intentional attempts to damage hardware or disrupt operations
3. **Software Bugs**: Defects in HCAL or dependencies causing unsafe operations
4. **Automation Failures**: Runaway calibration loops or cascading failures

### Attack Vectors

1. **Direct API Abuse**: Calling HCAL APIs with dangerous parameters
2. **Policy Bypass**: Attempting to circumvent safety policies
3. **Audit Log Tampering**: Modifying or deleting audit logs
4. **Rate Limit Exhaustion**: Overwhelming systems with requests
5. **Supply Chain**: Compromised dependencies or hardware drivers

## Security Controls

### Defense-in-Depth Layers

1. **Dry-Run by Default**
   - All operations are simulated unless explicitly enabled
   - Requires explicit `--actuate` flag in CLI or `dry_run=False` in API
   - Policy can enforce dry-run mode regardless of user preference

2. **Policy-Driven Enforcement**
   - Declarative YAML policies define allowed operations
   - Device allowlists prevent unauthorized device access
   - Backend restrictions limit which drivers can be used
   - Hardware limits enforce safe operating ranges
   - Rate limiting prevents abuse

3. **Approval Gates**
   - Production environments can require multi-party approval
   - GitHub PR approval, OIDC, or Sigstore methods supported
   - Configurable minimum number of approvers

4. **Vendor API Boundaries**
   - HCAL never crosses vendor API boundaries
   - No firmware writes or microcode loading
   - Only uses officially supported APIs (NVML, ROCm-SMI, etc.)

5. **Automatic Rollback**
   - Baseline configuration captured before changes
   - Pre/post validation with automatic rollback on failure
   - Emergency stop capability to halt all operations

6. **Audit Logging**
   - All operations logged with tamper-evident SHA256 chain
   - Immutable append-only logs
   - Cryptographic chain verification

## Operating Envelopes

### Safe Operating Limits

HCAL enforces the following default limits:

| Device Type | Parameter | Default Limit |
|------------|-----------|---------------|
| NVIDIA GPU | Power | 100W - 300W |
| NVIDIA GPU | Temperature | < 85°C |
| NVIDIA GPU | Clock | < 2000 MHz |
| CPU | Power | < 150W |
| CPU | Temperature | < 90°C |

These limits can be customized in policy files but should not exceed manufacturer specifications.

### Prohibited Operations

HCAL explicitly prohibits:

1. **Firmware Modifications**: No BIOS, UEFI, or device firmware writes
2. **Microcode Loading**: No CPU or GPU microcode updates
3. **Voltage Control**: Direct voltage manipulation (use vendor APIs only)
4. **Over-Voltage**: Exceeding manufacturer voltage specifications
5. **Over-Temperature**: Allowing devices to exceed thermal limits

## Regulatory Compliance

### Export Control

**ITAR/EAR Considerations:**

HCAL itself does not contain export-controlled technology, but:

- Hardware controlled by HCAL may be export-controlled
- Calibration data may reveal performance characteristics subject to export control
- Deployment to restricted countries may require export licenses

**Guidance:**
- Consult legal counsel before deploying to non-US entities
- Do not share calibration data or performance characteristics without authorization
- Implement access controls for sensitive configurations

### Industry Standards

**Compliance Targets:**

- **ISO/IEC 27001**: Information security management (audit logging, access control)
- **IEC 61508**: Functional safety (fail-safe defaults, validation)
- **SOC 2**: Service organization controls (audit trails, policy enforcement)

### Data Protection

- Audit logs may contain personally identifiable information (PII) such as usernames
- Implement data retention and deletion policies as required by GDPR/CCPA
- Encrypt audit logs at rest and in transit

## Best Practices

### Development Environment

```yaml
# dev-policy.yaml
environment: dev
dry_run_default: true
device_allowlist: ["*"]
approval_gate:
  required: false
```

### Lab Environment

```yaml
# lab-policy.yaml
environment: lab
dry_run_default: true
device_allowlist:
  - "gpu0"
  - "gpu1"
backend_restrictions:
  - "XilinxXrtBackend"  # No FPGA reconfig in lab
approval_gate:
  required: false
rate_limit:
  commands_per_minute: 60
```

### Production Environment

```yaml
# prod-policy.yaml
environment: prod
dry_run_default: true  # ALWAYS true in production!
device_allowlist:
  - "gpu_prod_0"
  - "gpu_prod_1"
backend_restrictions: []
device_limits:
  gpu_prod_0:
    max_power_watts: 250.0
    max_temp_celsius: 80.0
approval_gate:
  required: true
  min_approvers: 2
  methods:
    - github_pr
rate_limit:
  commands_per_minute: 30
```

### Emergency Procedures

**If Hardware Exceeds Safe Limits:**

1. Immediately call `hcal.emergency_stop()` or `quasim-hcal stop`
2. Verify hardware status via vendor tools (nvidia-smi, etc.)
3. Review audit logs to identify root cause
4. Do not resume operations until investigated

**If Audit Chain is Broken:**

1. Immediately halt all HCAL operations
2. Preserve audit log file for forensic analysis
3. Investigate potential security incident
4. Review all operations since last valid checksum

**If Policy is Bypassed:**

1. Identify how bypass occurred (code bug, config error, etc.)
2. Halt operations and fix vulnerability
3. Audit all operations that may have bypassed policy
4. Update tests to prevent regression

## Known Limitations & Risks

### Current Limitations

1. **Incomplete Backend Coverage**: Only NVIDIA backend is fully implemented
2. **Test Coverage**: ~60% (target: 90%), some edge cases may not be tested
3. **No Hardware Interlocks**: HCAL relies on vendor APIs for safety, not direct hardware interlocks
4. **Limited Validation**: Post-validation uses telemetry, which may have sampling delays

### Known Risks

1. **Vendor API Bugs**: HCAL relies on vendor APIs (NVML, etc.) which may have bugs
2. **Race Conditions**: Concurrent operations on same device may have race conditions
3. **Calibration Divergence**: Optimization algorithms may diverge if objective function is poorly defined
4. **Dependency Vulnerabilities**: HCAL depends on pyyaml, click, numpy which may have CVEs

### Risk Mitigations

1. **Dry-Run by Default**: Minimizes impact of bugs
2. **Rate Limiting**: Prevents runaway automation
3. **Audit Logging**: Enables forensic analysis
4. **Policy Enforcement**: Limits blast radius of failures
5. **Automatic Rollback**: Recovers from failed operations

## Security Disclosure

**Reporting Vulnerabilities:**

If you discover a security vulnerability in HCAL:

1. **Do not** disclose publicly
2. Email: security@quasim.example.com (replace with actual contact)
3. Include: Description, reproduction steps, potential impact
4. Expected response: 48 hours acknowledgment, 30 days for fix

## References

- NIST SP 800-53: Security and Privacy Controls
- IEC 61508: Functional Safety
- ISO/IEC 27001: Information Security Management
- NVIDIA Management Library (NVML) Documentation
- AMD ROCm System Management Interface Documentation

## Changelog

### v0.1.0 (2025-11-05)

- Initial safety and compliance documentation
- Threat model and security controls
- Operating envelopes and best practices
- Regulatory compliance guidance

---

**Disclaimer**: This document provides guidance only. Users are responsible for compliance with all applicable laws, regulations, and manufacturer specifications. Consult legal and technical experts before deploying HCAL in production environments.
