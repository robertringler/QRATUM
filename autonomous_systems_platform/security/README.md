# Security Overview

* **TLS**: Managed by cert-manager with Let's Encrypt issuer
* **Network Policies**: Restrict pod-to-pod communication
* **IAM**: Least-privilege service account roles
* **OAuth2**: JWT validation for API endpoints (see oauth2.md)
