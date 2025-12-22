# QRATUM Platform API Documentation

This document provides comprehensive documentation for the QRATUM Platform API, including authentication flows, security controls, and compliance information.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Security Architecture](#security-architecture)
4. [Compliance](#compliance)
5. [API Endpoints](#api-endpoints)
6. [WebSocket Protocol](#websocket-protocol)

## Overview

The QRATUM Platform API provides a RESTful interface for:

- **Job Submission**: Submit quantum simulations, tensor analysis, VQE, QAOA, and other computational jobs
- **Status Monitoring**: Track job progress in real-time via HTTP polling or WebSocket
- **Results Retrieval**: Download simulation results and artifacts
- **Resource Management**: Monitor cluster utilization and manage quotas

### Base URL

| Environment | URL |
|-------------|-----|
| Production | `https://api.qratum.io/v1` |
| Staging | `https://staging-api.qratum.io/v1` |
| Development | `http://localhost:8000/v1` |

## Authentication

The API supports multiple authentication methods to accommodate different use cases.

### OAuth2 Client Credentials Flow

Recommended for server-to-server communication and automated systems.

```
┌──────────┐                              ┌──────────────┐
│  Client  │                              │  QRATUM API  │
└────┬─────┘                              └──────┬───────┘
     │                                           │
     │  1. POST /v1/auth/token                   │
     │     grant_type=client_credentials         │
     │     client_id=<client_id>                 │
     │     client_secret=<client_secret>         │
     │ ─────────────────────────────────────────>│
     │                                           │
     │  2. Token Response                        │
     │     access_token=<jwt>                    │
     │     refresh_token=<refresh_jwt>           │
     │     expires_in=3600                       │
     │ <─────────────────────────────────────────│
     │                                           │
     │  3. API Request                           │
     │     Authorization: Bearer <access_token>  │
     │ ─────────────────────────────────────────>│
     │                                           │
     │  4. API Response                          │
     │ <─────────────────────────────────────────│
     │                                           │
```

#### Token Request

```http
POST /v1/auth/token HTTP/1.1
Host: api.qratum.io
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&client_id=your-client-id&client_secret=your-client-secret&scope=read:jobs%20write:jobs
```

#### Token Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "scope": "read:jobs write:jobs read:resources"
}
```

### API Key Authentication

For simpler service-to-service authentication without token management.

```http
GET /v1/jobs HTTP/1.1
Host: api.qratum.io
X-API-Key: qratum_your_api_key_here
```

### Token Refresh Flow

```
┌──────────┐                              ┌──────────────┐
│  Client  │                              │  QRATUM API  │
└────┬─────┘                              └──────┬───────┘
     │                                           │
     │  1. POST /v1/auth/refresh                 │
     │     refresh_token=<refresh_token>         │
     │ ─────────────────────────────────────────>│
     │                                           │
     │  2. New Token Response                    │
     │     access_token=<new_jwt>                │
     │     refresh_token=<new_refresh_jwt>       │
     │ <─────────────────────────────────────────│
     │                                           │
```

### Token Revocation

```http
POST /v1/auth/revoke HTTP/1.1
Host: api.qratum.io
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type_hint": "access_token"
}
```

### Scopes

| Scope | Description |
|-------|-------------|
| `read:jobs` | Read job information and results |
| `write:jobs` | Submit and manage jobs |
| `read:resources` | View resource allocation and quotas |
| `admin` | Administrative access (internal use) |

## Security Architecture

### HashiCorp Vault Integration

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  QRATUM API  │────>│    Vault    │────>│  Secrets     │
│              │     │             │     │  - JWT Keys  │
│              │     │             │     │  - API Keys  │
│              │     │             │     │  - DB Creds  │
└──────────────┘     └─────────────┘     └──────────────┘
```

- **JWT Signing Keys**: Stored in Vault and rotated every 24 hours
- **API Key Validation**: Keys validated against Vault-stored hashes
- **Secret Rotation**: Automatic rotation with zero-downtime

### Request Flow with Security Controls

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Client  │───>│ Rate Limit  │───>│   Auth       │───>│  Endpoint   │
│          │    │ Middleware  │    │  Middleware  │    │  Handler    │
└──────────┘    └─────────────┘    └──────────────┘    └─────────────┘
                      │                   │                   │
                      v                   v                   v
                ┌───────────┐       ┌───────────┐       ┌───────────┐
                │ Audit Log │       │ Audit Log │       │ Audit Log │
                └───────────┘       └───────────┘       └───────────┘
```

### Security Headers

All responses include security headers:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Content-Type-Options` | `nosniff` | Prevent MIME sniffing |
| `X-Frame-Options` | `DENY` | Prevent clickjacking |
| `X-XSS-Protection` | `1; mode=block` | XSS protection |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | HTTPS enforcement |
| `Cache-Control` | `no-store` | Prevent caching of sensitive data |
| `X-Request-ID` | `<uuid>` | Request correlation |

### Rate Limiting

Token bucket algorithm with configurable limits:

| Tier | Requests/Minute | Burst Size |
|------|-----------------|------------|
| Free | 60 | 10 |
| Standard | 300 | 50 |
| Enterprise | 1000 | 200 |

Rate limit headers in responses:

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 298
X-RateLimit-Reset: 1702800000
Retry-After: 60  (when rate limited)
```

## Compliance

### DO-178C Level A Certification

The QRATUM platform maintains DO-178C Level A certification for aerospace applications:

- **100% MC/DC coverage** for safety-critical code paths
- **Deterministic execution** with <1μs seed replay drift tolerance
- **Comprehensive audit trails** for all operations
- **Formal verification** of core algorithms

### NIST 800-53 Security Controls

| Control | Implementation |
|---------|---------------|
| **AC-2** Account Management | User/service accounts managed via OAuth2/OIDC |
| **AC-3** Access Enforcement | Scope-based authorization on all endpoints |
| **AU-2** Audit Events | All API calls logged with timestamps |
| **AU-3** Content of Audit Records | User ID, action, timestamp, result logged |
| **AU-8** Time Stamps | UTC timestamps on all audit records |
| **IA-2** Identification | Unique user/client identification required |
| **IA-5** Authenticator Management | JWT tokens with expiration, HMAC signing |
| **SC-5** Denial of Service Protection | Rate limiting, request validation |
| **SC-8** Transmission Confidentiality | TLS 1.3 required for all connections |
| **SC-13** Cryptographic Protection | SHA-256 for signing, AES-256 for encryption |

### CMMC 2.0 Level 2

Defense contractor compliance maintained:

- **IA.2.076**: Unique identification for all users
- **IA.2.078**: Replay-resistant authentication
- **IA.2.079**: Identifier management
- **IA.2.081**: Cryptographic authentication

### Audit Log Format

All operations are logged in structured format:

```json
{
  "timestamp": "2025-12-17T07:00:00.000Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/v1/jobs",
  "status_code": 201,
  "duration_ms": 45.23,
  "client_ip": "192.168.1.100",
  "user_id": "user-123",
  "action": "job_submit",
  "resource": "job:abc-123"
}
```

## API Endpoints

### Health Checks

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | None | Health check |
| `/readiness` | GET | None | Kubernetes readiness probe |

### Authentication

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/auth/token` | POST | None | Get access token |
| `/v1/auth/refresh` | POST | None | Refresh access token |
| `/v1/auth/revoke` | POST | Bearer | Revoke token |

### Jobs

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/jobs` | GET | Bearer/API Key | List jobs |
| `/v1/jobs` | POST | Bearer/API Key | Submit job |
| `/v1/jobs/{job_id}` | GET | Bearer/API Key | Get job details |
| `/v1/jobs/{job_id}` | DELETE | Bearer/API Key | Cancel job |
| `/v1/jobs/{job_id}/status` | GET | Bearer/API Key | Get job status |
| `/v1/jobs/{job_id}/results` | GET | Bearer/API Key | Get job results |
| `/v1/jobs/{job_id}/artifacts` | GET | Bearer/API Key | List artifacts |
| `/v1/jobs/{job_id}/artifacts/{artifact_id}` | GET | Bearer/API Key | Download artifact |
| `/v1/validate` | POST | Bearer/API Key | Validate job config |

### Resources

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/resources` | GET | Bearer/API Key | Get resource dashboard |
| `/v1/resources/clusters` | GET | Bearer/API Key | List clusters |
| `/v1/resources/quotas` | GET | Bearer/API Key | Get quotas |

## WebSocket Protocol

### Connection

Connect to the WebSocket endpoint with your access token:

```
wss://api.qratum.io/v1/ws/status?token=<access_token>
```

### Message Format

All messages are JSON-encoded.

#### Subscribe to Job Updates

```json
{
  "action": "subscribe",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Unsubscribe

```json
{
  "action": "unsubscribe",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Status Update (Server → Client)

```json
{
  "type": "status",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.45,
  "message": "Iteration 45/100"
}
```

#### Ping/Pong

```json
{"action": "ping"}
{"type": "pong"}
```

### Connection Lifecycle

```
┌──────────┐                              ┌──────────────┐
│  Client  │                              │  QRATUM API  │
└────┬─────┘                              └──────┬───────┘
     │                                           │
     │  1. WebSocket Connect                     │
     │     wss://api.qratum.io/v1/ws/status     │
     │     ?token=<access_token>                 │
     │ ────────────────────────────────────────> │
     │                                           │
     │  2. Connection Accepted                   │
     │ <──────────────────────────────────────── │
     │                                           │
     │  3. Subscribe                             │
     │     {"action":"subscribe","job_id":"..."}│
     │ ────────────────────────────────────────> │
     │                                           │
     │  4. Current Status                        │
     │     {"type":"status","status":"running"}  │
     │ <──────────────────────────────────────── │
     │                                           │
     │  5. Status Updates (periodic)             │
     │     {"type":"status","progress":0.5}      │
     │ <──────────────────────────────────────── │
     │                                           │
     │  6. Job Complete                          │
     │     {"type":"status","status":"completed"}│
     │ <──────────────────────────────────────── │
     │                                           │
     │  7. Unsubscribe                           │
     │     {"action":"unsubscribe","job_id":"..."}│
     │ ────────────────────────────────────────> │
     │                                           │
```

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "invalid_request",
  "error_description": "The request was invalid",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Malformed request |
| `unauthorized` | 401 | Authentication required |
| `invalid_token` | 401 | Token invalid or expired |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limit_exceeded` | 429 | Too many requests |
| `internal_error` | 500 | Server error |

## SDK Support

Official SDKs are available for:

- **TypeScript/JavaScript**: `@qratum/sdk` - [Documentation](../../sdk/typescript/README.md)
- **Python**: Coming soon

## Support

For API support, please contact:

- **Documentation**: <https://docs.qratum.io>
- **Issues**: <https://github.com/robertringler/QRATUM/issues>
- **Security**: <security@qratum.io>
