# QRATUM Key Lifecycle Diagram

## Overview

This document describes the cryptographic key lifecycle in QRATUM, covering key generation, distribution, usage, rotation, and destruction.

## Key Types

### 1. Biokey (Ephemeral Session Key)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BIOKEY LIFECYCLE                                 │
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │   ENTROPY    │───▶│  DERIVATION  │───▶│   ACTIVE     │              │
│   │  COLLECTION  │    │              │    │   (≤30s)     │              │
│   └──────────────┘    └──────────────┘    └──────────────┘              │
│          │                   │                   │                       │
│          ▼                   ▼                   ▼                       │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  • Genomic   │    │  • Blending  │    │  • Encrypt   │              │
│   │  • TRNG      │    │  • Projection│    │  • Sign      │              │
│   │  • Device FP │    │  • SHA3-512  │    │  • Derive    │              │
│   │  • System    │    │              │    │              │              │
│   └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                   │                      │
│                                      ┌────────────┴────────────┐        │
│                                      ▼                         ▼        │
│                              ┌──────────────┐         ┌──────────────┐  │
│                              │   ROTATION   │         │  EXPIRATION  │  │
│                              │   (Epoch+1)  │         │   (>30s)     │  │
│                              └──────────────┘         └──────────────┘  │
│                                      │                         │        │
│                                      │                         ▼        │
│                                      │                ┌──────────────┐  │
│                                      │                │  ZEROIZATION │  │
│                                      └───────────────▶│              │  │
│                                                       └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

States:
  • ENTROPY COLLECTION: Gathering high-entropy input from multiple sources
  • DERIVATION: Blending and projecting entropy through SHA3-512
  • ACTIVE: Key available for cryptographic operations (max 30 seconds)
  • ROTATION: Creating new epoch key from current key
  • EXPIRATION: Key exceeded maximum lifetime
  • ZEROIZATION: Secure erasure of all key material
```

### 2. PQC Keys (Long-term Identity Keys)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PQC KEY LIFECYCLE (Kyber/Dilithium)                  │
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  GENERATION  │───▶│   STORAGE    │───▶│    USAGE     │              │
│   │              │    │              │    │              │              │
│   └──────────────┘    └──────────────┘    └──────────────┘              │
│          │                   │                   │                       │
│          ▼                   ▼                   ▼                       │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  • DRBG Seed │    │  • HSM/TEE   │    │  Kyber:      │              │
│   │  • NIST      │    │  • Encrypted │    │  • Encaps    │              │
│   │    Parameters│    │  • Backed Up │    │  • Decaps    │              │
│   │  • Level 5   │    │              │    │  Dilithium:  │              │
│   └──────────────┘    └──────────────┘    │  • Sign      │              │
│                                           │  • Verify    │              │
│                                           └──────────────┘              │
│                                                   │                      │
│                              ┌────────────────────┴──────────────┐      │
│                              ▼                                   ▼      │
│                      ┌──────────────┐                   ┌──────────────┐│
│                      │   ROTATION   │                   │  REVOCATION  ││
│                      │  (Annual or  │                   │  (Compromise)││
│                      │  on Policy)  │                   │              ││
│                      └──────────────┘                   └──────────────┘│
│                              │                                   │      │
│                              └───────────────┬───────────────────┘      │
│                                              ▼                          │
│                                      ┌──────────────┐                   │
│                                      │   ARCHIVAL   │                   │
│                                      │  (Decrypt    │                   │
│                                      │   old data)  │                   │
│                                      └──────────────┘                   │
│                                              │                          │
│                                              ▼                          │
│                                      ┌──────────────┐                   │
│                                      │ DESTRUCTION  │                   │
│                                      │ (After       │                   │
│                                      │  retention)  │                   │
│                                      └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. DRBG State Keys

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DRBG STATE LIFECYCLE                             │
│                                                                          │
│   ┌──────────────┐                                                       │
│   │ INSTANTIATION│                                                       │
│   │              │                                                       │
│   │  • Entropy   │                                                       │
│   │  • Nonce     │                                                       │
│   │  • Personal  │                                                       │
│   └──────┬───────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│   │    ACTIVE    │────▶│   GENERATE   │────▶│   RESEED     │            │
│   │              │◀────│              │◀────│              │            │
│   └──────────────┘     └──────────────┘     └──────────────┘            │
│          │                                          │                    │
│          │              Request                     │ Every 2^48        │
│          │              Counter++                   │ requests or       │
│          │                                          │ on demand         │
│          │                                          │                    │
│          │         ┌───────────────────────────────┘                    │
│          │         │                                                     │
│          ▼         ▼                                                     │
│   ┌──────────────────────┐                                              │
│   │    UNINSTANTIATE     │                                              │
│   │                      │                                              │
│   │  • Zeroize K         │                                              │
│   │  • Zeroize V         │                                              │
│   │  • Clear counters    │                                              │
│   └──────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4. HKDF-Derived Keys

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      HKDF DERIVATION LIFECYCLE                           │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                       INPUT KEY MATERIAL (IKM)                    │  │
│   │                                                                    │  │
│   │  Sources: Biokey | PQC Shared Secret | Pre-shared Key | DH       │  │
│   └─────────────────────────────┬────────────────────────────────────┘  │
│                                 │                                        │
│                                 ▼                                        │
│                   ┌──────────────────────────┐                          │
│                   │        EXTRACT           │                          │
│                   │                          │                          │
│                   │  PRK = HMAC(salt, IKM)   │                          │
│                   │                          │                          │
│                   └────────────┬─────────────┘                          │
│                                │                                        │
│              ┌─────────────────┼─────────────────┐                      │
│              │                 │                 │                      │
│              ▼                 ▼                 ▼                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │    EXPAND    │  │    EXPAND    │  │    EXPAND    │                 │
│   │   (enc key)  │  │   (mac key)  │  │     (IV)     │                 │
│   │              │  │              │  │              │                 │
│   │ info="enc"   │  │ info="mac"   │  │ info="iv"    │                 │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
│          │                 │                 │                          │
│          ▼                 ▼                 ▼                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │   AES-256    │  │  HMAC-SHA3   │  │   Nonce/IV   │                 │
│   │    Key       │  │    Key       │  │              │                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                 │
│          │                 │                 │                          │
│          └─────────────────┼─────────────────┘                          │
│                            │                                            │
│                            ▼                                            │
│                   ┌──────────────────┐                                  │
│                   │    SESSION USE   │                                  │
│                   │                  │                                  │
│                   │  Encrypt/Decrypt │                                  │
│                   │  MAC/Verify      │                                  │
│                   └────────┬─────────┘                                  │
│                            │                                            │
│                            ▼                                            │
│                   ┌──────────────────┐                                  │
│                   │   ZEROIZATION    │                                  │
│                   │                  │                                  │
│                   │  On session end  │                                  │
│                   │  or key rotation │                                  │
│                   └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key State Transitions

### Biokey State Machine

```
            ┌─────────────────────────────────────────────────┐
            │                                                 │
            ▼                                                 │
     ┌──────────────┐                                         │
     │  COLLECTING  │ ──── add_entropy() ────┐                │
     └──────────────┘                        │                │
            │                                │                │
            │ finalize()                     │                │
            ▼                                ▼                │
     ┌──────────────┐                 ┌──────────────┐        │
     │   DERIVING   │                 │  COLLECTING  │        │
     └──────────────┘                 └──────────────┘        │
            │                                                 │
            │ derive_blended()                                │
            ▼                                                 │
     ┌──────────────┐                                         │
     │    VALID     │ ────── <30s ─────────────────┐          │
     └──────────────┘                              │          │
            │                                      │          │
     ┌──────┴───────┐                              │          │
     │              │                              │          │
     ▼              ▼                              ▼          │
┌──────────┐  ┌──────────────┐            ┌──────────────┐    │
│ ROTATED  │  │ INVALIDATED  │            │   EXPIRED    │    │
│          │  │   (manual)   │            │   (>30s)     │    │
└──────────┘  └──────────────┘            └──────────────┘    │
     │              │                              │          │
     │              └──────────────┬───────────────┘          │
     │                             │                          │
     │                             ▼                          │
     │                      ┌──────────────┐                  │
     │                      │   ZEROIZED   │                  │
     │                      └──────────────┘                  │
     │                                                        │
     └────────────────────────────────────────────────────────┘
                           (new epoch)
```

## Security Controls by Phase

| Phase | Controls |
|-------|----------|
| Generation | DRBG with entropy pooling, min 256-bit security |
| Storage | TEE/HSM, encrypted at rest, access logging |
| Distribution | PQC key encapsulation, authenticated channels |
| Usage | Constant-time ops, operation counting, lifetime checks |
| Rotation | Epoch-based, forward secrecy, old key zeroization |
| Revocation | Immediate zeroization, revocation list update |
| Destruction | Zeroization, memory fencing, verification |

## Audit Events

All key lifecycle events generate audit records:

```json
{
  "event_type": "KEY_LIFECYCLE",
  "key_type": "BIOKEY|PQC|DRBG|DERIVED",
  "operation": "GENERATE|ACTIVATE|USE|ROTATE|REVOKE|DESTROY",
  "timestamp": "2024-12-29T00:00:00Z",
  "key_id_hash": "sha3_256(key_id)",
  "actor": "system|user_id",
  "success": true,
  "details": {
    "algorithm": "SHA3-512|KYBER-1024|DILITHIUM-5",
    "security_level": 256,
    "lifetime_remaining_ms": 15000
  }
}
```
