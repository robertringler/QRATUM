# NIST Post-Quantum Cryptography Compliance

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2024-12-29 |
| NIST Standards | FIPS 203, FIPS 204, FIPS 205 |
| Security Level | Level 5 (256-bit) |

## Executive Summary

QRATUM implements NIST-standardized post-quantum cryptographic algorithms to protect against future quantum computer attacks. This document details our PQC implementation compliance with FIPS 203 (ML-KEM/Kyber), FIPS 204 (ML-DSA/Dilithium), and FIPS 205 (SLH-DSA/SPHINCS+).

## Algorithm Implementation Matrix

| Algorithm | NIST Standard | Parameter Set | Security Level | Status |
|-----------|---------------|---------------|----------------|--------|
| ML-KEM | FIPS 203 | ML-KEM-1024 (Kyber-1024) | Level 5 | ✅ Implemented |
| ML-DSA | FIPS 204 | ML-DSA-87 (Dilithium-5) | Level 5 | ✅ Implemented |
| SLH-DSA | FIPS 205 | SLH-DSA-SHA2-256s (SPHINCS+) | Level 5 | ✅ Implemented |

## ML-KEM (Kyber) Implementation

### Specification Compliance

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML-KEM-1024 Parameters                        │
├─────────────────────────────────────────────────────────────────┤
│  n (polynomial degree)          │ 256                           │
│  k (module rank)                │ 4                             │
│  q (modulus)                    │ 3329                          │
│  η₁ (noise parameter)           │ 2                             │
│  η₂ (noise parameter)           │ 2                             │
│  du (compression)               │ 11                            │
│  dv (compression)               │ 5                             │
│  Public Key Size                │ 1568 bytes                    │
│  Secret Key Size                │ 3168 bytes                    │
│  Ciphertext Size                │ 1568 bytes                    │
│  Shared Secret Size             │ 32 bytes                      │
│  Security Level                 │ NIST Level 5 (256-bit)        │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Details

**Location**: `crypto/pqc/crystals_kyber.rs`

**Key Generation**:
```rust
pub fn generate_keypair() -> Result<(PublicKey, SecretKey), KyberError>
```
- Uses DRBG with SHA3-512 for randomness
- Constant-time polynomial arithmetic
- Zeroization of intermediate values

**Encapsulation**:
```rust
pub fn encapsulate(pk: &PublicKey) -> Result<(SharedSecret, Ciphertext), KyberError>
```
- Deterministic encapsulation with provided randomness
- Implicit rejection on decapsulation failure

**Decapsulation**:
```rust
pub fn decapsulate(ct: &Ciphertext, sk: &SecretKey) -> Result<SharedSecret, KyberError>
```
- Constant-time comparison
- Returns pseudo-random value on failure (FO transform)

### Security Measures

| Measure | Implementation | NIST Requirement |
|---------|----------------|------------------|
| Constant-time operations | ✅ | Mandatory |
| Zeroization on drop | ✅ | Recommended |
| Implicit rejection | ✅ | Mandatory (FIPS 203) |
| RNG quality | DRBG (SP 800-90A) | Mandatory |

## ML-DSA (Dilithium) Implementation

### Specification Compliance

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML-DSA-87 Parameters                          │
├─────────────────────────────────────────────────────────────────┤
│  n (polynomial degree)          │ 256                           │
│  k (rows)                       │ 8                             │
│  l (columns)                    │ 7                             │
│  q (modulus)                    │ 8380417                       │
│  η (secret key range)           │ 2                             │
│  β (signature bound)            │ 120                           │
│  γ₁ (masking range)             │ 2^19                          │
│  γ₂ (low bits dropped)          │ (q-1)/32                      │
│  τ (challenge weight)           │ 60                            │
│  Public Key Size                │ 2592 bytes                    │
│  Secret Key Size                │ 4896 bytes                    │
│  Signature Size                 │ 4627 bytes                    │
│  Security Level                 │ NIST Level 5 (256-bit)        │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Details

**Location**: `crypto/pqc/crystals_dilithium.rs`

**Key Generation**:
```rust
pub fn generate_keypair() -> Result<(PublicKey, SecretKey), DilithiumError>
```
- Deterministic from seed
- Matrix expansion using SHAKE-256

**Signing**:
```rust
pub fn sign(message: &[u8], sk: &SecretKey) -> Result<Signature, DilithiumError>
pub fn sign_with_context(message: &[u8], ctx: &[u8], sk: &SecretKey) -> Result<Signature, DilithiumError>
```
- Rejection sampling for signature generation
- Context string support (FIPS 204 §5.2)
- Deterministic signing with hedged randomness

**Verification**:
```rust
pub fn verify(message: &[u8], sig: &Signature, pk: &PublicKey) -> Result<bool, DilithiumError>
pub fn verify_with_context(message: &[u8], ctx: &[u8], sig: &Signature, pk: &PublicKey) -> Result<bool, DilithiumError>
```
- Constant-time verification
- Context binding support

### Security Measures

| Measure | Implementation | NIST Requirement |
|---------|----------------|------------------|
| Deterministic signing | ✅ (hedged) | Recommended |
| Context binding | ✅ | Optional (FIPS 204) |
| Constant-time ops | ✅ | Mandatory |
| Rejection sampling | ✅ | Mandatory |
| Zeroization | ✅ | Recommended |

## SLH-DSA (SPHINCS+) Implementation

### Specification Compliance

```
┌─────────────────────────────────────────────────────────────────┐
│                SLH-DSA-SHA2-256s Parameters                      │
├─────────────────────────────────────────────────────────────────┤
│  n (security parameter)         │ 32                            │
│  h (total tree height)          │ 64                            │
│  d (layers)                     │ 8                             │
│  FORS trees                     │ 35                            │
│  FORS height                    │ 9                             │
│  WOTS+ w                        │ 16                            │
│  Public Key Size                │ 64 bytes                      │
│  Secret Key Size                │ 128 bytes                     │
│  Signature Size                 │ 17088 bytes                   │
│  Security Level                 │ NIST Level 5 (256-bit)        │
│  Stateless                      │ Yes                           │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Details

**Location**: `crypto/pqc/sphincs_plus.rs`

**Key Generation**:
```rust
pub fn generate_keypair() -> Result<(PublicKey, SecretKey), SPHINCSError>
```
- Uses getrandom for seed generation
- Derives root from SK and public seed

**Signing**:
```rust
pub fn sign(message: &[u8], sk: &SecretKey) -> Result<Signature, SPHINCSError>
```
- Randomized message hash (prevents side-channels)
- FORS + Hypertree signature

**Verification**:
```rust
pub fn verify(message: &[u8], sig: &Signature, pk: &PublicKey) -> Result<bool, SPHINCSError>
```
- Stateless verification
- No state to manage (unlike XMSS)

### Security Advantages

| Property | Benefit |
|----------|---------|
| Hash-based | Conservative assumption (no algebraic attacks) |
| Stateless | No state management issues |
| Long-term security | Suitable for long-lived signatures |
| Conservative | Most trust in hash function security |

## Algorithm Selection Guidelines

### Use ML-KEM (Kyber) For:
- Key encapsulation/exchange
- Session key establishment
- Hybrid key exchange with classical ECDH

### Use ML-DSA (Dilithium) For:
- General-purpose digital signatures
- Authentication
- TXO signing
- Validator signatures

### Use SLH-DSA (SPHINCS+) For:
- Long-term document signing
- Root certificates
- Maximum conservatism requirements
- When signature size is acceptable

## Hybrid Mode Support

QRATUM supports hybrid classical+PQC operation:

```
┌─────────────────────────────────────────────────────────────────┐
│                      HYBRID KEY EXCHANGE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Classical (X25519)     ┌──────────────┐     PQC (ML-KEM)     │
│         SS₁               │    HKDF      │          SS₂         │
│           └───────────────┤   Combine    ├───────────────┘      │
│                           └──────┬───────┘                       │
│                                  │                               │
│                                  ▼                               │
│                           ┌──────────────┐                       │
│                           │  Session Key │                       │
│                           │   (256-bit)  │                       │
│                           └──────────────┘                       │
│                                                                  │
│   Security: max(classical, PQC)                                  │
│   If either algorithm is broken, other provides protection       │
└─────────────────────────────────────────────────────────────────┘
```

## Compliance Verification

### Test Vectors

| Algorithm | Test Source | Status |
|-----------|-------------|--------|
| ML-KEM-1024 | NIST KAT vectors | ⚠️ Pending |
| ML-DSA-87 | NIST KAT vectors | ⚠️ Pending |
| SLH-DSA-256s | NIST KAT vectors | ⚠️ Pending |

### Interoperability Testing

| Test | Partner | Status |
|------|---------|--------|
| ML-KEM | liboqs | ⚠️ Planned |
| ML-DSA | liboqs | ⚠️ Planned |
| Hybrid TLS | wolfSSL | ⚠️ Planned |

## Migration Path

### Current State
- PQC implementations complete
- Classical crypto still primary
- Hybrid mode available

### Phase 1: Hybrid Default (Q1 2025)
- Enable hybrid by default
- Classical fallback for compatibility
- Monitor for issues

### Phase 2: PQC Primary (Q3 2025)
- PQC as primary algorithms
- Classical for legacy systems only
- Begin deprecation warnings

### Phase 3: PQC Only (Q1 2026)
- Remove classical-only paths
- Full PQC operation
- Harvest-now-decrypt-later mitigation complete

## Cryptographic Agility

QRATUM implements algorithm agility to support future migrations:

```rust
pub enum PQCAlgorithm {
    SPHINCSPlus,  // SLH-DSA
    Dilithium,    // ML-DSA
    Kyber,        // ML-KEM
}

impl PQCAlgorithm {
    pub fn recommended_for_signatures() -> Self;
    pub fn recommended_for_long_term_signatures() -> Self;
    pub fn recommended_for_key_exchange() -> Self;
}
```

## References

1. FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard
2. FIPS 204: Module-Lattice-Based Digital Signature Standard
3. FIPS 205: Stateless Hash-Based Digital Signature Standard
4. NIST SP 800-208: Recommendation for Stateful Hash-Based Signature Schemes
5. CISA Post-Quantum Cryptography Roadmap

---

*Document Control: v1.0 | 2024-12-29 | NIST PQC Compliance Assessment*
