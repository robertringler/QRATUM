# QRATUM Cryptography Design

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-01-05 |
| Classification | Internal |
| Status | Design Document |
| Domain | Cryptography (Core) |

---

## Table of Contents

1. [Court-Admissible Cryptographic Trust Chain](#1-court-admissible-cryptographic-trust-chain)
2. [Quantum Threat Model Analysis](#2-quantum-threat-model-analysis)
3. [Key Lifecycle Management](#3-key-lifecycle-management)
4. [Cryptographic Footguns in Deterministic Systems](#4-cryptographic-footguns-in-deterministic-systems)
5. [Merkle Provenance Trees](#5-merkle-provenance-trees)
6. [Hash-Based vs Lattice-Based Primitives](#6-hash-based-vs-lattice-based-primitives)
7. [Insider-Threat Cryptographic Failure Modes](#7-insider-threat-cryptographic-failure-modes)
8. [Cryptographic Rollback Protection](#8-cryptographic-rollback-protection)
9. [Forward Secrecy vs Audit Permanence](#9-forward-secrecy-vs-audit-permanence)
10. [Partial Key Compromise Red Team](#10-partial-key-compromise-red-team)

---

## 1. Court-Admissible Cryptographic Trust Chain

### 1.1 Design Objective

Design a cryptographic trust chain suitable for court-admissible evidence.

### 1.2 Legal Requirements for Digital Evidence

**Federal Rules of Evidence (FRE 901, 902):**
- Authentication: Evidence must be what proponent claims
- Chain of custody: Unbroken documentation of handling
- Integrity: No unauthorized modification
- Completeness: No material omissions

**International Standards:**
- ISO/IEC 27037: Guidelines for identification, collection, acquisition, preservation
- SWGDE Best Practices for Computer Forensics
- eIDAS (EU): Electronic signatures and seals

### 1.3 Trust Chain Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              QRATUM Court-Admissible Trust Chain                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Layer 1: ROOT OF TRUST                                                 │
│    ┌───────────────────────────────────────────────────────────────┐    │
│    │  Hardware Security Module (HSM) - FIPS 140-3 Level 3          │    │
│    │  Root CA Key: ECDSA P-384 + Dilithium-3 (hybrid)             │    │
│    │  Air-gapped ceremony, m-of-n key shares                       │    │
│    └───────────────────────────────────────────────────────────────┘    │
│                              ↓                                           │
│  Layer 2: INTERMEDIATE AUTHORITIES                                      │
│    ┌───────────────────────────────────────────────────────────────┐    │
│    │  Validator CA: Signs validator certificates                    │    │
│    │  Timestamp Authority: RFC 3161 compliant                      │    │
│    │  Audit CA: Signs audit attestations                           │    │
│    └───────────────────────────────────────────────────────────────┘    │
│                              ↓                                           │
│  Layer 3: OPERATIONAL CERTIFICATES                                      │
│    ┌───────────────────────────────────────────────────────────────┐    │
│    │  Validator Certs: Identity + signing authority                │    │
│    │  User Certs: Identity + authorization level                   │    │
│    │  Service Certs: Component authentication                      │    │
│    └───────────────────────────────────────────────────────────────┘    │
│                              ↓                                           │
│  Layer 4: EVIDENCE ARTIFACTS                                            │
│    ┌───────────────────────────────────────────────────────────────┐    │
│    │  Merkle-rooted event logs                                     │    │
│    │  Timestamped state commitments                                │    │
│    │  Multi-party signed attestations                              │    │
│    └───────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Evidence Packet Structure

```python
@dataclass(frozen=True)
class CourtAdmissibleEvidence:
    """
    Self-contained evidence packet for legal proceedings
    """
    
    # Event identification
    event_id: bytes              # SHA3-256 unique identifier
    event_type: str              # Classification (e.g., "STATE_TRANSITION")
    
    # Content
    event_content: bytes         # Actual evidence data
    content_hash: bytes          # SHA3-256(event_content)
    content_encoding: str        # MIME type or encoding
    
    # Temporal proof
    timestamp: RFC3161Timestamp  # Third-party timestamp
    logical_time: int            # Consensus logical time
    
    # Provenance chain
    previous_event_hash: bytes   # Hash chain link
    merkle_proof: MerkleProof    # Inclusion in global state
    
    # Authentication
    creator_certificate: X509Certificate
    creator_signature: bytes     # Over (content_hash || timestamp || previous_hash)
    
    # Multi-party attestation
    witness_attestations: List[WitnessAttestation]  # 2f+1 witnesses
    
    # Verification bundle
    trust_chain: List[X509Certificate]  # Full path to root
    ocsp_responses: List[OCSPResponse]  # Certificate validity at event time
    
    def verify(self, root_ca: Certificate) -> VerificationResult:
        """Complete self-verification"""
        # Verify trust chain
        if not self._verify_chain(root_ca):
            return VerificationResult.CHAIN_INVALID
        
        # Verify timestamp
        if not self._verify_timestamp():
            return VerificationResult.TIMESTAMP_INVALID
        
        # Verify creator signature
        if not self._verify_signature():
            return VerificationResult.SIGNATURE_INVALID
        
        # Verify witness attestations (2f+1 threshold)
        if not self._verify_witnesses():
            return VerificationResult.INSUFFICIENT_WITNESSES
        
        # Verify Merkle inclusion
        if not self._verify_merkle_proof():
            return VerificationResult.MERKLE_INVALID
        
        return VerificationResult.VALID
```

### 1.5 Timestamp Authority Integration

```
RFC 3161 Timestamp Protocol:

  Client                        TSA (Timestamp Authority)
    |                                    |
    | ---- TimeStampReq(hash) --------> |
    |                                    |
    |                           Sign(hash || time || serial)
    |                                    |
    | <--- TimeStampResp(token) -------- |
    |                                    |

QRATUM Integration:
  - Primary TSA: Internal (HSM-backed, GPS-synchronized)
  - Secondary TSA: External (e.g., DigiCert, Sectigo)
  - Cross-certification with legal timestamp services
```

---

## 2. Quantum Threat Model Analysis

### 2.1 Design Objective

Analyze QRATUM's cryptographic assumptions under quantum threat models.

### 2.2 Quantum Computing Timeline

| Milestone | Estimated Timeframe | Impact on QRATUM |
|-----------|-------------------|------------------|
| NISQ (1000 qubits) | 2025-2027 | No direct impact |
| Error-corrected (10K logical qubits) | 2030-2035 | ECDSA at risk |
| Cryptographically relevant (4000+ logical) | 2035-2040 | RSA/ECDSA broken |
| Harvest now, decrypt later | NOW | Long-term secrets at risk |

### 2.3 Vulnerable Primitives

| Primitive | Current Use in QRATUM | Quantum Attack | Post-Quantum Alternative |
|-----------|----------------------|----------------|-------------------------|
| ECDSA P-256 | Validator signatures | Shor's algorithm | Dilithium-3, SPHINCS+ |
| RSA-3072 | Legacy compatibility | Shor's algorithm | Kyber-768, Classic McEliece |
| ECDH P-256 | Key exchange | Shor's algorithm | Kyber-768, NTRU |
| AES-256 | Symmetric encryption | Grover (√N) | AES-256 (adequate) |
| SHA3-256 | Hashing | Grover (√N) | SHA3-256 (adequate) |

### 2.4 Quantum-Safe Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              QRATUM Quantum-Safe Cryptographic Stack                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  HYBRID SIGNATURE SCHEME                                                │
│    Signature = ECDSA(P-384) || Dilithium(Level 3)                      │
│    Verification: ACCEPT if BOTH valid                                   │
│    Rationale: Security of strongest component                           │
│                                                                          │
│  HYBRID KEY ENCAPSULATION                                               │
│    KEM = X25519 || Kyber-768                                           │
│    SharedSecret = HKDF(X25519_SS || Kyber_SS)                          │
│    Rationale: Hybrid provides quantum + classical security              │
│                                                                          │
│  HASH-BASED SIGNATURES (Long-term audit)                               │
│    Scheme: SPHINCS+-256s (stateless)                                   │
│    Use: Root CA, audit attestations requiring 50+ year validity        │
│    Trade-off: Large signatures (49KB) but conservative security        │
│                                                                          │
│  SYMMETRIC PRIMITIVES (Quantum-adequate)                               │
│    Encryption: AES-256-GCM                                             │
│    Hashing: SHA3-256 / SHAKE256                                        │
│    KDF: HKDF-SHA3-256                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Migration Strategy

**Phase 1: Hybrid Deployment (Current)**
- All new signatures use hybrid ECDSA+Dilithium
- Legacy ECDSA accepted during transition
- All key exchanges use hybrid X25519+Kyber

**Phase 2: PQC Primary (2027)**
- PQC signature mandatory for all new operations
- ECDSA deprecated for new certificates
- Existing ECDSA certificates grandfathered until expiry

**Phase 3: Classical Removal (2030)**
- All classical-only cryptography removed
- Re-sign all long-term audit data with SPHINCS+
- Complete PQC-only operation

### 2.6 Harvest-Now-Decrypt-Later Defense

```python
class HNDLDefense:
    """
    Defense against Harvest Now, Decrypt Later attacks
    """
    
    def classify_sensitivity(self, data: bytes) -> SensitivityLevel:
        """Classify data by required secrecy duration"""
        # Examples:
        # - Session keys: LOW (hours)
        # - Transaction details: MEDIUM (years)
        # - Strategic secrets: HIGH (decades)
        # - Constitutional keys: CRITICAL (permanent)
        pass
    
    def select_protection(self, level: SensitivityLevel) -> CryptoScheme:
        if level == SensitivityLevel.LOW:
            return CryptoScheme.CLASSICAL  # AES-256 sufficient
        elif level == SensitivityLevel.MEDIUM:
            return CryptoScheme.HYBRID     # AES-256 + Kyber
        elif level == SensitivityLevel.HIGH:
            return CryptoScheme.PQC_ONLY   # Kyber-1024 + AES-256
        else:  # CRITICAL
            return CryptoScheme.CONSERVATIVE  # Classic McEliece + AES-256
```

---

## 3. Key Lifecycle Management

### 3.1 Design Objective

Propose key lifecycle management under sovereign custody.

### 3.2 Key Types and Lifecycles

| Key Type | Generation | Usage Period | Rotation | Destruction |
|----------|------------|--------------|----------|-------------|
| Root CA | Air-gapped ceremony | Permanent | Never (backup only) | N/A |
| Intermediate CA | HSM ceremony | 5 years | 4 year overlap | Secure deletion |
| Validator Signing | HSM generation | 1 year | 11 month overlap | Secure deletion |
| Session Keys | On-demand | Hours | Per session | Immediate |
| Audit Sealing | Annual ceremony | 10 years | 9 year overlap | Archive only |

### 3.3 Sovereign Custody Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│              QRATUM Sovereign Key Custody                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  KEY GENERATION CEREMONY                                                │
│                                                                          │
│    Location: Air-gapped SCIF (Sensitive Compartmented Info Facility)   │
│    Personnel: 5 key custodians from different organizations             │
│    Equipment: FIPS 140-3 Level 4 HSM, Faraday cage                     │
│    Process:                                                             │
│      1. Generate entropy from 3 independent HRNGs                       │
│      2. Create key via Shamir Secret Sharing (3-of-5)                  │
│      3. Distribute shares to custodians on smart cards                 │
│      4. Zero all intermediate values                                    │
│      5. Video record ceremony (retained under seal)                    │
│                                                                          │
│  KEY STORAGE                                                            │
│                                                                          │
│    Primary: HSM in sovereign data center (jurisdiction A)              │
│    Backup 1: HSM in different sovereign data center (jurisdiction B)   │
│    Backup 2: Paper key shares in bank vault (cold storage)             │
│    Escrow: Government key escrow (jurisdiction-specific)               │
│                                                                          │
│  KEY ACCESS CONTROL                                                     │
│                                                                          │
│    Authentication: Multi-factor (smart card + PIN + biometric)         │
│    Authorization: Quorum approval (2-of-3 administrators)              │
│    Logging: All access attempts logged to WORM storage                 │
│    Rate limiting: Max 100 operations/minute                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Key Rotation Protocol

```python
class KeyRotationProtocol:
    """
    Zero-downtime key rotation with sovereignty preservation
    """
    
    def rotate_validator_key(self, validator_id: str) -> RotationResult:
        """
        Rotate validator signing key without service interruption
        """
        # Phase 1: Generate new key
        new_key = self.hsm.generate_key(
            algorithm="Dilithium-3",
            exportable=False
        )
        
        # Phase 2: Create transition certificate
        transition_cert = self.create_certificate(
            public_key=new_key.public,
            old_key_signature=self.old_key.sign(new_key.public),
            valid_from=now(),
            valid_until=now() + timedelta(days=365)
        )
        
        # Phase 3: Announce to network
        self.broadcast_key_rotation(
            validator_id=validator_id,
            new_certificate=transition_cert,
            overlap_period=timedelta(days=30)
        )
        
        # Phase 4: Begin dual-signing
        # Both old and new keys sign during overlap period
        
        # Phase 5: Revoke old key (after overlap)
        self.schedule_revocation(
            old_certificate=self.current_cert,
            revocation_date=now() + timedelta(days=30)
        )
        
        return RotationResult.SUCCESS
```

---

## 4. Cryptographic Footguns in Deterministic Systems

### 4.1 Design Objective

Identify cryptographic footguns in deterministic systems.

### 4.2 Footgun Taxonomy

| Footgun | Description | Impact | Mitigation |
|---------|-------------|--------|------------|
| **Deterministic Nonce** | Reusing nonces in ECDSA | Private key recovery | RFC 6979 deterministic nonces |
| **Predictable RNG** | Seed from deterministic state | Key prediction | Hybrid entropy sources |
| **Timing Leaks** | Execution time reveals secrets | Side-channel attack | Constant-time implementations |
| **Error Oracle** | Different errors reveal info | Padding oracle attack | Generic error messages |
| **Replay Attack** | Same signature valid twice | Double-spend | Include unique transaction ID |

### 4.3 Detailed Footgun Analysis

**4.3.1 Deterministic Nonce Catastrophe**

```
ECDSA Signature: (r, s) where k is random nonce

FOOTGUN: If k is deterministic and predictable:
  k1 = k2 (same nonce used twice)
  
  Then: private_key = (s1 - s2)^{-1} * (m1 - m2) mod n

REAL-WORLD EXAMPLE: Sony PS3 ECDSA implementation (2010)
  - Used constant k for all signatures
  - Private key extracted, PS3 security compromised

QRATUM MITIGATION:
  - Implement RFC 6979: k = HMAC-DRBG(private_key, message_hash)
  - Nonce deterministic but secret-dependent
  - Audit all ECDSA implementations for RFC 6979 compliance
```

**4.3.2 Predictable Randomness**

```python
class DeterminismRNGFootgun:
    """
    FOOTGUN: Deriving randomness from deterministic state
    """
    
    # BAD: Predictable seed
    def bad_key_generation(self, block_height: int):
        seed = sha256(str(block_height))  # Predictable!
        random.seed(seed)
        return random.randbytes(32)  # Attacker can predict
    
    # GOOD: Hybrid entropy
    def good_key_generation(self, block_height: int):
        # Combine multiple entropy sources
        entropy = b''
        entropy += self.hsm.get_random(32)      # Hardware RNG
        entropy += os.urandom(32)                # OS entropy pool
        entropy += sha256(str(block_height))     # Deterministic component
        entropy += self.tpm.get_random(32)       # TPM if available
        
        # Use HKDF to extract uniform randomness
        return hkdf_extract(entropy, salt=b'QRATUM-KEY-GEN')
```

---

## 5. Merkle Provenance Trees

### 5.1 Design Objective

Design Merkle provenance trees optimized for replayability.

### 5.2 Tree Structure

```
                        Root Hash
                           |
              ┌────────────┴────────────┐
              |                         |
         State Root                Events Root
              |                         |
     ┌────────┴────────┐      ┌────────┴────────┐
     |                 |      |                 |
  Accounts       Contracts  Epoch N-1        Epoch N
     |                 |      |                 |
  [Merkle]        [Merkle]  [Merkle]        [Merkle]
```

### 5.3 Provenance-Optimized Design

```python
@dataclass
class ProvenanceNode:
    """
    Merkle node optimized for replay verification
    """
    # Content
    data_hash: bytes           # SHA3-256 of actual data
    
    # Provenance metadata (included in hash)
    operation_type: str        # CREATE, UPDATE, DELETE
    causal_predecessor: bytes  # Hash of causing event
    logical_timestamp: int     # Lamport timestamp
    
    # Replay hints (not in hash, for efficiency)
    data_location: str         # Where to find actual data
    replay_dependencies: List[bytes]  # Other nodes needed for replay
    
    def compute_hash(self) -> bytes:
        """Deterministic hash including provenance"""
        return sha3_256(
            self.data_hash +
            self.operation_type.encode() +
            self.causal_predecessor +
            self.logical_timestamp.to_bytes(8, 'big')
        )
```

---

## 6. Hash-Based vs Lattice-Based Primitives

### 6.1 Design Objective

Compare hash-based vs lattice-based primitives for long-term archives.

### 6.2 Comparison Matrix

| Criterion | Hash-Based (SPHINCS+) | Lattice-Based (Dilithium) | Winner for Archives |
|-----------|----------------------|---------------------------|---------------------|
| **Security Assumption** | Hash function security | Lattice problems (LWE, SIS) | Hash ✓ |
| **Quantum Security** | Very high confidence | High confidence | Hash ✓ |
| **Signature Size** | ~8-50 KB | ~2.4 KB | Lattice ✓ |
| **Public Key Size** | ~32-64 bytes | ~1.3 KB | Hash ✓ |
| **Signing Speed** | Slow (~50 ms) | Fast (~1 ms) | Lattice ✓ |
| **Long-term Confidence** | Very high | High | Hash ✓ |

### 6.3 Use Case Recommendations

- **Root CA Signature (50+ year)**: SPHINCS+-256f (hash-based)
- **Audit Seal (10-20 year)**: SPHINCS+-192s (hash-based)
- **Validator Signing (1 year)**: Dilithium-3 (lattice-based)
- **Transaction Signing**: Dilithium-2 or hybrid ECDSA+Dilithium

---

## 7. Insider-Threat Cryptographic Failure Modes

### 7.1 Design Objective

Model insider-threat cryptographic failure modes.

### 7.2 Failure Mode Analysis

| Threat Actor | Access Level | Attack Vector | Mitigation |
|--------------|--------------|---------------|------------|
| Key Custodian | Key share | Collusion (m-of-n breach) | Geographic distribution |
| Developer | Source code | Subtle weakness insertion | Code review, formal verification |
| System Admin | Infrastructure | HSM interface abuse | Dual control, audit logging |
| Operator | Runtime access | Unauthorized signing | Per-signature authorization |

---

## 8. Cryptographic Rollback Protection

### 8.1 Design Objective

Propose cryptographic rollback protection.

### 8.2 Protection Mechanisms

1. **Monotonic Counters**: TPM/HSM-protected counters in every signature
2. **Hash Chain Commitment**: Each state commits to all previous states
3. **Distributed Timestamps**: Multiple independent timestamp authorities
4. **Version Binding**: Signatures include protocol version

---

## 9. Forward Secrecy vs Audit Permanence

### 9.1 Design Objective

Evaluate forward secrecy vs audit permanence tradeoffs.

### 9.2 Resolution Strategy

| Tier | Description | Forward Secrecy | Audit Retention |
|------|-------------|-----------------|-----------------|
| Ephemeral | Internal operational | Full | None |
| Auditable | Business transactions | None | Full |
| Hybrid | Sensitive but auditable | Against attackers | Via escrow |

---

## 10. Partial Key Compromise Red Team

### 10.1 Design Objective

Red-team QRATUM cryptography assuming partial key compromise.

### 10.2 Scenario Analysis

| Scenario | Compromised Asset | Blast Radius | Recovery |
|----------|------------------|--------------|----------|
| S1 | 1 of 5 root shares | None (threshold not met) | Re-key custodian |
| S2 | 1 validator key | Can impersonate 1 validator | Revoke + rotate |
| S3 | Intermediate CA | Can issue rogue certs | Revoke CA, re-issue all |

### 10.3 Defense Mechanisms

- Certificate Transparency Log
- OCSP Stapling Requirement  
- Certificate Pinning
- Short Certificate Lifetime (90 days)
- Dual CA Requirement for critical certs

---

## Appendix: Algorithm Reference

| Purpose | Algorithm | Parameters | NIST Status |
|---------|-----------|------------|-------------|
| Signing (classical) | ECDSA | P-384 | Approved |
| Signing (PQC) | Dilithium | Level 3 | Selected |
| Signing (hash-based) | SPHINCS+ | 256s | Selected |
| Key exchange | X25519 + Kyber | 768 | Selected |
| Hashing | SHA3 | 256 bits | Approved |
| Symmetric | AES-GCM | 256 bits | Approved |
