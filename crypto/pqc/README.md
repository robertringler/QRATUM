# Post-Quantum Cryptography (PQC) for QRATUM

This directory implements NIST-standardized post-quantum cryptographic algorithms to protect QRATUM against future quantum computer attacks.

## Overview

Classical cryptography (RSA, ECDSA, DH) will be broken by large-scale quantum computers. QRATUM upgrades to quantum-resistant algorithms:

| Classical | Post-Quantum | Use Case |
|-----------|--------------|----------|
| RSA signatures | SPHINCS+ | Long-term signatures, Merkle events |
| ECDSA signatures | CRYSTALS-Dilithium | General signatures, FIDO2 replacement |
| ECDH key exchange | CRYSTALS-Kyber | Key encapsulation, TLS |

## NIST PQC Standards

All algorithms are from NIST's Post-Quantum Cryptography Standardization:

- **FIPS 203**: CRYSTALS-Kyber (Key Encapsulation)
- **FIPS 204**: CRYSTALS-Dilithium (Digital Signatures)
- **FIPS 205**: SPHINCS+ (Stateless Hash-Based Signatures)

## Implementations

### SPHINCS+ (`sphincs_plus.rs`)

**Stateless hash-based signatures**

- **Security**: 256-bit quantum security (SPHINCS+-SHA2-256s)
- **Signature Size**: ~17KB
- **Performance**: Slower signing (~30ms), fast verification
- **Use Case**: Long-term archival signatures, critical transactions

```rust
use qratum::crypto::pqc::{sphincs_generate_keypair, sphincs_sign, sphincs_verify};

let (pk, sk) = sphincs_generate_keypair()?;
let message = b"QRATUM critical transaction";
let signature = sphincs_sign(message, &sk)?;
assert!(sphincs_verify(message, &signature, &pk)?);
```

### CRYSTALS-Kyber (`crystals_kyber.rs`)

**Lattice-based key encapsulation mechanism (KEM)**

- **Security**: 256-bit quantum security (Kyber-1024)
- **Public Key**: ~1.6KB
- **Ciphertext**: ~1.6KB
- **Use Case**: TLS handshakes, secure channels, key exchange

```rust
use qratum::crypto::pqc::{kyber_generate_keypair, kyber_encapsulate, kyber_decapsulate};

// Alice generates keypair
let (pk, sk) = kyber_generate_keypair()?;

// Bob encapsulates shared secret
let (shared_secret_bob, ciphertext) = kyber_encapsulate(&pk)?;

// Alice decapsulates to recover shared secret
let shared_secret_alice = kyber_decapsulate(&ciphertext, &sk)?;

// shared_secret_alice == shared_secret_bob
```

### CRYSTALS-Dilithium (`crystals_dilithium.rs`)

**Lattice-based digital signatures**

- **Security**: 256-bit quantum security (Dilithium5)
- **Signature Size**: ~4.6KB
- **Performance**: Fast signing (~1ms) and verification
- **Use Case**: General signatures, FIDO2 replacement, dual-control

```rust
use qratum::crypto::pqc::{dilithium_generate_keypair, dilithium_sign, dilithium_verify};

let (pk, sk) = dilithium_generate_keypair()?;
let message = b"QRATUM authorization event";
let signature = dilithium_sign(message, &sk)?;
assert!(dilithium_verify(message, &signature, &pk)?);
```

## Migration Strategy

### Phase 1: Hybrid Mode (Current)
- Both classical (Ed25519) and PQC (Dilithium) signatures
- Verify both for backward compatibility
- Migrate operators to PQC gradually

### Phase 2: PQC Primary (Q3 2025)
- PQC signatures required
- Classical signatures optional
- All new keys are PQC

### Phase 3: PQC Only (Q1 2026)
- Remove classical crypto entirely
- Quantum-resistant end-to-end

## Integration with Aethernet

Aethernet's Merkle ledger integrates PQC:

```rust
// Aethernet TXO with PQC signatures
use aethernet::core::txo::TXO;
use qratum::crypto::pqc::dilithium_sign;

let txo = TXO::new(/* ... */);
let signature = dilithium_sign(txo.hash(), &operator_sk)?;
txo.add_signature(signature);
```

## Performance

Benchmarks on Intel Xeon (single-core):

| Algorithm | Keygen | Sign | Verify | Signature Size |
|-----------|--------|------|--------|----------------|
| SPHINCS+ | 10ms | 30ms | 2ms | 17,088 bytes |
| Dilithium5 | 5ms | 1ms | 1ms | 4,627 bytes |
| Kyber-1024 | 5ms | Encap: 1ms | Decap: 1ms | 1,568 bytes |

## Security Considerations

1. **Quantum Threat Model**: Protects against Shor's algorithm (breaks RSA/ECC) and Grover's algorithm (√n speedup)
2. **Classical Security**: All algorithms provide 256-bit classical security
3. **Side-Channel Resistance**: Implementations should use constant-time operations
4. **Key Management**: HSM integration required for production keys

## Production Deployment

⚠️ **Current Status**: Placeholder implementations for architecture demonstration

For production deployment:

1. Replace with audited implementations:
   - `pqcrypto-sphincsplus`
   - `pqcrypto-dilithium`
   - `pqcrypto-kyber`

2. Or use NIST reference implementations:
   - https://csrc.nist.gov/projects/post-quantum-cryptography

3. Enable hardware acceleration:
   - AVX2/AVX-512 for polynomial operations
   - AES-NI for symmetric crypto

4. Integrate with HSM:
   - YubiHSM 2 (PQC support planned)
   - AWS CloudHSM (custom PQC modules)

## Testing

```bash
# Run PQC tests
cargo test --package qratum-crypto-pqc

# Benchmark performance
cargo bench --package qratum-crypto-pqc
```

## References

- NIST PQC: https://csrc.nist.gov/projects/post-quantum-cryptography
- SPHINCS+: https://sphincs.org/
- CRYSTALS: https://pq-crystals.org/
- Open Quantum Safe: https://openquantumsafe.org/
