# Aethernet

**Accountable, Reversible Overlay Network Substrate for QRATUM Sovereign Intelligence**

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](../LICENSE)
[![QRATUM Platform](https://img.shields.io/badge/QRATUM-Platform-blue.svg)](https://github.com/robertringler/QRATUM)

---

## Overview

Aethernet is a deterministic, zone-aware overlay network that provides accountable and reversible transaction execution for QRATUM's sovereign AI platform. It integrates cryptographic provenance, dual-control authorization, ephemeral biometric keys, and comprehensive compliance enforcement.

**Key Features:**

- ✅ **TXO (Transaction Object)** - CBOR-primary encoding with JSON-secondary
- ✅ **RTF (Reversible Transaction Framework)** - Zone enforcement (Z0→Z3) with rollback
- ✅ **Ephemeral Biokeys** - SNP-based key derivation with 60-second TTL and auto-wipe
- ✅ **Zero-Knowledge Proofs** - Risc0/Halo2 integration for privacy
- ✅ **Merkle Ledger** - Append-only with snapshot-based rollback
- ✅ **Compliance Modules** - HIPAA and GDPR implementation
- ✅ **Legal Analysis** - JURIS-verified legal compliance

---

## Repository Structure

```
Aethernet/
├── core/
│   ├── txo/
│   │   ├── schema.yaml          # Full canonical TXO schema
│   │   └── txo.rs               # Rust structs with serde_cbor
│   ├── rtf/
│   │   ├── api.rs               # execute_txo, commit_txo, rollback_txo
│   │   └── enclave_main.rs      # no_std runtime entrypoint
│   ├── biokey/
│   │   ├── derivation.rs        # Ephemeral SNP-loci key derivation
│   │   └── zkp_verify.rs        # Risc0/Halo2 guest verification
│   ├── ledger/
│   │   └── merkle_ledger.rs     # Append-only, zone-aware, reversible
│   └── lib.rs                   # Main library file
├── integration/
│   └── vitra_e0_adapter.nf      # Nextflow hooks for TXO execution
├── compliance/
│   ├── hipaa.rs                 # HIPAA compliance module
│   └── gdpr.rs                  # GDPR compliance module
├── docs/
│   └── ARCHITECTURE.md          # Comprehensive architecture documentation
├── Cargo.toml                   # Rust package configuration
├── run_juris_legal_analysis.py # JURIS legal analysis script
└── LEGAL_ANALYSIS_REPORT.json  # Complete legal compliance report
```

---

## Quick Start

### Build

```bash
cd Aethernet

# Standard build
cargo build --release

# no_std build (for embedded/enclave)
cargo build --release --no-default-features
```

### Run Tests

```bash
# Run all tests
cargo test

# Run compliance tests
cargo test --features compliance
```

### Legal Analysis

```bash
# Run JURIS legal compliance analysis
python3 run_juris_legal_analysis.py

# View legal report
cat LEGAL_ANALYSIS_REPORT.json
```

---

## Core Components

### 1. TXO (Transaction Object)

The fundamental data structure for transactions in Aethernet.

```rust
use aethernet::txo::*;

let sender = Sender {
    identity_type: IdentityType::Operator,
    id: [1u8; 16],
    biokey_present: false,
    fido2_signed: true,
    zk_proof: None,
};

let receiver = Receiver {
    identity_type: IdentityType::Node,
    id: [2u8; 16],
};

let payload = Payload {
    payload_type: PayloadType::Genome,
    content_hash: [3u8; 32],
    encrypted: true,
};

let txo = TXO::new(
    [0u8; 16],
    sender,
    receiver,
    OperationClass::Genomic,
    payload,
);
```

### 2. RTF (Reversible Transaction Framework)

Zone-aware execution with rollback capability.

```rust
use aethernet::rtf::api::*;
use aethernet::ledger::MerkleLedger;

let ledger = MerkleLedger::new([0u8; 32]);
let mut ctx = RTFContext::new(Zone::Z1, ledger);

// Execute TXO
ctx.execute_txo(&mut txo)?;

// Commit to ledger
ctx.commit_txo(&mut txo)?;

// Rollback if needed
ctx.rollback_txo(target_epoch, "reason".to_string())?;
```

### 3. Ephemeral Biokeys

SNP-based key derivation with automatic wipe.

```rust
use aethernet::biokey::derivation::*;

let loci = [
    SNPLocus {
        chromosome: 1,
        position: 12345,
        ref_allele: b'A',
        alt_allele: b'G',
    },
];

let salt = b"operator-uuid-12345";
let biokey = EphemeralBiokey::derive(&loci, salt, 60);

// Use key (once only)
let key_material = biokey.get_key_material();

// Auto-wiped on Drop
```

### 4. Zone Topology

```
Z0 (Genesis) ────auto────> Z1 (Staging) ────sig A + validation────> Z2 (Production) ────sig A+B + airgap────> Z3 (Archive)
   Immutable                 Rollback OK                               Emergency rollback                          Immutable
   No signatures             No signatures                             Single signature                            Dual signatures
```

---

## Legal Compliance

### JURIS Analysis Summary

**Overall Assessment:**

- ✅ **Legal Risk:** MEDIUM
- ✅ **Technical Strength:** HIGH
- ✅ **Compliance Readiness:** HIGH (HIPAA: 5/5, GDPR: 6/6)
- ✅ **Privacy Protection:** HIGH
- ⚠️  **Recommendation:** PROCEED WITH ATTORNEY REVIEW

**Key Findings:**

1. **Contract Structure:** Well-defined technical controls, medium risk level
2. **HIPAA Compliance:** Full implementation of administrative, physical, and technical safeguards
3. **GDPR Compliance:** Comprehensive data subject rights, encryption, breach notification
4. **Privacy Analysis:** Strong privacy-by-design with ephemeral biokeys and ZK proofs
5. **Litigation Risk:** LOW-MEDIUM with strong defensive position

**Critical Actions Required:**

- ⚠️  Implement explicit consent mechanism for genetic data (GDPR Article 9)
- ⚠️  Add dispute resolution clause to TXO schema
- ⚠️  Document emergency rollback procedures
- ⚠️  Conduct formal DPIA for genetic data processing

**Recommended:**

- 📋 Independent security audit of ZK proof implementation
- 📋 Quarterly compliance audits and monitoring
- 📋 Attorney review before production deployment
- 📋 Consider patent filing for biokey derivation method

See [LEGAL_ANALYSIS_REPORT.json](./LEGAL_ANALYSIS_REPORT.json) for complete analysis.

---

## Integration

### VITRA-E0 Genomics Pipeline

Aethernet integrates with VITRA-E0 deterministic WGS pipeline via Nextflow adapter:

```bash
nextflow run integration/vitra_e0_adapter.nf \
  --zone Z2 \
  --operation_class GENOMIC \
  --dual_control true \
  --enable_biokey true
```

---

## Security Invariants

1. **Determinism:** Same input TXO → same output state
2. **Dual Control:** Critical operations require two independent authorizations
3. **Sovereignty:** No external dependencies or data egress
4. **Reversibility:** Zone-appropriate rollback capability
5. **Auditability:** Complete provenance chain with Merkle verification

---

## Cryptographic Primitives

| Purpose | Algorithm | Key Size |
|---------|-----------|----------|
| Hashing | SHA3-256 | 256 bits |
| Signatures | Ed25519 | 256 bits |
| Encryption (rest) | AES-256-GCM | 256 bits |
| Encryption (transit) | TLS 1.3 | 256 bits |

---

## License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Support

- **Issues:** <https://github.com/robertringler/QRATUM/issues>
- **Discussions:** <https://github.com/robertringler/QRATUM/discussions>
- **Email:** <info@qratum.ai>

---

## Citation

```bibtex
@software{aethernet_2024,
  title = {Aethernet: Accountable, Reversible Overlay Network Substrate},
  author = {QRATUM Platform},
  year = {2024},
  url = {https://github.com/robertringler/QRATUM/tree/main/Aethernet},
  note = {Part of the QRATUM-ASI platform}
}
```

---

## Legal Disclaimer

⚖️  This software is provided "as is" without warranty. Organizations implementing Aethernet should:

1. Consult qualified legal counsel in relevant jurisdictions
2. Conduct jurisdiction-specific compliance assessments
3. Obtain appropriate legal opinions before deployment
4. Ensure ongoing legal and compliance oversight

This does not constitute legal advice. Consult a licensed attorney for legal matters.

---

**Built with 💚 by the QRATUM Team**

*Sovereign. Deterministic. Auditable.*
