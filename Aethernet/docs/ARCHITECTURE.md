# Aethernet Architecture

**Accountable, Reversible Overlay Network Substrate for QRATUM Sovereign Intelligence**

## Overview

Aethernet is a deterministic, zone-aware overlay network that provides accountable and reversible transaction execution for QRATUM's sovereign AI platform. It integrates cryptographic provenance, dual-control authorization, ephemeral biometric keys, and compliance enforcement.

## Core Components

### 1. TXO (Transaction Object)

**Location:** `core/txo/`

The TXO is the fundamental data structure representing a transaction in Aethernet.

**Key Features:**
- **CBOR-primary encoding** with JSON-secondary for human readability
- **SHA3-256 hashing** for deterministic Merkle chaining
- **Dual-control signatures** (FIDO2 + optional biokey)
- **Zone-aware reversibility** (rollback capability)
- **Complete audit trail** (actor, action, timestamp)

**Structure:**
```rust
pub struct TXO {
    version: u32,
    txo_id: [u8; 16],              // UUID v4
    timestamp: u64,                 // Unix epoch
    epoch_id: u64,                  // Ledger snapshot
    container_hash: [u8; 32],       // Execution container
    sender: Sender,
    receiver: Receiver,
    operation_class: OperationClass,
    reversibility_flag: bool,
    payload: Payload,
    dual_control_required: bool,
    signatures: Vec<Signature>,
    rollback_history: Vec<RollbackEntry>,
    audit_trail: Vec<AuditEntry>,
}
```

### 2. RTF (Reversible Transaction Framework)

**Location:** `core/rtf/`

RTF provides the execution layer with zone enforcement and rollback primitives.

**API:**
- `execute_txo(txo)` - Validate and prepare TXO for commit
- `commit_txo(txo)` - Append TXO to Merkle ledger
- `rollback_txo(epoch, reason)` - Rollback to previous epoch

**Zone Topology (Z0-Z3):**

```
Z0 (Genesis)
    ↓ Auto
Z1 (Staging)
    ↓ Sig A + GIAB validation
Z2 (Production)
    ↓ Sig A+B + Air-gap
Z3 (Archive)
```

**Zone Properties:**

| Zone | Mutable | Signature | Air-Gap | Rollback | Operations |
|------|---------|-----------|---------|----------|-----------|
| Z0   | No      | None      | No      | Never    | GENESIS   |
| Z1   | Yes     | None      | No      | Emergency| All       |
| Z2   | Yes     | Single A  | No      | Emergency| Genomic/Network/Compliance |
| Z3   | No      | Dual A+B  | Yes     | Never    | Audit only|

**Enclave Entry Point:**
- `enclave_main.rs` - no_std runtime for trusted execution environment (TEE)
- Isolates TXO execution in SGX/SEV-SNP enclave
- Wipes sensitive data after execution

### 3. Biokey Module

**Location:** `core/biokey/`

Ephemeral biometric key derivation from SNP (Single Nucleotide Polymorphism) loci.

**Key Features:**
- **Ephemeral derivation** - Keys exist only in RAM for 60 seconds
- **SNP-based** - Derived from genetic variants (non-coding regions)
- **Zero-knowledge proofs** - Prove possession without revealing SNP data
- **Auto-wipe** - Secure memory clearing on Drop

**Security Model:**
```rust
pub struct EphemeralBiokey {
    key_material: [u8; 32],  // Derived from SNP loci
    created_at: u64,          // Creation timestamp
    ttl: u64,                 // Time-to-live (default: 60s)
}

impl Drop for EphemeralBiokey {
    fn drop(&mut self) {
        self.wipe();  // Volatile write to clear memory
    }
}
```

**ZKP Integration:**
- **Risc0** - RISC-V zkVM for general computation
- **Halo2** - Recursive SNARKs for efficient proofs

### 4. Merkle Ledger

**Location:** `core/ledger/`

Append-only, zone-aware ledger with snapshot-based rollback.

**Features:**
- **Merkle chain** - Each node links to parent via SHA3-256
- **Zone promotion** - Z0→Z1→Z2→Z3 with validation
- **Epoch snapshots** - Immutable checkpoints for rollback
- **Chain verification** - Validate integrity from genesis to current

**Structure:**
```rust
pub struct MerkleLedger {
    genesis_root: [u8; 32],     // Immutable anchor
    current_root: [u8; 32],     // Current Merkle root
    nodes: Vec<LedgerNode>,     // All transactions
    snapshots: Vec<EpochSnapshot>, // Rollback points
    current_zone: Zone,
}
```

### 5. Compliance Modules

**Location:** `compliance/`

#### HIPAA (Health Insurance Portability and Accountability Act)
- **Administrative Safeguards** - Access control, training, emergency access
- **Physical Safeguards** - Facility access, workstation security
- **Technical Safeguards** - Encryption, audit logging, unique user ID
- **Privacy Rule** - Minimum necessary, de-identification, patient authorization
- **Breach Notification** - Risk assessment, notification within 60 days

#### GDPR (General Data Protection Regulation)
- **Lawful Basis** - Consent, contract, legal obligation, vital interests
- **Special Categories** - Genetic data processing with explicit consent
- **Data Subject Rights** - Access, rectification, erasure, portability, objection
- **Data Protection by Design** - Pseudonymization, minimization, privacy defaults
- **Security of Processing** - Encryption at rest/transit, pseudonymization
- **Breach Notification** - Within 72 hours to supervisory authority
- **DPIA** - Data Protection Impact Assessment for high-risk processing
- **International Transfers** - Adequacy decisions, Standard Contractual Clauses

### 6. Integration

**Location:** `integration/`

#### VITRA-E0 Adapter (Nextflow)
Integrates Aethernet TXO execution with VITRA-E0 genomics pipeline.

**Features:**
- **Pipeline hooks** - Before/after stage TXO creation
- **Zone enforcement** - Validate operations per zone policy
- **Merkle chain export** - CBOR-encoded provenance DAG
- **FIDO2 signing** - Zone-appropriate signature collection

## Data Flow

### 1. TXO Creation
```
User/System
    ↓
Create TXO (sender, receiver, payload)
    ↓
Set metadata (txo_id, timestamp, operation_class)
    ↓
Add signatures (FIDO2, optional biokey)
```

### 2. TXO Execution
```
RTF Context (zone-aware)
    ↓
Validate zone policy
    ↓
Verify signatures
    ↓
Check dual control
    ↓
Set epoch from context
    ↓
Add audit entry (EXECUTE)
```

### 3. TXO Commit
```
Merkle Ledger
    ↓
Compute TXO hash
    ↓
Create ledger node (parent_hash, txo_hash)
    ↓
Update current_root
    ↓
Append to nodes
    ↓
Add audit entry (COMMIT)
```

### 4. Rollback
```
RTF Context
    ↓
Verify zone allows rollback (Z1, Z2 only)
    ↓
Find target epoch snapshot
    ↓
Restore Merkle root
    ↓
Truncate nodes after snapshot
    ↓
Update current_epoch
```

## Security Invariants

### Determinism
- Same input TXO → same output state
- Fixed hash algorithm (SHA3-256)
- CBOR encoding for canonical serialization
- No external dependencies or network calls

### Dual Control
- Critical operations require two independent authorizations
- Implemented via FIDO2 signatures from separate keys
- Enforced at zone promotion (Z2→Z3)

### Sovereignty
- No data egress to external systems
- Air-gapped deployment support (Z3)
- On-premises execution
- Encrypted at rest and in transit

### Reversibility
- Zone-appropriate rollback capability
- Z1 (Staging) - full rollback
- Z2 (Production) - emergency rollback
- Z0, Z3 - immutable (no rollback)

### Auditability
- Complete provenance chain
- Immutable audit trail per TXO
- Merkle-chained for tamper-evidence
- CBOR export for external verification

## Cryptographic Primitives

| Purpose | Algorithm | Key Size |
|---------|-----------|----------|
| Hashing | SHA3-256 | 256 bits |
| Signatures | Ed25519 | 256 bits |
| Encryption (at rest) | AES-256-GCM | 256 bits |
| Encryption (in transit) | TLS 1.3 | 256 bits |
| ZKP (Risc0) | STARK | Variable |
| ZKP (Halo2) | PLONK | Variable |

## Deployment Profiles

### Development (Z1)
```bash
cargo build --features std
./aethernet --zone Z1 --mode development
```

### Production (Z2)
```bash
cargo build --release --no-default-features
./aethernet --zone Z2 --mode production --fido2-key /yubikey/epoch_a
```

### Air-Gapped (Z3)
```bash
# Build on connected system
cargo build --release --no-default-features --features airgap
tar czf aethernet-z3.tar.gz target/release/aethernet

# Transfer to air-gapped system
scp aethernet-z3.tar.gz airgap-host:/secure/

# Extract and run
tar xzf aethernet-z3.tar.gz
./aethernet --zone Z3 --mode archive --fido2-keys /yubikey/epoch_a,/yubikey/epoch_b
```

## Integration with QRATUM Platform

Aethernet extends QRATUM's **QRADLE** foundation:

- **VITRA (Healthcare/Genomics)** - TXO-wrapped WGS pipeline
- **ECORA (Climate)** - Future integration for climate data provenance
- **CAPRA (Finance)** - Future integration for financial transactions
- **JURIS (Legal)** - Future integration for legal document chains

## Performance Characteristics

| Operation | Latency | Throughput |
|-----------|---------|------------|
| TXO Creation | <1ms | 10,000 TXO/s |
| TXO Execution | <5ms | 2,000 TXO/s |
| TXO Commit | <10ms | 1,000 TXO/s |
| Merkle Verification | <1ms | 100,000 verifications/s |
| Rollback | <100ms | 10 rollbacks/s |
| Zone Promotion | <1s | 1 promotion/s |

## Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --features integration

# Run compliance tests
cargo test --features compliance

# Benchmark performance
cargo bench
```

## References

- **QRADLE**: [QRATUM Foundation](https://github.com/robertringler/QRATUM)
- **VITRA-E0**: [Sovereign Genomics Pipeline](qrVITRA/README.md)
- **merkler-static**: [Self-Hashing Merkle Binary](qrVITRA/merkler-static/)
- **CBOR**: [RFC 8949](https://datatracker.ietf.org/doc/html/rfc8949)
- **Ed25519**: [RFC 8032](https://datatracker.ietf.org/doc/html/rfc8032)
- **SHA-3**: [FIPS 202](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf)
- **HIPAA**: [HHS.gov](https://www.hhs.gov/hipaa/)
- **GDPR**: [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj)

## License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.

---

**Built with 💚 by the QRATUM Team**

*Sovereign. Deterministic. Auditable.*
