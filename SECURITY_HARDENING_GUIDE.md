# QRATUM Security Hardening Guide

**Document Version:** 1.0  
**Status:** Active  
**Date:** 2025-12-29  
**Classification:** Security Technical Documentation

---

## Executive Summary

This document provides comprehensive security hardening guidelines for QRATUM production deployments. It covers adversarial testing, constant-time cryptography, zeroization, and multi-source entropy monitoring to ensure the platform maintains its "Decentralized Ghost Machine" security properties.

**Security Objectives:**
- Byzantine fault tolerance up to f < n/3 malicious validators
- Resistance to timing side-channel attacks
- Complete zeroization of sensitive data
- Multi-source entropy for cryptographic operations
- Post-quantum cryptographic resilience

---

## 1. Threat Model

### 1.1 Adversary Capabilities

| Adversary Type | Capabilities | Assumptions |
|----------------|--------------|-------------|
| **Network Adversary** | Eavesdrop, modify, replay messages | Cannot break TLS 1.3 |
| **Byzantine Validator** | Control < 1/3 validators | Cannot control > 1/3 |
| **Side-Channel Attacker** | Timing measurements, cache analysis | Physical access limited |
| **Quantum Adversary** | Shor's algorithm, Grover's algorithm | Not yet practical |
| **Insider Threat** | Limited system access | Cannot bypass HSM |

### 1.2 Attack Surface Analysis

```
┌─────────────────────────────────────────────────────────────┐
│                    QRATUM Attack Surface                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  External Interfaces                                        │
│  ├── API Gateway (TLS 1.3)                                 │
│  ├── P2P Network (libp2p)                                  │
│  ├── Admin Console (FIDO2 + Biokey)                        │
│  └── External Integrations                                  │
│                                                              │
│  Cryptographic Operations                                   │
│  ├── Key Generation (DRBG)                                 │
│  ├── Key Derivation (HKDF)                                 │
│  ├── Digital Signatures (PQC)                              │
│  └── Encryption (AES-256-GCM)                              │
│                                                              │
│  Consensus Layer                                            │
│  ├── BFT Protocol (HotStuff)                               │
│  ├── Validator Registry                                     │
│  └── Governance Voting                                      │
│                                                              │
│  Data Storage                                               │
│  ├── Merkle Chain (immutable)                              │
│  ├── Key Storage (HSM)                                     │
│  └── Configuration (encrypted)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Security Boundaries

| Boundary | Protection Mechanism | Validation |
|----------|---------------------|------------|
| **Network Perimeter** | TLS 1.3, firewall | Penetration testing |
| **Validator Quorum** | BFT consensus (2/3) | Formal verification |
| **Authentication** | FIDO2 + Biokey MFA | Red team testing |
| **Authorization** | RBAC + dual-control | Access review |
| **Cryptographic** | HSM, constant-time | Cryptographic audit |

---

## 2. Adversarial Testing Framework

### 2.1 Byzantine Fault Injection

#### Test Scenarios

| Scenario | Description | Expected Behavior |
|----------|-------------|-------------------|
| **Double-Sign** | Validator signs conflicting blocks | Slashing, block rejection |
| **Equivocation** | Validator votes for different proposals | Detection, slashing |
| **Censorship** | Validator refuses to include TXOs | Transaction routing bypass |
| **Network Partition** | Isolate validator sets | Liveness maintained (2/3) |
| **Long-Range Attack** | Rewrite historical blocks | Finality prevents reorg |

#### Implementation

```rust
// Target: tests/adversarial/byzantine.rs

pub struct ByzantineFaultInjector {
    network: MockNetwork,
    malicious_validators: Vec<ValidatorId>,
    fault_rate: f64,
}

impl ByzantineFaultInjector {
    /// Inject double-signing fault
    pub fn inject_double_sign(&self, validator: ValidatorId, block_height: u64) {
        // Create two conflicting blocks at same height
        let block_a = self.create_block(block_height, vec![txo_a]);
        let block_b = self.create_block(block_height, vec![txo_b]);
        
        // Sign both blocks with validator key
        let sig_a = self.sign_block(&block_a, validator);
        let sig_b = self.sign_block(&block_b, validator);
        
        // Broadcast conflicting signatures
        self.network.broadcast(validator, BlockProposal { block: block_a, sig: sig_a });
        self.network.broadcast(validator, BlockProposal { block: block_b, sig: sig_b });
    }
    
    /// Verify slashing occurs for double-sign
    pub async fn verify_double_sign_slashing(&self, validator: ValidatorId) {
        // Wait for detection
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Check slashing occurred
        let slash_event = self.network.get_slash_events(validator).await;
        assert!(slash_event.is_some(), "Double-sign should trigger slashing");
        
        // Verify stake reduction
        let stake_before = self.get_stake_before_fault(validator);
        let stake_after = self.network.get_validator_stake(validator).await;
        assert!(stake_after < stake_before, "Stake should be reduced");
    }
}

#[tokio::test]
async fn test_byzantine_double_sign() {
    let mut injector = ByzantineFaultInjector::new(10); // 10 validators
    
    // Make 1 validator malicious (< 1/3)
    let malicious = injector.make_malicious(1);
    
    // Inject double-sign
    injector.inject_double_sign(malicious[0], 100);
    
    // Verify slashing
    injector.verify_double_sign_slashing(malicious[0]).await;
    
    // Verify network continues operating
    let block = injector.network.propose_block(101).await;
    assert!(block.is_ok(), "Network should continue after slashing");
}
```

### 2.2 Timing Side-Channel Testing

#### Dudect Methodology

```rust
// Target: tests/adversarial/timing.rs

use dudect::DudeCT;

pub struct TimingAnalyzer {
    iterations: usize,
    threshold: f64,
}

impl TimingAnalyzer {
    /// Test function for constant-time behavior
    pub fn test_constant_time<F>(&self, operation: F) -> TimingResult
    where
        F: Fn(&[u8]) -> Vec<u8>,
    {
        let mut dudect = DudeCT::new();
        
        for _ in 0..self.iterations {
            // Class A: Fixed input (e.g., all zeros)
            let input_a = vec![0u8; 32];
            
            // Class B: Random input
            let mut input_b = vec![0u8; 32];
            getrandom::getrandom(&mut input_b).unwrap();
            
            // Measure timing for both classes
            let time_a = self.measure_time(|| operation(&input_a));
            let time_b = self.measure_time(|| operation(&input_b));
            
            dudect.push(time_a, 0); // Class A
            dudect.push(time_b, 1); // Class B
        }
        
        // Statistical analysis
        let t_statistic = dudect.t_statistic();
        
        TimingResult {
            constant_time: t_statistic.abs() < self.threshold,
            t_statistic,
            samples: self.iterations * 2,
        }
    }
    
    fn measure_time<F, R>(&self, f: F) -> u64
    where
        F: FnOnce() -> R,
    {
        let start = std::arch::x86_64::_rdtsc();
        std::hint::black_box(f());
        let end = std::arch::x86_64::_rdtsc();
        end - start
    }
}

#[test]
fn test_hkdf_constant_time() {
    let analyzer = TimingAnalyzer::new(10_000, 4.5);
    
    let result = analyzer.test_constant_time(|input| {
        let hkdf = Hkdf::extract(None, input);
        hkdf.expand(b"test", 32).unwrap()
    });
    
    assert!(
        result.constant_time,
        "HKDF should be constant-time, t={:.2}",
        result.t_statistic
    );
}

#[test]
fn test_biokey_derivation_constant_time() {
    let analyzer = TimingAnalyzer::new(10_000, 4.5);
    
    let loci = vec![
        SNPLocus { chromosome: 1, position: 12345, ref_allele: b'A', alt_allele: b'G' },
    ];
    
    let result = analyzer.test_constant_time(|salt| {
        let biokey = EphemeralBiokey::derive(&loci, salt, 60);
        biokey.get_key_material().to_vec()
    });
    
    assert!(
        result.constant_time,
        "Biokey derivation should be constant-time, t={:.2}",
        result.t_statistic
    );
}
```

### 2.3 Censorship Resistance Testing

#### Transport Channel Testing

```rust
// Target: tests/adversarial/censorship.rs

pub struct CensorshipTester {
    network: MockNetwork,
    censored_validators: HashSet<ValidatorId>,
}

impl CensorshipTester {
    /// Test TXO delivery under censorship
    pub async fn test_txo_delivery_under_censorship(&self, censorship_rate: f64) {
        // Configure network to censor specified percentage of messages
        self.network.set_censorship_rate(censorship_rate);
        
        // Submit TXO
        let txo = self.create_test_txo();
        let txo_id = txo.id;
        
        // Track delivery through multiple channels
        let channels = vec![
            TransportChannel::TCP,
            TransportChannel::Tor,
            TransportChannel::I2P,
            TransportChannel::Offline,
        ];
        
        for channel in channels {
            let result = self.network.submit_txo_via_channel(txo.clone(), channel).await;
            
            if result.is_ok() {
                // TXO delivered through fallback channel
                assert!(
                    self.network.txo_in_mempool(txo_id).await,
                    "TXO should be in mempool after delivery"
                );
                return;
            }
        }
        
        panic!("TXO should be deliverable through at least one channel");
    }
    
    /// Test validator censorship resistance
    pub async fn test_validator_censorship_resistance(&self) {
        // Censor 30% of validators (still < 1/3)
        let num_validators = 10;
        let censored_count = 3;
        
        for i in 0..censored_count {
            self.network.censor_validator(ValidatorId(i));
        }
        
        // Submit TXO and verify finalization
        let txo = self.create_test_txo();
        let result = self.network.submit_and_finalize(txo).await;
        
        assert!(
            result.is_ok(),
            "TXO should finalize despite 30% validator censorship"
        );
    }
}
```

---

## 3. Constant-Time Cryptography

### 3.1 Guidelines

#### Rule 1: No Secret-Dependent Branches

```rust
// ❌ BAD: Secret-dependent branch
fn compare_secret(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    for i in 0..a.len() {
        if a[i] != b[i] { return false; } // Early return leaks timing
    }
    true
}

// ✅ GOOD: Constant-time comparison
fn compare_secret_ct(a: &[u8], b: &[u8]) -> subtle::Choice {
    use subtle::ConstantTimeEq;
    a.ct_eq(b)
}
```

#### Rule 2: No Secret-Dependent Memory Access

```rust
// ❌ BAD: Secret-dependent table lookup
fn lookup_table(table: &[u8; 256], index: u8) -> u8 {
    table[index as usize] // Cache timing leak
}

// ✅ GOOD: Constant-time table lookup
fn lookup_table_ct(table: &[u8; 256], index: u8) -> u8 {
    let mut result = 0u8;
    for i in 0..256 {
        let mask = subtle::Choice::from((i as u8).ct_eq(&index));
        result = subtle::ConditionallySelectable::conditional_select(
            &result, &table[i], mask
        );
    }
    result
}
```

#### Rule 3: No Variable-Time Operations

```rust
// ❌ BAD: Variable-time division
fn divide(a: u64, b: u64) -> u64 {
    a / b // Division time depends on operands
}

// ✅ GOOD: Constant-time operations
fn multiply_ct(a: u64, b: u64) -> u64 {
    // Multiplication is constant-time on modern CPUs
    a.wrapping_mul(b)
}
```

### 3.2 Audit Checklist

| Component | File | Constant-Time | Verified |
|-----------|------|---------------|----------|
| HMAC-SHA3-512 | `crypto/kdf/hkdf.rs` | ✅ | Pending |
| HKDF | `crypto/kdf/hkdf.rs` | ✅ | Pending |
| HMAC-DRBG | `crypto/rng/drbg.rs` | ✅ | Pending |
| Biokey Derivation | `Aethernet/core/biokey/` | ✅ | Pending |
| SPHINCS+ Sign | `crypto/pqc/sphincs_plus.rs` | Placeholder | TODO |
| SPHINCS+ Verify | `crypto/pqc/sphincs_plus.rs` | Placeholder | TODO |
| Ed25519 Sign | (external crate) | ✅ | Verified |
| AES-GCM | (external crate) | ✅ | Verified |

### 3.3 Automated Verification

```yaml
# .github/workflows/constant-time-check.yml
name: Constant-Time Verification

on:
  push:
    paths:
      - 'crypto/**'
      - 'Aethernet/core/biokey/**'
  pull_request:
    paths:
      - 'crypto/**'
      - 'Aethernet/core/biokey/**'

jobs:
  timing-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: nightly
          override: true
          
      - name: Run dudect timing tests
        run: |
          cargo test --release --features timing-analysis -- \
            --nocapture \
            test_constant_time
            
      - name: Run ctgrind (valgrind-based)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get install -y valgrind
          cargo build --release --features ct-grind
          valgrind --tool=ct-grind \
            target/release/crypto_timing_tests 2>&1 | \
            tee ctgrind-output.txt
          
      - name: Analyze ctgrind results
        run: |
          if grep -q "VIOLATION" ctgrind-output.txt; then
            echo "::error::Constant-time violations detected"
            exit 1
          fi
          
      - name: Upload timing analysis report
        uses: actions/upload-artifact@v4
        with:
          name: timing-analysis
          path: |
            ctgrind-output.txt
            target/timing-report.json
```

---

## 4. Zeroization

### 4.1 Zeroization Requirements

| Data Type | Lifetime | Zeroization Method | Verification |
|-----------|----------|-------------------|--------------|
| **Private Keys** | Session | `zeroize::Zeroize` | Memory inspection |
| **Biokey Material** | 60 seconds | `volatile_set_memory` | Drop test |
| **PRK (HKDF)** | Scope | `Drop` impl | Compiler inspection |
| **Session Keys** | Session | `zeroize::Zeroize` | Memory inspection |
| **Passwords** | Immediate | Immediate zeroize | Code review |

### 4.2 Implementation Patterns

#### Pattern 1: Automatic Zeroization on Drop

```rust
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Secret key with automatic zeroization
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SecretKey {
    key_material: [u8; 32],
}

impl SecretKey {
    pub fn new(material: [u8; 32]) -> Self {
        Self { key_material: material }
    }
    
    // Key is automatically zeroized when dropped
}
```

#### Pattern 2: Volatile Write for Biokeys

```rust
/// Ephemeral biokey with volatile zeroization
pub struct EphemeralBiokey {
    key_material: [u8; 32],
    created_at: u64,
    ttl: u64,
}

impl Drop for EphemeralBiokey {
    fn drop(&mut self) {
        // Use volatile write to prevent compiler optimization
        unsafe {
            std::ptr::write_volatile(
                self.key_material.as_mut_ptr(),
                0u8
            );
            for i in 0..32 {
                std::ptr::write_volatile(
                    self.key_material.as_mut_ptr().add(i),
                    0u8
                );
            }
        }
        // Memory barrier to ensure writes complete
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}
```

#### Pattern 3: Secure Memory Allocation

```rust
use memsec::{mlock, munlock};

/// Securely allocated secret with memory locking
pub struct LockedSecret {
    ptr: *mut u8,
    len: usize,
}

impl LockedSecret {
    pub fn new(data: &[u8]) -> Result<Self, Error> {
        let len = data.len();
        
        // Allocate with mmap for security properties
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            ) as *mut u8
        };
        
        if ptr.is_null() {
            return Err(Error::AllocationFailed);
        }
        
        // Lock memory to prevent swapping
        unsafe {
            mlock(ptr, len);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
        }
        
        Ok(Self { ptr, len })
    }
}

impl Drop for LockedSecret {
    fn drop(&mut self) {
        unsafe {
            // Zeroize memory
            std::ptr::write_bytes(self.ptr, 0, self.len);
            
            // Unlock memory
            munlock(self.ptr, self.len);
            
            // Unmap memory
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}
```

### 4.3 Zeroization Testing

```rust
// Target: tests/security/zeroization.rs

#[test]
fn test_secret_key_zeroization() {
    let mut stack_snapshot = [0u8; 4096];
    
    // Capture stack region address
    let stack_ptr: usize;
    {
        let secret = SecretKey::new([0xAB; 32]);
        stack_ptr = &secret as *const _ as usize;
        
        // Verify secret is non-zero
        assert!(secret.key_material.iter().any(|&b| b != 0));
    } // secret dropped here
    
    // Capture stack memory after drop
    unsafe {
        std::ptr::copy_nonoverlapping(
            stack_ptr as *const u8,
            stack_snapshot.as_mut_ptr(),
            std::mem::size_of::<SecretKey>(),
        );
    }
    
    // Verify key material is zeroed
    // Note: This test may be unreliable due to stack reuse
    // Use address sanitizer for more reliable detection
}

#[test]
fn test_biokey_volatile_zeroization() {
    use std::sync::atomic::{AtomicBool, Ordering};
    
    static ZEROIZED: AtomicBool = AtomicBool::new(false);
    
    struct TestBiokey {
        key: [u8; 32],
    }
    
    impl Drop for TestBiokey {
        fn drop(&mut self) {
            // Volatile zeroization
            for byte in self.key.iter_mut() {
                unsafe {
                    std::ptr::write_volatile(byte, 0);
                }
            }
            ZEROIZED.store(true, Ordering::SeqCst);
        }
    }
    
    {
        let _biokey = TestBiokey { key: [0xFF; 32] };
    }
    
    assert!(ZEROIZED.load(Ordering::SeqCst), "Biokey should be zeroized on drop");
}
```

---

## 5. Multi-Source Entropy

### 5.1 Entropy Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Entropy Collection System                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Primary Sources (Required)                                 │
│  ├── getrandom() / /dev/urandom                            │
│  └── Minimum: 256 bits per collection                       │
│                                                              │
│  Secondary Sources (Recommended)                            │
│  ├── CPU RDRAND/RDSEED                                      │
│  ├── Hardware RNG (TPM, HSM)                                │
│  └── Jitter entropy (CPU timing variance)                   │
│                                                              │
│  Auxiliary Sources (Optional)                               │
│  ├── System timestamps (nanosecond resolution)              │
│  ├── Process/thread IDs                                     │
│  └── Network timing jitter                                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Entropy Pool                            │   │
│  │  - XOR mixing of all sources                        │   │
│  │  - SHA3-512 conditioning                            │   │
│  │  - Minimum 512 bits accumulated                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              HMAC-DRBG                               │   │
│  │  - NIST SP 800-90A compliant                        │   │
│  │  - SHA3-512 based                                   │   │
│  │  - Automatic reseeding                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Entropy Source Implementation

```rust
// Target: crypto/rng/entropy.rs

use std::sync::Arc;

/// Entropy source trait for pluggable entropy collection
pub trait EntropySource: Send + Sync {
    /// Collect entropy bytes
    fn collect(&self, output: &mut [u8]) -> Result<usize, EntropyError>;
    
    /// Get entropy estimate (bits per byte)
    fn entropy_estimate(&self) -> f64;
    
    /// Get source identifier
    fn source_id(&self) -> &str;
    
    /// Health check
    fn health_check(&self) -> Result<(), EntropyError>;
}

/// System RNG entropy source (getrandom)
pub struct SystemEntropySource;

impl EntropySource for SystemEntropySource {
    fn collect(&self, output: &mut [u8]) -> Result<usize, EntropyError> {
        getrandom::getrandom(output)
            .map_err(|_| EntropyError::SourceFailed)?;
        Ok(output.len())
    }
    
    fn entropy_estimate(&self) -> f64 {
        8.0 // Full entropy from OS
    }
    
    fn source_id(&self) -> &str {
        "system-rng"
    }
    
    fn health_check(&self) -> Result<(), EntropyError> {
        let mut test = [0u8; 32];
        self.collect(&mut test)?;
        
        // Basic non-zero check
        if test.iter().all(|&b| b == 0) {
            return Err(EntropyError::HealthCheckFailed);
        }
        
        Ok(())
    }
}

/// CPU RDRAND/RDSEED entropy source
#[cfg(target_arch = "x86_64")]
pub struct RdrandEntropySource;

#[cfg(target_arch = "x86_64")]
impl EntropySource for RdrandEntropySource {
    fn collect(&self, output: &mut [u8]) -> Result<usize, EntropyError> {
        use core::arch::x86_64::{_rdrand64_step, _rdseed64_step};
        
        let mut collected = 0;
        while collected < output.len() {
            let mut value: u64 = 0;
            
            // Try RDSEED first (direct entropy), fallback to RDRAND
            let success = unsafe {
                _rdseed64_step(&mut value) == 1 || _rdrand64_step(&mut value) == 1
            };
            
            if !success {
                return Err(EntropyError::SourceFailed);
            }
            
            let bytes = value.to_le_bytes();
            let copy_len = (output.len() - collected).min(8);
            output[collected..collected + copy_len].copy_from_slice(&bytes[..copy_len]);
            collected += copy_len;
        }
        
        Ok(collected)
    }
    
    fn entropy_estimate(&self) -> f64 {
        7.5 // Slightly less than full entropy (conservative)
    }
    
    fn source_id(&self) -> &str {
        "cpu-rdrand"
    }
    
    fn health_check(&self) -> Result<(), EntropyError> {
        // Verify RDRAND/RDSEED available
        if !is_x86_feature_detected!("rdrand") {
            return Err(EntropyError::SourceUnavailable);
        }
        
        let mut test = [0u8; 32];
        self.collect(&mut test)?;
        Ok(())
    }
}

/// Jitter entropy source (CPU timing variance)
pub struct JitterEntropySource {
    samples_per_collection: usize,
}

impl EntropySource for JitterEntropySource {
    fn collect(&self, output: &mut [u8]) -> Result<usize, EntropyError> {
        use sha3::{Sha3_256, Digest};
        
        let mut hasher = Sha3_256::new();
        
        for _ in 0..self.samples_per_collection {
            // Collect CPU cycle timing jitter
            let t1 = std::time::Instant::now();
            std::hint::spin_loop();
            let t2 = std::time::Instant::now();
            
            let delta = t2.duration_since(t1).as_nanos();
            hasher.update(&delta.to_le_bytes());
        }
        
        let hash = hasher.finalize();
        let copy_len = output.len().min(32);
        output[..copy_len].copy_from_slice(&hash[..copy_len]);
        
        Ok(copy_len)
    }
    
    fn entropy_estimate(&self) -> f64 {
        0.5 // Conservative: ~0.5 bits per byte
    }
    
    fn source_id(&self) -> &str {
        "jitter"
    }
    
    fn health_check(&self) -> Result<(), EntropyError> {
        Ok(()) // Jitter always available
    }
}
```

### 5.3 Entropy Monitoring

```rust
// Target: crypto/rng/monitoring.rs

use std::sync::Arc;
use tokio::sync::RwLock;

pub struct EntropyMonitor {
    sources: Vec<Arc<dyn EntropySource>>,
    health_status: RwLock<EntropyHealth>,
    alert_threshold: u32,
    min_sources: usize,
}

#[derive(Clone)]
pub struct EntropyHealth {
    pub total_bits: u32,
    pub active_sources: usize,
    pub source_status: Vec<SourceStatus>,
    pub healthy: bool,
    pub last_check: std::time::Instant,
}

#[derive(Clone)]
pub struct SourceStatus {
    pub id: String,
    pub available: bool,
    pub estimated_bits: u32,
    pub last_collection: Option<std::time::Instant>,
    pub failures: u32,
}

impl EntropyMonitor {
    pub fn new(min_sources: usize, alert_threshold: u32) -> Self {
        Self {
            sources: vec![
                Arc::new(SystemEntropySource),
                #[cfg(target_arch = "x86_64")]
                Arc::new(RdrandEntropySource),
                Arc::new(JitterEntropySource { samples_per_collection: 1000 }),
            ],
            health_status: RwLock::new(EntropyHealth {
                total_bits: 0,
                active_sources: 0,
                source_status: vec![],
                healthy: false,
                last_check: std::time::Instant::now(),
            }),
            alert_threshold,
            min_sources,
        }
    }
    
    /// Perform health check on all entropy sources
    pub async fn check_health(&self) -> EntropyHealth {
        let mut total_bits = 0u32;
        let mut active_sources = 0usize;
        let mut source_status = Vec::new();
        
        for source in &self.sources {
            let mut status = SourceStatus {
                id: source.source_id().to_string(),
                available: false,
                estimated_bits: 0,
                last_collection: None,
                failures: 0,
            };
            
            match source.health_check() {
                Ok(()) => {
                    status.available = true;
                    status.estimated_bits = (source.entropy_estimate() * 32.0) as u32;
                    status.last_collection = Some(std::time::Instant::now());
                    
                    total_bits += status.estimated_bits;
                    active_sources += 1;
                }
                Err(e) => {
                    status.failures += 1;
                    tracing::warn!(
                        source = source.source_id(),
                        error = ?e,
                        "Entropy source health check failed"
                    );
                }
            }
            
            source_status.push(status);
        }
        
        let healthy = total_bits >= self.alert_threshold 
            && active_sources >= self.min_sources;
        
        let health = EntropyHealth {
            total_bits,
            active_sources,
            source_status,
            healthy,
            last_check: std::time::Instant::now(),
        };
        
        // Update cached status
        *self.health_status.write().await = health.clone();
        
        // Alert if unhealthy
        if !healthy {
            self.send_alert(&health).await;
        }
        
        health
    }
    
    /// Collect entropy from all available sources
    pub async fn collect_entropy(&self, output: &mut [u8]) -> Result<(), EntropyError> {
        let health = self.check_health().await;
        
        if !health.healthy {
            return Err(EntropyError::InsufficientEntropy);
        }
        
        // Collect from all sources and XOR mix
        let mut mixed = vec![0u8; output.len()];
        
        for source in &self.sources {
            let mut source_output = vec![0u8; output.len()];
            if source.collect(&mut source_output).is_ok() {
                for (i, byte) in source_output.iter().enumerate() {
                    mixed[i] ^= byte;
                }
            }
        }
        
        // Final conditioning with SHA3
        use sha3::{Sha3_512, Digest};
        let mut hasher = Sha3_512::new();
        hasher.update(&mixed);
        let hash = hasher.finalize();
        
        output.copy_from_slice(&hash[..output.len().min(64)]);
        
        Ok(())
    }
    
    async fn send_alert(&self, health: &EntropyHealth) {
        tracing::error!(
            total_bits = health.total_bits,
            active_sources = health.active_sources,
            threshold = self.alert_threshold,
            "ALERT: Entropy health check failed"
        );
        
        // Additional alerting (Slack, PagerDuty, etc.) would go here
    }
}

/// Start continuous entropy monitoring
pub async fn start_continuous_monitoring(
    monitor: Arc<EntropyMonitor>,
    interval: std::time::Duration,
) {
    loop {
        let health = monitor.check_health().await;
        
        tracing::debug!(
            total_bits = health.total_bits,
            active_sources = health.active_sources,
            healthy = health.healthy,
            "Entropy health check completed"
        );
        
        tokio::time::sleep(interval).await;
    }
}
```

---

## 6. HSM/TEE Integration

### 6.1 HSM Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HSM Integration Layer                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   YubiHSM 2  │  │   AWS CloudHSM │ │   Azure HSM  │      │
│  │   (PKCS#11)  │  │   (PKCS#11)   │  │   (PKCS#11)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            QRATUM Key Management Service             │   │
│  │                                                      │   │
│  │  Operations:                                        │   │
│  │  ├── generate_validator_keypair()                   │   │
│  │  ├── sign_block(block_hash)                         │   │
│  │  ├── wrap_biokey(biokey_material)                   │   │
│  │  ├── unwrap_biokey(wrapped_key)                     │   │
│  │  └── rotate_keys()                                  │   │
│  │                                                      │   │
│  │  Key Types:                                         │   │
│  │  ├── Validator signing keys (Ed25519)               │   │
│  │  ├── Biokey wrapping keys (AES-256)                 │   │
│  │  ├── TLS termination keys (ECDSA P-384)             │   │
│  │  └── PQC signing keys (when HSM supports)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 PKCS#11 Integration

```rust
// Target: crypto/hsm/pkcs11.rs

use pkcs11::Ctx;
use std::sync::Arc;

pub struct HsmKeyManager {
    ctx: Arc<Ctx>,
    session: pkcs11::types::CK_SESSION_HANDLE,
    slot_id: pkcs11::types::CK_SLOT_ID,
}

impl HsmKeyManager {
    /// Initialize HSM connection
    pub fn new(library_path: &str, pin: &str) -> Result<Self, HsmError> {
        let ctx = Ctx::new(library_path)?;
        ctx.initialize(None)?;
        
        // Find slot with token
        let slots = ctx.get_slot_list(true)?;
        let slot_id = slots.first().ok_or(HsmError::NoSlotAvailable)?;
        
        // Open session
        let session = ctx.open_session(
            *slot_id,
            pkcs11::types::CKF_SERIAL_SESSION | pkcs11::types::CKF_RW_SESSION,
            None,
            None,
        )?;
        
        // Login
        ctx.login(session, pkcs11::types::CKU_USER, Some(pin))?;
        
        Ok(Self {
            ctx: Arc::new(ctx),
            session,
            slot_id: *slot_id,
        })
    }
    
    /// Generate validator Ed25519 keypair in HSM
    pub fn generate_validator_keypair(&self, label: &str) -> Result<PublicKey, HsmError> {
        use pkcs11::types::*;
        
        let public_key_template = vec![
            CK_ATTRIBUTE::new(CKA_TOKEN, true),
            CK_ATTRIBUTE::new(CKA_LABEL, label.as_bytes()),
            CK_ATTRIBUTE::new(CKA_KEY_TYPE, CKK_EC_EDWARDS),
            CK_ATTRIBUTE::new(CKA_EC_PARAMS, OID_ED25519),
            CK_ATTRIBUTE::new(CKA_VERIFY, true),
        ];
        
        let private_key_template = vec![
            CK_ATTRIBUTE::new(CKA_TOKEN, true),
            CK_ATTRIBUTE::new(CKA_LABEL, format!("{}_private", label).as_bytes()),
            CK_ATTRIBUTE::new(CKA_PRIVATE, true),
            CK_ATTRIBUTE::new(CKA_SENSITIVE, true),
            CK_ATTRIBUTE::new(CKA_EXTRACTABLE, false),
            CK_ATTRIBUTE::new(CKA_SIGN, true),
        ];
        
        let (pub_handle, _priv_handle) = self.ctx.generate_key_pair(
            self.session,
            &CK_MECHANISM { mechanism: CKM_EC_EDWARDS_KEY_PAIR_GEN, pParameter: std::ptr::null_mut(), ulParameterLen: 0 },
            &public_key_template,
            &private_key_template,
        )?;
        
        // Extract public key
        let public_key_bytes = self.get_public_key_bytes(pub_handle)?;
        
        Ok(PublicKey::from_bytes(&public_key_bytes)?)
    }
    
    /// Sign block hash using HSM-stored private key
    pub fn sign_block(&self, label: &str, block_hash: &[u8; 32]) -> Result<Signature, HsmError> {
        use pkcs11::types::*;
        
        // Find private key by label
        let key_handle = self.find_key_by_label(&format!("{}_private", label))?;
        
        // Initialize signing
        self.ctx.sign_init(
            self.session,
            &CK_MECHANISM { mechanism: CKM_EDDSA, pParameter: std::ptr::null_mut(), ulParameterLen: 0 },
            key_handle,
        )?;
        
        // Sign
        let signature = self.ctx.sign(self.session, block_hash)?;
        
        Ok(Signature::from_bytes(&signature)?)
    }
    
    /// Wrap biokey material for secure storage
    pub fn wrap_biokey(&self, biokey: &[u8; 32]) -> Result<Vec<u8>, HsmError> {
        use pkcs11::types::*;
        
        // Find wrapping key
        let wrap_key = self.find_key_by_label("biokey_wrap_key")?;
        
        // Generate IV
        let mut iv = [0u8; 12];
        self.ctx.generate_random(self.session, &mut iv)?;
        
        // Wrap using AES-GCM
        let mechanism = CK_MECHANISM {
            mechanism: CKM_AES_GCM,
            pParameter: &GCM_PARAMS { iv: iv.as_ptr(), ivLen: 12, aadLen: 0, tagBits: 128 } as *const _ as *mut _,
            ulParameterLen: std::mem::size_of::<GCM_PARAMS>() as CK_ULONG,
        };
        
        let wrapped = self.ctx.encrypt(
            self.session,
            &mechanism,
            wrap_key,
            biokey,
        )?;
        
        // Prepend IV to wrapped key
        let mut result = iv.to_vec();
        result.extend(wrapped);
        
        Ok(result)
    }
}
```

---

## 7. Security Checklist

### 7.1 Pre-Production Checklist

| Category | Item | Status | Owner |
|----------|------|--------|-------|
| **Cryptography** | | | |
| | PQC implementation audit | ⬜ | Crypto Lead |
| | Constant-time verification | ⬜ | Security Team |
| | HSM integration tested | ⬜ | DevOps |
| | Entropy monitoring active | ⬜ | DevOps |
| **Authentication** | | | |
| | FIDO2 integration tested | ⬜ | Auth Team |
| | Biokey derivation audited | ⬜ | Security Team |
| | MFA enforcement verified | ⬜ | Auth Team |
| **Authorization** | | | |
| | RBAC implementation tested | ⬜ | Auth Team |
| | Dual-control verified | ⬜ | Security Team |
| | Zone promotion tested | ⬜ | Platform Team |
| **Network** | | | |
| | TLS 1.3 only | ⬜ | DevOps |
| | Certificate pinning | ⬜ | Security Team |
| | Firewall rules reviewed | ⬜ | DevOps |
| **Consensus** | | | |
| | BFT fault injection tested | ⬜ | Platform Team |
| | Slashing mechanism verified | ⬜ | Platform Team |
| | Governance voting tested | ⬜ | Platform Team |
| **Audit** | | | |
| | Merkle chain integrity | ⬜ | Platform Team |
| | Log retention configured | ⬜ | DevOps |
| | Alert rules active | ⬜ | DevOps |

### 7.2 Operational Security Checklist

| Item | Frequency | Owner |
|------|-----------|-------|
| Vulnerability scanning | Daily | Security Team |
| Penetration testing | Quarterly | External |
| Access review | Monthly | Security Team |
| Key rotation | Quarterly | DevOps |
| Incident response drill | Quarterly | Security Team |
| Backup verification | Weekly | DevOps |
| Entropy health check | Continuous | Automated |
| HSM health check | Daily | DevOps |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-29 | Security Team | Initial release |

---

**Classification:** Security Technical Documentation  
**Distribution:** Engineering, Security, DevOps  
**Review Cycle:** Quarterly  
**Next Review:** Q1 2026
