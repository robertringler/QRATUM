# Q-Substrate v1.0

**Ultra-Lightweight Deterministic AI + Quantum Runtime**

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Binary Size](https://img.shields.io/badge/binary-<500KB-green.svg)](BUILD.md)

## Overview

Q-Substrate is a sovereign, deterministic runtime combining:

- **MiniLM-L6-v2 Q4**: Streaming 4-bit quantized inference (~8MB model, ~20KB active)
- **Mini QuASIM**: 12-qubit quantum simulation with full gate set
- **DCGE**: Deterministic Code Generation Engine with >99% compile success
- **WASM Pods**: Isolated execution for AI, Quantum, and DCGE modules

### Key Features

| Feature | Specification |
|---------|---------------|
| Binary Size | 400-500 KB compressed |
| Memory Footprint | ≤32 MB default |
| Qubits | 12 (Desktop), 8 (Micro), 6 (Embedded) |
| AI Model | MiniLM-L6-v2 Q4 quantized |
| Languages | Rust, Python, JavaScript, C |
| Determinism | Fully seed-controlled |

## Quick Start

```rust
use q_substrate::{QSubstrate, QuantumGate};

fn main() {
    // Initialize runtime
    let mut qs = QSubstrate::new();
    
    // Run quantum simulation
    let gates = vec![
        QuantumGate::Hadamard(0),
        QuantumGate::CNOT(0, 1),
    ];
    let probs = qs.run_quantum(&gates);
    println!("Bell state: P(00)={:.3}, P(11)={:.3}", probs[0], probs[3]);
    
    // Run AI inference
    let embedding = qs.run_inference("quantum supremacy test");
    let intent = qs.classify_intent("generate fibonacci code");
    
    // Generate code
    let code = qs.generate_code("fibonacci function", "rust").unwrap();
    println!("{}", code.source);
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Q-Substrate Core                      │
├───────────────┬───────────────┬───────────────┬─────────┤
│   AI Pod      │  Quantum Pod  │   DCGE Pod    │  Audit  │
│  MiniLM Q4    │ Mini QuASIM   │  Code Gen     │  Log    │
│  Streaming    │  12 Qubits    │  Multi-Lang   │         │
└───────────────┴───────────────┴───────────────┴─────────┘
         │               │               │
         └───────────────┴───────────────┘
                    WASM Pod Isolation
                    (Provenance Logged)
```

## Quantum Gates

| Gate | Description | Usage |
|------|-------------|-------|
| H | Hadamard | `QuantumGate::Hadamard(qubit)` |
| X, Y, Z | Pauli gates | `QuantumGate::PauliX(qubit)` |
| S | Phase gate | `QuantumGate::Phase(qubit)` |
| T, T† | T gates | `QuantumGate::T(qubit)` |
| CNOT | Controlled-NOT | `QuantumGate::CNOT(ctrl, tgt)` |
| CZ | Controlled-Z | `QuantumGate::CZ(ctrl, tgt)` |
| SWAP | Swap qubits | `QuantumGate::SWAP(q1, q2)` |
| Toffoli | CCNOT | `QuantumGate::Toffoli(c1, c2, t)` |
| RX, RY, RZ | Rotations | `QuantumGate::RX(qubit, theta)` |

## Runtime Modes

```rust
// Desktop (full features)
let qs = QSubstrate::with_config(QSubstrateConfig::desktop());

// Micro (ESP32/RP2040)
let qs = QSubstrate::with_config(QSubstrateConfig::micro());

// Embedded (minimal footprint)
let qs = QSubstrate::with_config(QSubstrateConfig::embedded());

// WASM Browser
let qs = QSubstrate::with_config(QSubstrateConfig::wasm_browser());
```

## Supremacy Invariants

Q-Substrate guarantees these 8 invariants:

1. **ℛ(t) ≥ 0**: Non-negative resources
2. **Deterministic Execution**: Same seed → same output
3. **No Side-Channel Leaks**: Pod isolation
4. **Pod Isolation**: AI/Quantum/DCGE sandboxed
5. **Rollback Compatible**: Checkpoint/restore
6. **Audit Trail Complete**: All operations logged
7. **Memory Bounds**: Configurable limits enforced
8. **No Goal Drift**: Fixed behavior specification

## Failure Modes

| Code | Condition | Containment |
|------|-----------|-------------|
| Q001 | Qubit out of range | Silently ignored |
| Q002 | Normalization loss | Auto-renormalize |
| Q003 | Invalid gate params | Clamp to range |
| A001 | Seed corruption | Reset to seed 42 |
| A002 | Embedding overflow | Normalize |
| P001 | Memory exceeded | Pod rollback |
| P002 | Inter-pod failure | Retry |
| C001 | Code validation fail | Regenerate |

## Build

```bash
# Standard build
cargo build --release

# Size-optimized (target <500KB)
RUSTFLAGS="-C opt-level=z -C lto=fat" cargo build --release

# Run tests
cargo test

# Run demo
cargo run --release
```

See [BUILD.md](BUILD.md) for detailed build instructions.

## Performance

| Metric | Desktop | Micro | Embedded |
|--------|---------|-------|----------|
| Bell state | <1ms | <5ms | <10ms |
| GHZ state | <2ms | <10ms | <20ms |
| AI embed | <10ms | <50ms | <100ms |
| Code gen | <5ms | <20ms | <50ms |
| Memory | 32MB | 4MB | 1MB |

## License

Apache-2.0

## Related

- [QRATUM Platform](../README.md) - Full QRATUM system
- [QuASIM](../QuASIM/) - Full quantum simulation
- [QRATUM Desktop](../qratum_desktop/) - Desktop application

---

**Q-Substrate** - *Supremacy through Minimalism*
