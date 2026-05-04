# Q-Substrate v1.0 - Build Instructions

## Overview

Q-Substrate is an ultra-lightweight deterministic AI + Quantum runtime targeting:

- **Binary size**: 400-500 KB compressed
- **Memory footprint**: ≤32 MB default, configurable
- **Features**: MiniLM Q4 inference + 12-qubit quantum simulation + DCGE code generation

## Quick Build

### Standard Build

```bash
cd q-substrate
cargo build --release
```

### Size-Optimized Build (Sub-500KB Target)

```bash
cd q-substrate
RUSTFLAGS="-C opt-level=z -C lto=fat -C codegen-units=1 -C panic=abort" \
  cargo build --release

# Check binary size
ls -la target/release/q-substrate
```

### Compressed Build (UPX)

```bash
# Build release binary
cargo build --release

# Compress with UPX (if available)
upx --ultra-brute target/release/q-substrate

# Alternative: LZMA compression
lzma -9 target/release/q-substrate
```

## Build Profiles

### Desktop (Full Features)

```bash
cargo build --release
```

- 12 qubits
- Full MiniLM inference
- 32 MB memory limit

### Micro (ESP32/RP2040)

```bash
cargo build --release --features micro --no-default-features
```

- 8 qubits
- Streaming inference
- 4 MB memory limit

### WASM Browser

```bash
cargo build --release --target wasm32-unknown-unknown --features wasm
```

- 12 qubits
- Browser-compatible
- WebGL integration ready

## Binary Size Analysis

```bash
# Show section sizes
size target/release/q-substrate

# Detailed analysis
cargo bloat --release

# Function-level breakdown
cargo bloat --release --filter q_substrate
```

## Performance Benchmarks

```bash
# Run benchmarks
cargo bench

# Memory profiling
cargo run --release -- --benchmark
```

## Target Sizes

| Component | Target | Justification |
|-----------|--------|---------------|
| .text | ≤4 KB | Minimal code section |
| .stack | ≤1 KB | Stack allocation |
| .heap | 0 | Zero-heap quantum pod |
| Quantum state | 32 KB | 4096 amplitudes × 8 bytes |
| Total | ≤500 KB | Compressed binary |

## Supremacy Invariants

The build must preserve these invariants:

1. **ℛ(t) ≥ 0**: Non-negative resources at all times
2. **Deterministic**: Same inputs always produce same outputs
3. **Pod Isolation**: AI, Quantum, DCGE modules isolated
4. **Audit Trail**: All operations logged
5. **Rollback Compatible**: Can restore to any checkpoint

## Verification

```bash
# Run tests
cargo test

# Verify determinism
cargo test test_determinism

# Check binary metrics
cargo run --release -- --metrics
```

## Configuration

Edit `Cargo.toml` profile section for custom optimization:

```toml
[profile.release]
opt-level = "z"          # Size optimization
lto = true               # Link-time optimization
codegen-units = 1        # Single codegen unit
panic = "abort"          # No unwinding
strip = true             # Strip symbols
```

## Cross-Compilation

### ARM64 (Raspberry Pi)

```bash
cargo build --release --target aarch64-unknown-linux-gnu
```

### ESP32

```bash
cargo build --release --target xtensa-esp32-espidf
```

### WASM

```bash
cargo build --release --target wasm32-unknown-unknown
wasm-opt -Oz target/wasm32-unknown-unknown/release/q_substrate.wasm -o q-substrate.wasm
```

## Memory Configuration

Default memory limits can be adjusted in `src/config.rs`:

```rust
MemoryConfig {
    total_limit_mb: 32,        // Total RAM limit
    ai_pod_limit_kb: 8192,     // AI module
    quantum_pod_limit_kb: 64,   // Quantum module
    dcge_pod_limit_kb: 4,       // Code generation
}
```

## License

Apache-2.0
