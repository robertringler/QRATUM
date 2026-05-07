# P0.6 — Replay Receipt Specification

## Receipt structure

Every replay produces an append-only stream of *receipts*. Each
receipt is a 32-byte digest (Blake3 host-side, deterministic
32-byte digest kernel-side; see `crypto/blake3_kernel.rs`).

Receipts form a Merkle tree (`runtime_model::receipt_merkle`). The
**root** is the cryptographic anchor for the run.

## Invariants

| INV   | Property                                              | Enforced by                        |
|-------|-------------------------------------------------------|------------------------------------|
| INV-T | Every AP startup emits exactly one startup receipt     | `smp::ap_startup`                  |
| INV-U | Checkpoint restore reproduces identical receipt root  | `audit_replay_checkpoint`          |
| INV-V | Work stealing replay-stable across N cores            | `smp::deterministic_work_steal`    |
| INV-W | xAPIC and x2APIC paths execution-equivalent           | `smp::x2apic`                      |
| INV-X | Receipt Merkle roots deterministic                    | `runtime_model::receipt_merkle`    |
| INV-Y | Divergence envelope rejects schedule mutation         | `runtime_model::divergence_envelope` |
| INV-Z | Cryptographic receipt anchors append-only             | `audit_receipt_anchor`             |

## Determinism contract

Given the same `(input_stream, tick_clock, topology_id)`, the receipt
Merkle root is identical across runs. Topology equivalence (same
canonical (cpu_id, apic_id) mapping) is sufficient — physical APIC
ordering is not part of the determinism boundary.
