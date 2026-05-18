# P0.6 — Cryptographic Anchor Specification

## Hashes

* **Host side** — real Blake3 (`blake3 = "=1.5.4"`) over canonical
  receipt bytes. Used for cross-run cryptographic verification.
* **Kernel side** — deterministic 32-byte digest in
  `kernel::crypto::blake3_kernel`. `no_std`, allocation-free,
  collision-resistant for the receipt-replay determinism property
  (not claimed cryptographic against a real adversary).

The kernel digest's role is **append-only chained anchoring**, not
cryptographic verification. INV-Z requires append-only behaviour:
once an anchor at seq `s` is committed, neither its `merkle_root`
nor its `prev_link` may change.

## Verification path

External verifiers receive:

1. The kernel's Merkle root,
2. The full leaf vector,
3. The host's Blake3 root over the same leaves.

Both roots must agree on equality of inputs (i.e. identical leaf
vector ⇒ both roots stable across runs). The host root is the
externally-trusted value.

## Anchor schema

```
seq:        u64       // monotone, append-only
epoch:      u64       // execution_epoch at anchor time
merkle_root:[u8; 32]  // root over leaves committed before anchor
prev_link:  [u8; 32]  // chain link (digest of prior anchor + this root)
```
