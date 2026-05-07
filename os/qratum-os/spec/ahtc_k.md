# AHTC-K — Anti-Holographic Tensor Canonical Kernel Layer (formal spec)

> Pre-module to P0.3. Sits **between** arbitration output and scheduler enqueue.
> AHTC-K is a **pure deterministic canonicalization + batching** layer.

## 0. Position in the pipeline

```
intent (raw IntentEnvelope)
   │
   ▼
arbiter::evaluate (intent.rs)        ← arbitration sees RAW intents only
   │
   ├── Deny → audit::IntentVerdict, return verdict
   │
   └── Accept
        │
        ▼
   ahtc_k::canonicalize(intent) → CanonicalKey      (K3, K4, K5)
        │
        ▼
   ahtc_k::fold_or_insert(rq, key, intent_id, ts)   (K1, K2)
        │   ├── new key  → audit::BatchCreate
        │   └── existing → audit::BatchIncrement (count += 1)
        ▼
   sched::enqueue_batch(IntentBatch)                 (frozen interface)
```

AHTC-K **never** affects the verdict. The verdict is computed before
canonicalization and is recorded in audit unchanged.

## 1. Hard constraints (audit map)

| ID  | Constraint                                                 | Enforced by                                              |
|-----|------------------------------------------------------------|----------------------------------------------------------|
| K1  | identical input stream → identical canonical output stream | `canonicalize` is total + pure; `fold_or_insert` linear  |
| K2  | replay reconstructs original intent stream from audit only | `BatchCreate`/`BatchIncrement` carry full `intent_id`+ts |
| K3  | arbitration invariance                                     | `intent::submit` runs *before* `ahtc_k::*`               |
| K4  | lock_key immutability                                      | `lock_key` is an *input* to canonicalize, not derived    |
| K5  | grouping only by exact structural equivalence              | `CanonicalKey` equality is bytewise; no fuzzing          |

## 2. Canonical Key (ABI)

```rust
#[repr(C, align(8))]
pub struct CanonicalKey {
    pub authority:   u8,    // 1..=4
    pub scope:       u8,    // 1..=4
    pub priority:    u8,    // 1..=4
    pub _pad:        u8,
    pub name_hash:   u64,   // Blake3(name_bytes)[..8] LE   (host)
                            // FNV-1a-64(name_bytes)        (kernel, XXX-blake3)
    pub params_hash: u64,   // Blake3(params_bytes)[..8] LE (host)
                            // FNV-1a-64(params_bytes)      (kernel, XXX-blake3)
    pub lock_key:    u64,   // verbatim from arbiter; AHTC-K never derives
}
```

Equality is **bitwise** over the struct (modulo `_pad`). No tolerance.
No semantic comparison.

### 2.1 Hash determinism

* Both implementations produce the same hash for identical input bytes
  *across runs of the same binary* (K1). Cross-impl parity (kernel vs
  host) is **not required** — replay reconstructs the original intent
  stream from `intent_id`s in audit, not from re-hashed keys.
* `XXX-blake3`: kernel currently uses FNV-1a-64. Host uses real
  Blake3-64. Kernel will swap to Blake3 in P0.3.1 (same milestone as
  `audit_persist` Blake3 swap).

## 3. IntentBatch (ABI)

```rust
#[repr(C)]
pub struct IntentBatch {
    pub key:      CanonicalKey,
    pub count:    u32,
    pub first_ts: u64,
    pub last_ts:  u64,
    pub first_id: [u8; 16],   // first intent id folded into batch
}
```

`first_id` lets a kernel-only consumer find the originating
`BatchCreate` audit frame without scanning. The full member list is
recovered from `BatchIncrement` frames.

## 4. RunQueue semantics

* Linear ordered vector of `IntentBatch`. Insertion order = arrival
  order of the **first** intent of each new key.
* `fold_or_insert` walks the queue head→tail; on the first key match
  increments `count` and updates `last_ts`. No reordering.
* No cross-key merging. No splitting. No eviction in P0.3.

## 5. Audit format additions

| FrameKind        | Code | Payload (≤ 62 bytes)                                    |
|------------------|------|---------------------------------------------------------|
| `BatchCreate`    | 0x1B | `intent_id[16] ‖ a,s,p,_pad ‖ name_h[8] ‖ params_h[8] ‖ lock_key[8] ‖ ts[8]` (52 B) |
| `BatchIncrement` | 0x1C | `intent_id[16] ‖ batch_idx[4] ‖ ts[8]` (28 B)           |
| `BatchExpand`    | 0x1D | replay-only synthetic frame; payload = `batch_idx[4] ‖ count[4]` |

Audit total ordering (single global SEQ in `audit.rs`) gives the
canonical replay order: walk frames, collect every
`BatchCreate`+`BatchIncrement` in seq order, emit
`(intent_id, key, ts)` tuples = original intent stream.

## 6. Replay equivalence theorem

Let `S = [i₁, i₂, …, iₙ]` be a sequence of intents accepted by the
arbiter (raw). Let `B = fold(S)` be the batch sequence produced by
folding `S`. Let `A = audit_log_of(fold(S))`. Then:

```
expand(A) = S
```

Proof sketch: `fold` emits exactly one audit frame per accepted
intent (`BatchCreate` for the first of a key, `BatchIncrement`
thereafter). Each frame carries the full `intent_id` and `ts`. `expand`
walks frames in seq order and reconstructs `(id, ts, key)` 1:1 with
`S`. ∎

## 7. Failure modes (forbidden — must not compile or must trip an invariant)

| Forbidden behaviour             | Detection                                                |
|---------------------------------|----------------------------------------------------------|
| semantic / fuzzy grouping       | bitwise `CanonicalKey` PartialEq — no tolerance          |
| probabilistic hashing           | seeded deterministic hasher; tests assert byte equality  |
| unordered batch expansion       | `expand` follows audit `seq` strictly                    |
| cross-lock-key merging          | `fold_or_insert` rejects merge unless full key matches   |
| modifying arbiter logic         | arbiter calls precede `ahtc_k::*` in a single function   |
| dropping `BatchIncrement` frame | `invariants::trip(InvariantCode::AuditDrop)` on overflow |

## 8. Performance contract

Under N intents with K distinct canonical keys (`K ≤ N`):

| Metric                          | Before AHTC-K | After AHTC-K |
|---------------------------------|---------------|--------------|
| scheduler enqueue calls         | N             | K            |
| audit frames                    | N             | N            |
| RunQueue scan length            | O(N)          | O(K)         |

Test 3 in `tests/ahtc_k_tests.rs` asserts ≥ 10× reduction at 70 %
duplication.

## 9. P0.3 acceptance map (delta)

| Item                                 | Status | Evidence                                  |
|--------------------------------------|--------|-------------------------------------------|
| AHTC-K canonicalization              | ✅     | `crates/qratum-arbiter/src/ahtc_k.rs`     |
| AHTC-K kernel layer                  | ✅     | `src/kernel/ahtc_k/`                      |
| determinism test                     | ✅     | `ahtc_k_tests::determinism_*`             |
| replay equivalence test              | ✅     | `ahtc_k_tests::replay_equivalence_*`      |
| scheduler reduction test (≥ 10×)     | ✅     | `ahtc_k_tests::scheduler_reduction_10x`   |
| arbitration invariance test          | ✅     | `ahtc_k_tests::arbitration_invariance`    |

Deferred to P0.3.1 (named):

| Item                                          | Why                                                        |
|-----------------------------------------------|------------------------------------------------------------|
| Kernel hash → real Blake3-64                  | shared swap window with `audit_persist::weak_chain`        |
| Replace `sched::spawn_with_verdict` callsites | requires intent-driven scheduler; AHTC-K interface frozen  |
