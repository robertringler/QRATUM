# QRATUM Arbiter — Formal State Machine (P0.3)

> Authoritative model. The kernel arbiter (`os/qratum-os/src/kernel/intent.rs`)
> and the host-side reference implementation
> (`crates/qratum-arbiter/src/state_machine.rs`) **MUST** produce identical
> verdict sequences for any identical input sequence.

## 1. Sets

| Symbol | Meaning                                                        |
|--------|----------------------------------------------------------------|
| `I`    | Intent space — set of all `IntentEnvelope` values              |
| `L`    | Lock-table space — set of all `LockState` values               |
| `T`    | Tick / monotonic counter — `u64`                                |
| `C`    | Authority class — `enum { User=1, Console=2, Service=3, System=4 }` |
| `S`    | Scope class — `enum { Process=1, Session=2, Machine=3, Fleet=4 }` |
| `V`    | Verdict space — `enum { Accept(reason), Deny(reason) }`         |

### `LockState` (`L`)

```
L = {
    inflight        : u32,        // 0..=MAX_INFLIGHT
    holder_authority: Option<C>,
    last_console_tick: u64,        // for dominance window
    override_count   : u32,        // for thrash detection
    last_override_tick: u64,
}
```

### `IntentEnvelope` (`i ∈ I`)

```
i = {
    id        : [u8; 16],
    kind      : u8,          // domain-specific, see kernel/intent.rs
    authority : C,
    scope     : S,
    tick      : u64,         // monotonic, strictly increasing across submits
    payload_h : [u8; 32],    // Blake3 hash of payload — content-addressed
}
```

## 2. Transition function

```
V : (I × L × T) → (Verdict × L')
```

Pure function. **No hidden state.** The kernel's arbiter must call it
with a snapshot of `L` taken under the same lock as the verdict's
emission, so that `L'` replaces `L` atomically.

### 2.1 Decision pseudocode

```text
fn arbitrate(i: Intent, L: LockState, t: Tick) -> (Verdict, LockState):
    L' = L

    // R1 — schema
    if i.kind == 0 || i.authority == 0 || i.scope == 0:
        return (Deny(DenySchema), L)

    // R2 — queue full
    if L.inflight >= MAX_INFLIGHT:
        return (Deny(DenyQueueFull), L)

    // R3 — dominance: a Console authority cannot hold the table for >
    // DOMINANCE_WINDOW ticks against a higher authority.
    if i.authority == Console
        && L.holder_authority is Some(h)
        && h > Console
        && (t - L.last_console_tick) > DOMINANCE_WINDOW:
        return (Deny(DenyConsoleDominance), L)

    // R4 — autonomy block: System cannot be preempted by lower authorities.
    if L.holder_authority == Some(System) && i.authority < System:
        return (Deny(DenyAutonomyBlock), L)

    // R5 — override thrash: more than OVERRIDE_LIMIT overrides within
    // OVERRIDE_WINDOW ticks is rejected.
    if i.authority == System
        && L.holder_authority is Some(h) && h != System
        && L.override_count >= OVERRIDE_LIMIT
        && (t - L.last_override_tick) < OVERRIDE_WINDOW:
        return (Deny(DenyOverrideThrash), L)

    // ----- Accept paths -----
    L'.inflight = L.inflight + 1
    if L.holder_authority is None:
        L'.holder_authority = Some(i.authority)
        return (Accept(AcceptFree), L')

    if L.holder_authority == Some(i.authority):
        return (Accept(AcceptRefresh), L')

    if i.authority > L.holder_authority.unwrap():
        L'.holder_authority = Some(i.authority)
        if i.authority == System:
            L'.override_count    = L.override_count + 1
            L'.last_override_tick = t
        return (Accept(AcceptAuthority), L')

    // i.authority < holder — rejected by R4 above unless holder is None,
    // so this is unreachable for non-System holders. Defensive Deny.
    return (Deny(DenyAuthorityRegression), L)
```

### 2.2 Constants

| Constant            | Value | Rationale                              |
|---------------------|------:|----------------------------------------|
| `MAX_INFLIGHT`      | 64    | bounded queue, deterministic backpress |
| `DOMINANCE_WINDOW`  | 100   | ticks (1 s @ 100 Hz)                   |
| `OVERRIDE_LIMIT`    | 4     | thrash bound                           |
| `OVERRIDE_WINDOW`   | 50    | ticks                                  |

## 3. Total ordering

Intents are totally ordered by the lex tuple `(tick, id)`. For
identical-tick inputs, the **lower `id`** is processed first. The
kernel guarantees this by sorting submissions per tick under the
`SCHED` lock before invoking `arbitrate`.

## 4. Invariants (machine-checked at runtime)

| INV   | Statement |
|-------|-----------|
| INV-1 | `L.inflight <= MAX_INFLIGHT` always |
| INV-2 | `L.holder_authority is None ⇔ L.inflight == 0` |
| INV-3 | `Accept` strictly increases `L.inflight`; `release_one()` strictly decreases |
| INV-4 | replay of any prefix `(i₁..iₙ)` yields the same `Lₙ` and verdict sequence — **determinism** |
| INV-5 | `arbitrate` is pure: no syscalls, no allocator, no time-of-day reads |

## 5. Conflict resolution rules (summary)

```
priority within same tick:
    System     > Service > Console > User
    same authority → lower `id` wins
    same id      → impossible (id is content + nonce)
```

## 6. Replay equivalence

Given:

* the canonical initial `L₀ = LockState::new()`
* a sequence `Σ = [(i₁, t₁), …, (iₙ, tₙ)]` with `tⱼ` non-decreasing

then for any two implementations `A`, `B` of `arbitrate`:

```
∀ k ≤ n:  A(Σ[..k], L₀) = B(Σ[..k], L₀)
```

This is the equivalence proved by `tests/golden.rs` between the host
reference (`state_machine.rs`) and the kernel's `intent.rs` (verified
in P0.3.1 via a kernel-side replay harness; in P0.3 we check the
reference against itself in two encoding modes — Rust enum vs JSON
round-trip — to ensure the **manifest is canonical**).

## 7. Failure semantics

Every `Deny(reason)` MUST emit exactly one `IntentDenied` audit frame
with the `reason` byte and the intent id. No silent denials.

## 8. Kernel ↔ host parity matrix

| Kernel symbol                        | Host symbol                                  |
|--------------------------------------|----------------------------------------------|
| `intent::submit`                     | `state_machine::arbitrate`                   |
| `intent::release_one`                | `LockState::release`                         |
| `INFLIGHT`, `HOLDER_AUTHORITY`       | `LockState{inflight, holder_authority}`      |
| arbiter denial reasons (u8)          | `DenyReason` enum (same numeric values)      |
| audit frame `IntentSubmitted`        | `ReplayEvent::Submit`                        |
| audit frame `IntentDenied`           | `ReplayEvent::Deny`                          |

The kernel and host MUST share the numeric encoding of `DenyReason`
and `AcceptReason`. Diverging values is an INV-5 violation.
