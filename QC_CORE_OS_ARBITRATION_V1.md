# QRATUM CORE OS — Control Arbitration System v1.0

**Status:** ratified by 12-agent synthesis swarm (arbitration layer)
**Supersedes:** none — extends v1.0 (wire), v1.1 (autonomy), v1.2 (console)
**Authority:** Agent 12 (System Synthesis Engineer)

A first-class arbitration layer that decides **who is allowed to mutate
the simulation at any moment**. Sits between `BROKER.publish()` and
fan-out to subscribers, so no intent reaches UE without a recorded
arbitration verdict.

---

## 0. Invariants (additive)

| # | Invariant | Owner |
|---|-----------|-------|
| R1 | Every intent carries `authority`, `scope`, `priority` — defaulted by the compiler, never blank | A1 + A3 |
| R2 | Arbitration is a **pure deterministic function** of (authority, scope, priority, ts, id) | A2 + A9 |
| R3 | Locks are visible in `launcher.json` and never silently extended | A6 + A10 |
| R4 | Losing intents produce a synthetic `result` frame so callers (console, autonomy critic) see a verdict, never silence | A7 + A8 |
| R5 | Arbitration cannot starve autonomy: console dominance windows expire | A11 |
| R6 | Arbitration decisions are logged to `arbitration.jsonl` with full inputs → identical replay reproduction | A9 + A10 |
| R7 | Arbitration runs **inside the broker publish path** — bypass is structurally impossible | A8 |

---

## 1. Authority Model (Agent 1)

### 1.1 Authorities (high → low)

| value      | rank | source |
|------------|------|--------|
| `system`   | 4    | failsafe / launcher itself (Ctrl+Alt+Q, kill-switch) |
| `console`  | 3    | UI REPL (`source=ui_console`) |
| `external` | 2    | qctl `push`, third-party APIs |
| `autonomy` | 1    | planner / critic (v1.1) |

System-issued intents are reserved (`debug.*`, `qratum.*`); they bypass
no checks but always win.

### 1.2 Scopes (broad → narrow)

| value     | rank | meaning |
|-----------|------|---------|
| `system`  | 3    | global launcher state |
| `world`   | 2    | environment-level (`/world/...`) |
| `actor`   | 1    | a single entity (`/actor/<id>/...`) |

Narrower scopes win ties because they carry less collateral risk.

### 1.3 Priority

`override` (4) > `high` (3) > `normal` (2) > `low` (1)

`override` requires `authority ∈ {system, console}`. Anything else is
clamped to `high`.

### 1.4 Default tags by source

| source                  | authority | default scope | default priority |
|-------------------------|-----------|---------------|------------------|
| console (`ui_console`)  | console   | actor         | normal           |
| autonomy (`goal_id` set)| autonomy  | actor         | normal           |
| qctl push, others       | external  | actor         | normal           |
| failsafe / kill-switch  | system    | system        | override         |

Compiler may override via explicit envelope fields; defaults apply only
when the field is absent.

---

## 2. Arbitration Function (Agent 2)

```
verdict(intent, world) =
    let key(i) = ( -authority_rank(i),     # higher first
                   -priority_rank(i),
                   +scope_rank(i),         # narrower (lower rank) first
                   +i.ts,                  # earlier wins
                   +i.id )                 # lex tiebreak
    among intents with overlapping target,
        winner = argmin key(i)
        losers = others
```

The total order is **strict** and **deterministic**. No randomness, no
clock skew sensitivity beyond the 1 ms tick cap (intents within 1 ms of
each other tie-break by id, never by arrival jitter).

### 2.1 Target overlap

Two intents *overlap* iff their compiled `lock_key` strings are equal.
The compiler computes:

```python
lock_key =
    if scope == "actor":  f"actor:{params.target}"
    if scope == "world":  f"world:{params.region or 'all'}"
    if scope == "system": "system"
```

Intents whose params do not include a target collapse to a synthetic
key derived from the intent name's first dotted segment
(`f"{ns}:_default"`), so two `vehicle.set_throttle` calls without an
explicit `target` still arbitrate against each other.

---

## 3. Locks (Agent 6)

A **lock** records the current owner of a `lock_key`:

```python
@dataclass
class Lock:
    key: str
    holder_authority: str
    holder_id: str            # client_id (console) or goal_id (autonomy) or "external"
    acquired_ts: float
    expires_ts: float         # acquired_ts + dominance_window_s
    priority: str
```

### 3.1 Acquisition

When a winning intent is published, it acquires (or refreshes) the
lock. The dominance window is:

| holder    | window |
|-----------|--------|
| `system`  | 0 (single-tick, never holds) |
| `console` | 5.0 s (configurable) |
| `external`| 1.0 s |
| `autonomy`| 0.25 s |

### 3.2 Eviction rules

A lower-rank authority is **denied** the lock if a higher-rank holder
exists and `now < expires_ts`. A higher-rank authority **always** evicts.
Same-rank authorities resolve by §2 ordering.

### 3.3 Visibility

Active locks appear in `launcher.json.arbitration.locks` and are
journaled (`lock_acquire`, `lock_evict`, `lock_expire`) in
`arbitration.jsonl`.

---

## 4. Pipeline Position (Agent 8)

```
producer → IntentValidator → BROKER.publish
                                  │
                                  ▼
                          ARBITER.evaluate
                            ┌─────┴─────┐
                          win          lose
                            │            │
                  fan-out to subs   synthetic result
                                    (E_ARBITRATION_LOSS)
                                    posted to BROKER.store_result
```

Replay determinism: `replay.jsonl` already records each `intent` and
`result` in publish order. With arbitration logged in
`arbitration.jsonl` and the arbitration function pure over
`(intent, locks_snapshot)`, replaying intents in the same order
reproduces identical winners.

---

## 5. Conflict Detection (Agent 7)

The arbiter examines, for each incoming intent, the active lock for
its `lock_key`. A *conflict* is recorded if a held lock blocks
acquisition. The conflict report carries:

```jsonc
{
  "kind": "intent_arbitration",
  "winner_id": "<id>",
  "loser_id":  "<id>",
  "winner_authority": "console",
  "loser_authority":  "autonomy",
  "scope": "actor",
  "lock_key": "actor:truck_17",
  "reason": "console_dominance"
}
```

`reason` is one of: `authority`, `priority`, `scope_narrower`,
`timestamp`, `id_tiebreak`, `console_dominance`, `system_override`,
`autonomy_blocked`.

---

## 6. Safeguards (Agent 11)

* **Anti-starvation**: a lock holder that has not emitted a winning
  intent in `2 × dominance_window` is dropped. Autonomy regains control.
* **Anti-deadlock**: locks have hard wall-clock expiry; nothing waits
  for a lock — losers are rejected, not queued.
* **Override loop guard**: if the same `(authority, lock_key)` has been
  acquired > 50× in the last 5 s, further acquisitions in that window
  are rate-limited to 5/s and counted as `override_thrash`.
* **Autonomy notification**: on each loss, the autonomy critic sees the
  synthetic result and records `autonomy_override` against the goal
  whose `goal_id` matches `params._qratum.goal_id`. Goal priority is
  decremented one step (per v1.2 §6).

---

## 7. Observability (Agent 10)

`launcher.json.arbitration`:

```jsonc
{
  "evaluated": 4012,
  "winners":   3987,
  "losers":      25,
  "conflicts":   25,
  "by_winner_authority": {"console": 17, "autonomy": 3970, "external": 0, "system": 0},
  "by_loser_authority":  {"console": 0,  "autonomy": 25,  "external": 0, "system": 0},
  "active_locks": 3,
  "override_thrash": 0,
  "locks": [
    {"key": "actor:truck_17", "holder": "console", "expires_in_s": 4.2}
  ]
}
```

`arbitration.jsonl` records every evaluation and lock event.

---

## 8. Acceptance

1. Two simultaneous `vehicle.set_throttle target=truck_17` intents
   (one console `priority=normal`, one autonomy `priority=high`) → the
   **console** intent wins, autonomy gets `E_ARBITRATION_LOSS`.
2. After 5 s of console silence, the same autonomy intent wins.
3. Two console intents racing same key are ordered by `(ts, id)`
   regardless of arrival thread — replay reproduces identical winner.
4. A `system` intent always evicts every other holder.
5. `override` priority issued by an autonomy goal is clamped to `high`.
6. `arbitration.jsonl` line counts equal `evaluated` in the snapshot.
7. Under a 1 000-intent burst, no intent is silently dropped: every id
   appears in `replay.jsonl` either as a delivered intent or as a
   `result.error == "E_ARBITRATION_LOSS"`.
