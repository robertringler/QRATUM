# QRATUM CORE OS — Canonical Stability + Intent Execution Spec v1.0

**Status:** ratified by 12-agent synthesis swarm
**Supersedes:** ad-hoc bridge v0.x
**Authority:** Agent 12 (System Synthesis Engineer)

This document is the single source of truth for the QRATUM intent
execution stack. All other agent outputs are reconciled here.

---

## 0. Architectural Invariants (system-wide rules)

| # | Invariant | Enforced by |
|---|-----------|-------------|
| I1 | Every intent is **validated** before it crosses the launcher boundary | Agent 2 layer (`IntentValidator`) |
| I2 | Every intent is **acknowledged** with a structured response | Agent 3 protocol |
| I3 | Every intent + response is **logged append-only** (replay.jsonl) | Agent 5 logger |
| I4 | Execution is **game-thread only** | Agent 6 dispatch boundary |
| I5 | Socket thread performs **zero gameplay work** | Agent 8 transport |
| I6 | The system **degrades gracefully** — never silent-crashes | Agents 1 + 6 |
| I7 | Replay mode is **deterministic** (same inputs → same outputs) | Agent 5 + 10 |
| I8 | All overflow drops are **observable** (counters in launcher.json) | Agent 11 |

---

## 1. Wire Protocol v1.0 (Agent 8)

All messages are **single-line UTF-8 JSON terminated with `\n`** on
`127.0.0.1:5555`. Maximum line length: **65 536 bytes**.

### 1.1 Message envelope

```jsonc
{
  "v": "1.0",                 // protocol version, required
  "type": "intent" |
          "ack"    |
          "result" |
          "heartbeat" |
          "control",          // required
  "id":  "<uuid-v4>",         // required for intent/ack/result
  "ts":  1714972800.123456,   // sender wall clock
  "seq": 4711,                // monotonic broker sequence (intents only)
  "name": "<intent.name>",    // intent only
  "params": { ... },          // intent only, object or empty
  "ok":   true,               // ack/result only
  "error": "string",          // ack/result error path
  "data":  { ... },           // result payload
  "epoch": 1234,              // optional kernel epoch
  "priority": "high"|"normal"|"low"  // intent only, default "normal"
}
```

### 1.2 Frame types

| `type` | Direction | Purpose |
|--------|-----------|---------|
| `intent` | Launcher → UE | Action request |
| `ack` | UE → Launcher | Validated + accepted into local queue |
| `result` | UE → Launcher | Final outcome of executing the intent |
| `heartbeat` | Both | Keepalive every 5 s; payload `{"v":"1.0","type":"heartbeat","ts":...}` |
| `control` | Client → Launcher | `subscribe`, `push_intent`, `query_result`, etc. |

### 1.3 Reconnect strategy

Exponential backoff: 250 ms → 500 ms → … → 5 s ceiling. On every
reconnect the client MUST re-issue `{"cmd":"subscribe","v":"1.0"}`. If
the server returns `{"ok":false,"error":"version_mismatch"}` the
client logs and exits with code 65.

---

## 2. Intent Validation (Agent 2)

### 2.1 Schema

```python
NAME_RE = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,4}$"
# e.g. "actor.spawn", "sim.tick", "vehicle.set_throttle"

MAX_PARAMS_DEPTH   = 8
MAX_PARAMS_KEYS    = 256
MAX_STRING_LEN     = 8 * 1024
ALLOWED_PARAM_TYPES = (str, int, float, bool, type(None), list, dict)
```

### 2.2 Reject codes

| code | meaning |
|------|---------|
| `E_VERSION` | unknown protocol version |
| `E_SCHEMA`  | missing required field |
| `E_NAME`    | name fails regex / not registered |
| `E_PARAMS`  | params too deep / too many keys / forbidden type |
| `E_SIZE`    | message exceeds 64 KiB |
| `E_RATE`    | per-client rate limit exceeded |
| `E_UNKNOWN` | name validated but no handler registered (UE side) |

Rejected intents NEVER reach UE. Launcher returns
`{"ok":false,"error":"E_NAME","detail":"…"}` to the submitter.

### 2.3 Sanitization

The validator is **non-mutating**. It either accepts the message
verbatim or rejects it. Normalization (case folding, defaulting) is
performed on a **copy** that is then frozen before being placed on
broker queues.

---

## 3. Acknowledgement & Result (Agent 3)

Each successful intent produces **two** UE→launcher messages:

1. `ack` — emitted within 1 frame of router dispatch confirming the
   handler was invoked. Includes dispatch-latency in ms.
2. `result` — emitted when the handler completes (synchronously now,
   asynchronously allowed in v1.1).

If a handler raises, `result` carries `ok:false` and a sanitized
error string (no native pointers, no stack traces).

The launcher maintains an **in-memory result cache** keyed by intent
id, TTL 600 s, capacity 4096. Clients query via
`{"cmd":"query_result","id":"<uuid>"}`.

---

## 4. Queue & Backpressure (Agent 4)

### 4.1 Per-subscriber priority lanes

Each UE subscriber gets three queues (`high`, `normal`, `low`),
capacities `(64, 512, 256)`. The writer thread drains
`high → normal → low` round-robin (4 high : 2 normal : 1 low).

### 4.2 Drop policies

| lane | overflow policy |
|------|-----------------|
| high | reject submission with `E_OVERFLOW` (caller retries) |
| normal | drop **oldest** entry, increment `dropped_normal` |
| low | drop **newest**, increment `dropped_low` |

All drops produce a `replay.jsonl` record so determinism is preserved
in replay mode (the dropped intent still appears in the log with
`dropped: true`).

### 4.3 Per-tick budget on UE side

Game-thread tick may dispatch at most `MaxIntentsPerTick = 32` or
`MaxDispatchUsPerTick = 2000` µs (whichever first). Excess intents
remain in the MPSC queue and run next frame.

---

## 5. Replay & Audit (Agent 5)

### 5.1 Logs

* `%LOCALAPPDATA%/QRATUM/bridge/replay.jsonl` — every intent +
  ack + result + drop, in submission order. Append-only, never
  rotated mid-session.
* `%LOCALAPPDATA%/QRATUM/bridge/state.json` — current world snapshot
  (already published by `UQRATUMBridgeSubsystem`).

### 5.2 Replay mode

When the launcher is started with `--replay <file>`:

1. The control socket rejects all `push_intent` calls.
2. The broker streams entries from the file at recorded `ts` deltas.
3. The result cache is cross-checked against recorded results; any
   divergence raises `DeterminismError` and halts.

### 5.3 Snapshots

Every 1000 intents OR every 60 s the launcher copies `state.json` to
`snapshots/<seq>.json`. Replay can resume from any snapshot.

---

## 6. Router Taxonomy (Agent 7)

Intent names are **dot-namespaced** with depth 2–5. Top-level
namespaces:

| namespace | owner | examples |
|-----------|-------|----------|
| `sim.*`   | simulation core | `sim.tick`, `sim.pause`, `sim.set_speed` |
| `actor.*` | gameplay actors | `actor.spawn`, `actor.destroy`, `actor.move` |
| `vehicle.*` | vehicle systems | `vehicle.set_throttle`, `vehicle.brake` |
| `ai.*`    | NPC + CIIR bridge | `ai.directive`, `ai.set_goal` |
| `ui.*`    | UI / HUD | `ui.toast`, `ui.set_overlay` |
| `debug.*` | dev only, gated | `debug.dump_state`, `debug.teleport` |
| `param.*` | parameter service | `param.set`, `param.reset` |
| `qratum.*` | meta / launcher | `qratum.shutdown`, `qratum.checkpoint` |

`debug.*` requires `QRATUM_ALLOW_DEBUG_INTENTS=1`. Unknown namespaces
are rejected with `E_NAME` at the launcher; unknown leaves with
registered namespace surface `E_UNKNOWN` from UE.

---

## 7. UE Execution Safety (Agent 6)

```cpp
// Pseudocode — see QRATUMCommandRouter::Dispatch
TRACE_CPUPROFILER_EVENT_SCOPE(QRATUM_Dispatch);
const double t0 = FPlatformTime::Seconds();
FQRATUMIntentResult R; R.Id = Intent.Id;
try {
    Handler->ExecuteIfBound(Intent);
    R.bOk = true;
} catch (...) {
    R.bOk  = false;
    R.Error = TEXT("handler_exception");
    UE_LOG(LogQRATUMRouter, Error, TEXT("handler '%s' raised"), *Intent.Name);
}
const double dt_ms = (FPlatformTime::Seconds() - t0) * 1000.0;
EmitResult(R, dt_ms);   // queues a "result" frame back to launcher
```

UE never aborts the process for a handler fault; routing continues.

---

## 8. Performance & Frame Budget (Agent 9)

Per-tick budget enforcement (UE):

```
frame_budget_ms = 16.6 (60 Hz)
intent_dispatch_budget_ms = 2.0  // 12 % of frame
```

Telemetry exposed via `qctl status`:

* `intents.received`, `intents.dispatched`, `intents.dropped`
* `dispatch_latency_p50_ms`, `dispatch_latency_p99_ms`
* `queue_depth_high/normal/low`
* `frame_skips` (frames where budget was exceeded)

---

## 9. CIIR Integration (Agent 10)

CIIR decisions are mapped to intents through a **deterministic
function** `ciir_to_intent(decision) -> Intent`. Constraints:

* CIIR may not bypass the validator.
* CIIR-issued intents carry `priority: "high"` only when the
  ε-gate is satisfied; otherwise `normal`.
* A CIIR loop guard caps emission at **100 intents/sec** per
  CIIR instance; over-cap drops with `E_RATE`.
* Replay re-issues CIIR intents from the log, not from CIIR state,
  so determinism survives model changes.

---

## 10. Observability (Agent 11)

`launcher.json` schema (extended):

```jsonc
{
  "v": "1.0",
  "phase": "RUN",
  "wall_clock": 1714972800.42,
  "supervisor": { "state": "running", "pid": 1234, "uptime_s": 88.7 },
  "control_port": 5555,
  "intent_broker": {
    "subscribers": 1,
    "published": 4711,
    "delivered": 4708,
    "dropped":   { "high": 0, "normal": 2, "low": 1 },
    "replay_log_bytes": 1048576,
    "queue_depth":      { "high": 0, "normal": 3, "low": 0 }
  },
  "ue": {
    "connected": true,
    "frames_observed": 5300,
    "intents_dispatched": 4705,
    "dispatch_latency_p50_ms": 0.12,
    "dispatch_latency_p99_ms": 1.40,
    "last_heartbeat_age_s": 0.7
  }
}
```

A connection is considered **stale** if no heartbeat or message has
been received for 15 s; the launcher then closes the subscriber and
records `subscriber_timeout`.

---

## 11. Roadmap

### v1.0 (this spec — implemented now)
* Protocol v1.0 envelope + version handshake
* Validator (`IntentValidator`)
* Replay log (`replay.jsonl`)
* Priority lanes + drop policies
* Heartbeat (5 s)
* UE handler isolation + `EmitResult`
* Result cache + `query_result`
* Extended `launcher.json` telemetry

### v1.1 (next)
* Async handler results (deferred completion via `result_pending`)
* Snapshot-based replay resume
* Per-namespace ACL tokens
* gRPC sidecar for non-line-protocol clients

### v2.0
* Multi-UE-instance fan-out (multi-shard sim)
* Distributed replay across hosts
* Quorum-based determinism check

### v3.0
* Cryptographically signed intent envelopes
* Hardware-attested execution receipts (TPM/SEV-SNP)
* Public CIIR adapter SDK

---

## 12. Acceptance Criteria

A build passes the v1.0 gate when **all** of:

1. `qctl push --name actor.spawn --params '{...}'` returns `ack`
   within 100 ms and `result` within 1 s.
2. Submitting a malformed name returns `E_NAME` and writes nothing
   to `replay.jsonl`.
3. 1 000 intents at 100 Hz with one frozen handler produce no UE
   crash and a populated `result.error` for the frozen ones.
4. `replay.jsonl` replayed against a clean launcher reproduces the
   identical sequence of `result` frames (within float tolerance).
5. Killing UE mid-stream causes the launcher to mark
   `ue.connected=false` within 15 s and to buffer at most
   `cap(high)+cap(normal)+cap(low)` intents before dropping.
6. `launcher.json` exposes every metric in §10.
