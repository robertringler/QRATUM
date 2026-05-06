# QRATUM CORE OS v1.2 — Real-Time Console Interface

**Status:** ratified by 12-agent synthesis swarm (console expansion)
**Supersedes:** none — extends v1.0 (wire) and v1.1 (autonomy)
**Authority:** Agent 12 (System Synthesis Engineer)

A live, low-latency text console embedded in Unreal that turns typed
commands into validated v1.0 intents, dispatches them through the
existing broker, and streams the resulting `ack`/`result` frames back
to UI clients in real time — without bypassing **any** v1.0/v1.1 invariant.

---

## 0. Invariants (additive)

| # | Invariant | Owner |
|---|-----------|-------|
| C1 | Console input MUST go through `IntentValidator` (no exceptions) | Agent 9 |
| C2 | Console-sourced intents are tagged `params._qratum.source="ui_console"` and logged in `replay.jsonl` | Agent 11 |
| C3 | Console clients MUST receive every `ack`/`result` for the intents they originated within ≤100 ms (local) | Agent 7 |
| C4 | Console subscribers do not block the broker, the autonomy loop, or the UE game thread | Agents 4 + 6 |
| C5 | UI rate-limit: 20 commands/s per client, hard cap 200/s aggregate | Agent 9 |
| C6 | `debug.*` namespace requires `--allow-debug` flag at console open time | Agent 9 |
| C7 | Replay is unaffected: replaying `replay.jsonl` reproduces UI intents byte-for-byte | Agent 12 |

---

## 1. Command Grammar (Agent 8)

```
line     := name params? priority?
name     := IDENT ('.' IDENT)+              # at least one dot
params   := (key '=' value)+                # whitespace-separated
key      := IDENT
value    := number | bool | string | tuple
tuple    := number ',' number (',' number)?
priority := '!high' | '!normal' | '!low'
```

Values:

| token        | parsed as            |
|--------------|----------------------|
| `42`, `3.14` | `int` / `float`      |
| `true|false` | `bool`               |
| `1,2,3`      | `list[float]`        |
| anything else (no quotes needed) | `str` |
| `"hello world"` | `str` (quotes preserved with spaces) |

### 1.1 Examples

```
actor.spawn type=truck location=0,0,100
vehicle.set_throttle value=0.7 !high
sim.tick
goal.submit name=convoy.demo count=3
```

### 1.2 Reserved meta-commands (parsed locally, not shipped as intents)

| meta | effect |
|------|--------|
| `:help`     | print grammar & registered namespaces |
| `:clear`    | clear local scrollback |
| `:trace on|off` | toggle latency display |
| `:autonomy off` | maps to `kill_autonomy` control cmd |

---

## 2. Parser → Intent (Agents 2, 3)

The parser produces a **proto-intent**:

```python
{"name": str, "params": dict, "priority": "high|normal|low"}
```

The console compiler:

1. injects `params._qratum = {"source": "ui_console", "client_id": <hex>}`,
2. assigns a fresh UUID `id`,
3. defaults `v="1.0"`, `type="intent"`, `epoch=<current launcher epoch>`,
4. runs `IntentValidator.validate()` — same call site as autonomy and
   external clients,
5. publishes via `BROKER.publish()`.

Failures return `{"ok": false, "error": "<code>", "detail": "<why>"}`
synchronously to the originator over the same socket.

---

## 3. Console Channel (Agent 4)

A new long-lived control command:

```jsonc
{ "cmd": "console_subscribe", "v": "1.0", "client_id": "<hex>" }
```

The launcher replies `{ "ok": true, "client_id": "..." }` and then
streams **newline-delimited JSON frames**:

| frame `type` | when | body |
|--------------|------|------|
| `console_intent` | after a `console_send` is validated and published | the full validated envelope |
| `console_ack`    | UE replied with `ack` for a console intent | `{id, ok, dispatch_ms}` |
| `console_result` | UE replied with `result` for a console intent | full result body |
| `console_error`  | parser/validator rejected input | `{error, detail, raw}` |
| `console_log`    | launcher emits a UI-relevant log line | `{level, msg}` |

### 3.1 Submission

```jsonc
{ "cmd": "console_send", "v": "1.0",
  "client_id": "<hex>", "raw": "actor.spawn type=truck",
  "allow_debug": false }
```

Reply: `{ "ok": true, "id": "<intent-uuid>" }` or
`{ "ok": false, "error": "...", "detail": "..." }`.

### 3.2 Routing rules

* Console hub keeps `client_id → connection`.
* Every result that flows through `BROKER.store_result()` is inspected
  for `params._qratum.source == "ui_console"`; if matched, the hub
  routes a `console_result` frame to the originating `client_id`.
* Heartbeats: hub piggybacks on the launcher's existing `heartbeat`
  timer, sending an empty `console_log` every 5 s so dead clients
  surface fast.

---

## 4. UE5 Widget (Agent 1)

`UQRATUMConsoleWidget` — UMG widget with:

* `UMultiLineEditableTextBox  OutputLog`  (read-only, scrollback 1024 lines)
* `UEditableTextBox           InputLine`  (single line, auto-focused)
* `OnTextCommitted` (Enter) → calls `UQRATUMConsoleClient::Submit(Raw)`
* exposed Blueprint events: `OnIntentEcho`, `OnAck`, `OnResult`, `OnError`

`UQRATUMConsoleClient` — UWorldSubsystem wrapping a thin TCP client
that issues `console_subscribe` once and `console_send` per submit.
Inbound frames are dispatched via `AsyncTask(ENamedThreads::GameThread, ...)`
so the widget never touches the game thread directly from the socket.

The widget never bypasses `UQRATUMBridgeSubsystem`; intents are routed
back through the broker → existing intent client → `QRATUMCommandRouter`.

---

## 5. Threading Model (Agent 6)

```
+-------------+   socket    +------------+    publish    +--------+
| UE Console  | <---------> |  Console   | ------------> | BROKER |
| Widget (GT) |             |  Hub (PY)  |               +---+----+
+-------------+             +------+-----+                   |
        ^                          |                         v
        |                          |                  +-------------+
        |                          +<----  result <---|  IntentClient
        |                                            |   (UE)
        |  GameThread dispatch (AsyncTask)            +-----+-------+
        +<-- console_result/ack stream <-- Console Hub <-----+
```

* No new threads on the UE side beyond the existing intent client.
* The launcher console hub piggybacks on per-connection threads
  spawned by `control_server`; one extra thread per *connected*
  console client, capped to `MAX_CONSOLE_CLIENTS=8`.
* Intent dispatch on UE remains single-MPSC drained on the game
  thread inside `UQRATUMBridgeSubsystem::Tick`. Console intents share
  the same path — no parallel dispatcher.

---

## 6. Conflict with Autonomy (Agent 10)

A goal-emitted intent and a console intent that target the **same
predicate path** (e.g. `/actor/truck0/throttle`) within a 200 ms
window: the console intent wins.

* Autonomy critic observes the override (it sees the result frame),
  records `autonomy_override` in `autonomy.jsonl`, and lowers the
  affected goal's priority by one step.
* No deadlock can occur: console intents do not wait on autonomy
  state; autonomy goals retry naturally on the next critic tick.

---

## 7. Safety (Agent 9)

* Per-client token bucket: 20 cmd/s, burst 5.
* Aggregate cap: 200 cmd/s; surplus rejected with `E_RATE`.
* `debug.*` is gated by both `allow_debug` flag (per submit) **and**
  the autonomy ACL (`policy.allow`).
* Hub strips any client-supplied `seq`, `id`, or
  `params._qratum.{goal_id,step_id}` before validation.

---

## 8. Observability (Agent 11)

`launcher.json.console`:

```jsonc
{
  "clients": 1,
  "submitted": 412,
  "rejected": 7,
  "results_routed": 405,
  "p50_latency_ms": 12.3,
  "p99_latency_ms": 64.1,
  "rate_limited": 0
}
```

Every console event also lands in `console.jsonl` next to `replay.jsonl`.

---

## 9. Acceptance (v1.2)

A build passes when **all** of:

1. `qctl console` opens a REPL; typing `sim.tick` round-trips an
   `ack`/`result` echo within 100 ms on localhost.
2. `replay.jsonl` shows the console intent with
   `params._qratum.source == "ui_console"`.
3. Submitting `actor.spawn` 100×/s gets clamped to 20/s with the
   surplus reported as `E_RATE` — no broker drops.
4. With autonomy active and a `convoy.demo` goal in flight, a console
   `vehicle.set_throttle value=0` for a goal-driven actor is logged as
   an `autonomy_override` and demotes the goal's priority.
5. `console.jsonl` + `replay.jsonl` together let a fresh launcher
   reproduce the live session byte-for-byte (replay determinism).
