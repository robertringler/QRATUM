# QRATUM CORE OS v2.0 — Closed-Loop Autonomy Spec

**Status:** ratified by 12-agent synthesis swarm (autonomy expansion)
**Supersedes:** none — extends [QC_CORE_OS_SPEC.md](QC_CORE_OS_SPEC.md) v1.0
**Authority:** Agent 12 (System Synthesis Engineer)

v1.0 turned QRATUM into a system where *external* clients ship intents to
UE5 and get deterministic results back. v2.0 lets QRATUM **generate its
own intents from goals** while preserving every v1.0 invariant.

---

## 0. Invariants (additive to v1.0 §0)

| # | Invariant | Owner |
|---|-----------|-------|
| A1 | Goals are **structured** — no freeform NL is executed | Agent 1 |
| A2 | Every autonomously generated intent passes the **same validator** as external ones | Agent 3 |
| A3 | Every control loop has an **explicit termination condition** | Agent 5 |
| A4 | Planner / critic decisions are **logged** to `autonomy.jsonl` | Agent 11 |
| A5 | A **kill-switch** halts all autonomy within one loop tick | Agent 10 |
| A6 | Replay determinism (v1.0 I7) is preserved by replaying **logged intents**, not re-running the planner | Agents 5 + 10 |
| A7 | No agent may bypass the wire protocol; all communication is via `intent` / `result` frames | All |

---

## 1. Goal Schema (Agent 1)

### 1.1 Structure

```python
@dataclass(frozen=True)
class Goal:
    id: str                       # uuid
    name: str                     # dot-namespaced, e.g. "convoy.reach_waypoint"
    parent_id: str | None         # for hierarchy
    params: dict                  # frozen, validator-clean
    success: list[Predicate]      # ALL must hold => success
    failure: list[Predicate]      # ANY holding => failure
    deadline_s: float | None      # wall-clock budget
    max_replans: int = 8          # convergence cap (Agent 5)
    priority: str = "normal"      # propagates to emitted intents
    created_ts: float
```

### 1.2 Predicate

```python
@dataclass(frozen=True)
class Predicate:
    kind: str          # "state_eq" | "state_ge" | "state_le" | "intent_ok" | "custom"
    path: str          # JSON-pointer-ish path into state.json, e.g. "/ue/connected"
    value: Any
    tolerance: float = 0.0
```

Predicates are evaluated against the world snapshot
(`state.json`) plus the result-cache. They MUST be pure
functions of (state, results).

### 1.3 Hierarchy

A goal may declare child goals via `parent_id`. The parent
is satisfied iff **all** children resolve to `success` (AND
semantics). For OR semantics, model with a `composite.any` parent.

### 1.4 Failure modes

| code | meaning |
|------|---------|
| `G_TIMEOUT`   | deadline elapsed |
| `G_REPLAN_CAP`| max_replans reached without success |
| `G_KILLED`    | safety kill-switch activated |
| `G_INFEASIBLE`| planner returned empty plan twice in a row |
| `G_CHILD_FAIL`| a non-tolerable child goal failed |

---

## 2. Planner (Agent 2)

### 2.1 Two modes

| mode | when | output |
|------|------|--------|
| **deterministic** | recipe exists in `plan_library/<goal.name>.yaml` | static intent sequence (replay-stable) |
| **heuristic**     | no recipe | rule-based decomposition + retry budget |

The planner is a **pure function**:

```python
def plan(goal: Goal, world: WorldView, history: list[Result]) -> Plan
```

`Plan` = ordered list of `Step`. Each `Step` is either an
`Intent` (leaf) or a `Goal` (sub-goal that recurses). The
planner NEVER calls the broker directly.

### 2.2 Decomposition rules

* Cap fan-out per replan: **16 steps**.
* No step may carry `priority: high` unless the goal does.
* Steps are tagged `step_id = sha1(goal.id || idx)` for
  idempotent re-issue.

### 2.3 Pseudocode

```python
def plan(goal, world, history):
    if recipe := PLAN_LIBRARY.get(goal.name):
        return recipe.bind(goal.params)
    if goal.name.startswith("composite."):
        return Plan(steps=[Step.subgoal(c) for c in goal.children])
    # heuristic fallback
    return heuristic_decompose(goal, world, history)
```

---

## 3. Intent Generation (Agent 3)

The planner emits **proto-intents**; the
`IntentCompiler` finalises them by:

1. defaulting `v="1.0"`, `type="intent"`,
2. assigning a fresh `id`,
3. inheriting `priority` from the originating goal,
4. running through `IntentValidator.validate()`,
5. tagging `meta.goal_id` and `meta.step_id` (these live in
   `params._qratum`, NOT the envelope, to avoid v1.0 schema drift).

If validation fails the planner records `G_INFEASIBLE_STEP`
and the critic must replan.

---

## 4. Evaluator / Critic (Agent 4)

After every emitted intent's `result` arrives (or its deadline
elapses), the critic computes:

```
step_score   = w1 * intent_ok + w2 * predicate_progress
goal_score   = harmonic_mean(step_scores) * coverage(success_preds)
```

Defaults: `w1=0.4, w2=0.6`. `coverage` is the fraction of
success predicates currently true.

Decision rules:

* `goal_score >= 0.95` AND all success preds hold → **success**
* any failure pred holds → **failure**
* `replan_count >= max_replans` → **G_REPLAN_CAP**
* `now >= deadline` → **G_TIMEOUT**
* otherwise → **continue** (request a replan or wait)

The critic also emits a **stability score** = `1 - var(goal_score
over last 5 ticks)`. If stability < 0.2 the stabilizer
(§6) takes over.

---

## 5. Control Loop (Agent 5)

```
                ┌────────────┐
                │  Goal      │
                └─────┬──────┘
                      ▼
                ┌────────────┐
        ┌──────►│  Planner   │──────┐
        │       └─────┬──────┘      │
        │             ▼             │
        │       ┌────────────┐      │
        │       │  Compiler  │      │
        │       └─────┬──────┘      │
        │             ▼             │
        │       ┌────────────┐      │
        │       │  Broker    │──────┼─► UE5
        │       └─────┬──────┘      │
        │             ▼             │
        │       ┌────────────┐      │
        │       │  Critic    │──────┘  replan / converge
        │       └─────┬──────┘
        │             ▼
        │       ┌────────────┐
        └───────│ Stabilizer │  (only on instability)
                └────────────┘
```

### 5.1 Termination conditions (any one halts)

1. critic → `success`
2. critic → `failure` (terminal predicate)
3. `replan_count > max_replans`
4. `now > deadline`
5. kill-switch fires
6. parent goal terminates

### 5.2 Convergence guard

A goal is **diverging** if `goal_score` moved < 0.05 over the
last `2 * max_replans / 4 = 4` ticks. The stabilizer then:

* freezes new intent emission for 1 s,
* drops the goal to `priority: low`,
* re-asks the planner with `mode="heuristic"` even if a recipe
  existed.

### 5.3 Oscillation prevention

The compiler refuses to emit the same `(name, hash(params))`
within a 2 s window unless the goal explicitly sets
`allow_repeat: true`.

---

## 6. Multi-Agent Orchestration (Agent 6)

Four named in-process roles share the autonomy supervisor:

| role | duty | tick rate |
|------|------|-----------|
| **Planner**   | goal → plan | on demand |
| **Executor**  | plan → broker submissions | 20 Hz |
| **Critic**    | result → score → decision | 10 Hz |
| **Stabilizer**| guards loop health, applies §5.2 | 1 Hz |

### 6.1 Communication

Roles communicate **only** through:

* the goal store (shared, lock-protected `dict[str, GoalState]`)
* the wire protocol (intents/results via the v1.0 broker)

No direct method calls between roles outside of constructor
wiring. This makes every interaction loggable.

### 6.2 Conflict resolution

Two goals targeting overlapping state (same predicate path) and
both `priority="high"` → vote: critic prefers the one with the
higher `goal_score` over the last 3 ticks; ties broken by older
`created_ts`. Loser drops to `normal`.

---

## 7. Distributed UE Execution (Agent 7)

### 7.1 Shards

A **shard** is one UE5 process owning a `world_id`. Each shard
subscribes to the launcher and is registered on
`{"cmd":"subscribe","v":"1.0","world_id":"<id>"}`. Without
`world_id` the shard joins `world_id="default"`.

### 7.2 Routing

The compiler may set `intent.params._qratum.world` to target a
specific shard. The broker maintains
`subs_by_world: dict[str, list[_Subscriber]]` and fan-outs only
to the matching set. If the world has no subscribers the
intent is queued to a **pending world buffer** (cap 1024) until
one connects.

### 7.3 Replication

Cross-shard replication uses a separate, opt-in
`replication.*` namespace. Replicated intents carry
`meta.from_world` so the receiving shard can ignore its own echo.

---

## 8. State Synchronization (Agent 8)

### 8.1 Modes

| mode | guarantee | use |
|------|-----------|-----|
| **strict**   | all shards see `result` before next tick | small worlds, tests |
| **eventual** | merged via vector clock at snapshot boundary | production |

Default: **eventual**. Strict is enabled per-goal via
`params._qratum.consistency: "strict"` and serializes the goal
through a single broker mutex.

### 8.2 Snapshot merge

Every `state.json` carries `vc: dict[world_id, int]`. The
launcher merges by **per-key max** with last-write-wins on
ties (deterministic by `world_id` lexical order).

---

## 9. Performance Guardrails (Agent 9)

| guard | default | enforcement |
|-------|---------|-------------|
| max goals in flight | 64 | reject `submit_goal` with `G_BUSY` |
| intents/sec per goal | 20 | excess deferred to next tick |
| intents/sec system | 500 | global token bucket |
| autonomy CPU budget | 5 % per loop | measured via `time.process_time` |
| broker queue total | sum of lane caps | hard cap; surplus dropped per §4 |

These are exposed in `launcher.json` under `autonomy.*`.

---

## 10. Safety Constraints (Agent 10)

### 10.1 Forbidden patterns

* Any goal whose plan emits more than `max_replans * 16 = 128`
  intents → auto-cancel with `G_RUNAWAY`.
* `debug.*` intents are unreachable from the planner unless the
  goal carries `params._qratum.debug_ack: <token>`.
* Composite cycles (goal A → child B → child A) are detected at
  submission and rejected with `G_CYCLE`.

### 10.2 Kill-switch

Three triggers:

1. `qctl kill_autonomy` → sets `KILL.set()` (one bit).
2. Ctrl+Alt+Q failsafe (already in v1.0) also flips it.
3. `autonomy.cpu_pct > 25 %` for 5 consecutive 1 s windows.

Effect within ≤1 loop tick:

* Executor drains pending steps → no-op.
* Critic finalises every active goal as `G_KILLED`.
* Broker continues to accept *external* intents (kill is
  scoped to autonomy).

### 10.3 ACL

Each top-level namespace can be turned on/off in
`%LOCALAPPDATA%/QRATUM/bridge/autonomy.policy.json`:

```jsonc
{
  "allow": ["sim", "actor", "ai"],
  "deny":  ["debug", "qratum"],
  "max_priority": "normal"
}
```

The compiler clamps emitted priority to `max_priority`.

---

## 11. Observability (Agent 11, expanded over v1.0 §10)

`launcher.json.autonomy`:

```jsonc
{
  "active_goals": 3,
  "completed":   { "success": 42, "failure": 7, "killed": 0 },
  "score_p50":   0.78,
  "score_p99":   0.31,
  "intents_emitted": 1180,
  "replans": 19,
  "stability_p50": 0.91,
  "kill_switch_armed": false,
  "cpu_pct_1s": 3.4,
  "shards": {
    "default": { "connected": true, "queue_depth": {...} },
    "city_a":  { "connected": true, "queue_depth": {...} }
  }
}
```

Extra log file: `%LOCALAPPDATA%/QRATUM/bridge/autonomy.jsonl`
records each `goal_submit | plan | step_emit | step_result |
critic_decision | goal_finalize` with timestamps.

---

## 12. Roadmap & Acceptance

### v1.1 (this spec — implementable now in launcher)

* Goal store, planner (recipe + heuristic), compiler, critic,
  stabilizer
* Multi-agent role threads
* Autonomy log (`autonomy.jsonl`)
* Kill-switch + ACL policy
* `submit_goal`, `goal_status`, `cancel_goal`,
  `kill_autonomy` control commands

### v2.0

* Multi-shard fan-out (§7.2) — UE side adds `world_id` to subscribe
* Eventual-consistency snapshot merge (§8)

### v3.0

* gRPC sidecar, signed envelopes, distributed quorum
  (carried over from v1.0 §11 v3.0)

### Acceptance gate (v1.1)

A build passes when **all** of:

1. `qctl goal --recipe convoy.demo` runs to terminal state
   (`success` or `G_TIMEOUT`) and writes a continuous record in
   `autonomy.jsonl`.
2. `qctl kill_autonomy` finalises every active goal as
   `G_KILLED` within 200 ms.
3. Submitting a goal whose plan would emit more than 128 intents
   is auto-cancelled with `G_RUNAWAY` and zero intents reach UE.
4. Replaying `replay.jsonl` against a clean launcher with
   autonomy disabled reproduces the exact ordered intent stream
   captured during the live autonomy run.
5. `launcher.json.autonomy.cpu_pct_1s` stays below 5 % during
   the 1 000-intent canonical workload.
