"""
QRATUM Autonomy Plane — v1.1 closed-loop control.

Implements QC_CORE_OS_SPEC_V2.md sections 1-11 inside the launcher
process. Four cooperating role-threads (Planner, Executor, Critic,
Stabilizer) operate on a shared GoalStore, communicating only via:

    - the GoalStore (lock-protected)
    - the v1.0 IntentBroker (intents/results)

All emitted intents pass through the same IntentValidator as external
ones, so v1.0 invariants survive untouched. Autonomy is opt-in and can
be killed in <=1 loop tick via the KILL switch.

Public surface:
    AUTONOMY        - singleton orchestrator
    submit_goal()   - external entry point
    goal_status()   - introspection
    cancel_goal()   - per-goal cancel
    kill_autonomy() - global kill-switch
    snapshot()      - dict for launcher.json
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Imports from the launcher are deliberately late-bound to avoid a
# cycle at module-import time. The launcher injects them on init.
_BROKER = None        # set by bind()
_VALIDATE = None      # IntentValidator.validate
_VALIDATION_ERROR = Exception
_LOG = print

# --- §10.3 ACL --------------------------------------------------------------
DEFAULT_POLICY = {
    "allow": ["sim", "actor", "ai", "vehicle", "ui", "param"],
    "deny":  ["debug", "qratum"],
    "max_priority": "normal",
}
PRIORITY_ORDER = {"low": 0, "normal": 1, "high": 2}

# --- §9 Guardrails ----------------------------------------------------------
MAX_ACTIVE_GOALS    = 64
MAX_INTENTS_PER_GOAL_PER_S = 20
MAX_INTENTS_GLOBAL_PER_S   = 500
RUNAWAY_HARD_CAP   = 128       # max intents over a goal's lifetime
CPU_KILL_PCT       = 25.0      # §10.2 trigger 3
CPU_KILL_WINDOWS   = 5         # consecutive 1 s windows


# ---------------------------------------------------------------------------
# Schema (§1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Predicate:
    kind: str
    path: str          # JSON-pointer-ish "/a/b/c"
    value: Any = None
    tolerance: float = 0.0


@dataclass(frozen=True)
class Goal:
    id: str
    name: str
    params: dict
    success: tuple[Predicate, ...]
    failure: tuple[Predicate, ...]
    deadline_s: float | None
    max_replans: int
    priority: str
    parent_id: str | None
    created_ts: float


@dataclass
class GoalState:
    goal: Goal
    status: str = "pending"       # pending|planning|executing|success|failure|G_*
    plan: list[dict] = field(default_factory=list)   # list of proto-intents
    in_flight: dict[str, dict] = field(default_factory=dict)  # intent_id -> meta
    replans: int = 0
    intents_emitted: int = 0
    last_score: float = 0.0
    score_history: deque = field(default_factory=lambda: deque(maxlen=8))
    finalized_ts: float | None = None
    error: str = ""

    def alive(self) -> bool:
        return self.status in ("pending", "planning", "executing")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_pointer(obj: Any, path: str) -> Any:
    """Tiny JSON-pointer-ish lookup. Returns None on miss."""
    if not path or path == "/":
        return obj
    cur = obj
    for part in path.lstrip("/").split("/"):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


def _eval_predicate(pred: Predicate, world: dict, results: dict) -> bool:
    if pred.kind == "intent_ok":
        r = results.get(pred.value)  # value is intent_id
        return bool(r and r.get("ok"))
    val = _json_pointer(world, pred.path)
    if pred.kind == "state_eq":
        return val == pred.value
    if pred.kind == "state_ge":
        try:
            return float(val) >= float(pred.value) - pred.tolerance
        except (TypeError, ValueError):
            return False
    if pred.kind == "state_le":
        try:
            return float(val) <= float(pred.value) + pred.tolerance
        except (TypeError, ValueError):
            return False
    if pred.kind == "custom" and callable(pred.value):
        try:
            return bool(pred.value(world, results))
        except Exception:
            return False
    return False


def _hash_step(goal_id: str, idx: int, name: str, params: dict) -> str:
    h = hashlib.sha1()
    h.update(goal_id.encode())
    h.update(str(idx).encode())
    h.update(name.encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Plan library (§2.1) — recipes shipped in code, deterministic.
# Each recipe is (goal_name) -> callable(goal) -> list[proto_intent].
# ---------------------------------------------------------------------------
PLAN_LIBRARY: dict[str, Callable[[Goal], list[dict]]] = {}


def recipe(name: str):
    def _wrap(fn):
        PLAN_LIBRARY[name] = fn
        return fn
    return _wrap


@recipe("convoy.demo")
def _convoy_demo(goal: Goal) -> list[dict]:
    n = int(goal.params.get("count", 3))
    return [
        {"name": "actor.spawn",
         "params": {"kind": "truck", "where": [0, i * 500, 100]}}
        for i in range(n)
    ] + [
        {"name": "sim.set_speed", "params": {"value": 1.0}},
    ]


@recipe("self_test.ping")
def _selftest_ping(goal: Goal) -> list[dict]:
    return [{"name": "sim.tick", "params": {}}]


def heuristic_decompose(goal: Goal) -> list[dict]:
    """§2.2 fallback. Last-resort: emit a single sim.tick to probe."""
    return [{"name": "sim.tick", "params": {"reason": "heuristic"}}]


# ---------------------------------------------------------------------------
# Compiler (§3) — proto-intent -> validated intent envelope.
# ---------------------------------------------------------------------------
class IntentCompiler:
    def __init__(self, policy: dict) -> None:
        self.policy = policy

    def _allowed(self, name: str) -> bool:
        ns = name.split(".", 1)[0]
        if ns in self.policy.get("deny", []):
            return False
        allow = self.policy.get("allow")
        return ns in allow if allow else True

    def compile(self, goal: Goal, idx: int, proto: dict) -> dict | None:
        name = proto.get("name", "")
        if not self._allowed(name):
            return None
        # clamp priority
        max_pri = self.policy.get("max_priority", "high")
        prio = goal.priority
        if PRIORITY_ORDER[prio] > PRIORITY_ORDER[max_pri]:
            prio = max_pri
        params = dict(proto.get("params") or {})
        meta = params.setdefault("_qratum", {})
        meta["goal_id"] = goal.id
        meta["step_id"] = _hash_step(goal.id, idx, name, params)
        envelope = {
            "v": "1.0",
            "type": "intent",
            "name": name,
            "params": params,
            "priority": prio,
        }
        try:
            return _VALIDATE(envelope)  # type: ignore[misc]
        except _VALIDATION_ERROR as e:  # type: ignore[misc]
            _LOG(f"[autonomy] compile reject '{name}': {e}", level="WARN")
            return None


# ---------------------------------------------------------------------------
# Autonomy logger (§A4) — append-only autonomy.jsonl
# ---------------------------------------------------------------------------
class AutonomyLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        try:
            self._fp = path.open("a", encoding="utf-8", buffering=1)
        except OSError:
            self._fp = None

    def write(self, kind: str, **fields) -> None:
        if self._fp is None:
            return
        rec = {"ts": time.time(), "kind": kind, **fields}
        with self._lock:
            try:
                self._fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Autonomy:
    def __init__(self) -> None:
        self.goals: dict[str, GoalState] = {}
        self._lock = threading.RLock()
        self.policy: dict = dict(DEFAULT_POLICY)
        self.compiler = IntentCompiler(self.policy)
        self.kill = threading.Event()
        self.stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.logger: AutonomyLogger | None = None
        self.world_view: Callable[[], dict] = lambda: {}
        # rate accounting
        self._global_intents_window: deque[float] = deque()
        self._cpu_over_windows = 0
        self._last_cpu_t = time.process_time()
        self._last_wall_t = time.time()
        # counters
        self.completed = {"success": 0, "failure": 0, "killed": 0}
        self.replans_total = 0
        self.intents_emitted_total = 0
        self.cpu_pct_1s = 0.0

    # -- bind -----------------------------------------------------------------
    def bind(self, broker, validate_fn, validation_err, log_fn,
             autonomy_log_path: Path,
             world_view: Callable[[], dict]) -> None:
        global _BROKER, _VALIDATE, _VALIDATION_ERROR, _LOG
        _BROKER = broker
        _VALIDATE = validate_fn
        _VALIDATION_ERROR = validation_err
        _LOG = log_fn
        self.logger = AutonomyLogger(autonomy_log_path)
        self.world_view = world_view
        # load policy if present
        pol_path = autonomy_log_path.parent / "autonomy.policy.json"
        if pol_path.exists():
            try:
                self.policy.update(json.loads(pol_path.read_text("utf-8")))
                self.compiler = IntentCompiler(self.policy)
            except (OSError, json.JSONDecodeError) as e:
                _LOG(f"[autonomy] policy load failed: {e}", level="WARN")

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        roles: list[tuple[str, Callable[[], None], float]] = [
            ("autonomy.executor",   self._executor_tick,   1.0 / 20.0),
            ("autonomy.critic",     self._critic_tick,     1.0 / 10.0),
            ("autonomy.stabilizer", self._stabilizer_tick, 1.0),
        ]
        for name, fn, period in roles:
            t = threading.Thread(target=self._loop, args=(fn, period),
                                 daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def shutdown(self) -> None:
        self.stop.set()

    def _loop(self, fn: Callable[[], None], period: float) -> None:
        while not self.stop.is_set():
            try:
                fn()
            except Exception as e:                      # noqa: BLE001
                _LOG(f"[autonomy] role tick error: {e}", level="ERROR")
            self.stop.wait(period)

    # -- public API -----------------------------------------------------------
    def submit(self, spec: dict) -> tuple[bool, str, str]:
        """Returns (ok, goal_id, error)."""
        try:
            goal = self._build_goal(spec)
        except ValueError as e:
            return False, "", str(e)
        with self._lock:
            if len([g for g in self.goals.values() if g.alive()]) >= MAX_ACTIVE_GOALS:
                return False, "", "G_BUSY"
            # cycle guard (§10.1)
            if goal.parent_id and self._has_cycle(goal):
                return False, "", "G_CYCLE"
            self.goals[goal.id] = GoalState(goal=goal, status="pending")
        if self.logger:
            self.logger.write("goal_submit", goal_id=goal.id, name=goal.name,
                              priority=goal.priority,
                              deadline_s=goal.deadline_s)
        # plan immediately on submitter's thread for low latency
        self._plan(goal.id)
        return True, goal.id, ""

    def cancel(self, goal_id: str) -> bool:
        with self._lock:
            st = self.goals.get(goal_id)
            if not st or not st.alive():
                return False
            self._finalize(st, "G_KILLED")
        return True

    def kill_all(self) -> int:
        self.kill.set()
        n = 0
        with self._lock:
            for st in self.goals.values():
                if st.alive():
                    self._finalize(st, "G_KILLED")
                    n += 1
        if self.logger:
            self.logger.write("kill_switch", count=n)
        return n

    def status(self, goal_id: str | None = None) -> dict:
        with self._lock:
            if goal_id:
                st = self.goals.get(goal_id)
                if not st:
                    return {"ok": False, "error": "not_found"}
                return {"ok": True, "goal": self._serialize(st)}
            return {
                "ok": True,
                "active": [self._serialize(s) for s in self.goals.values() if s.alive()],
                "completed": dict(self.completed),
            }

    def snapshot(self) -> dict:
        with self._lock:
            scores = [s.last_score for s in self.goals.values() if s.score_history]
            stab = [self._stability(s) for s in self.goals.values() if s.score_history]
            return {
                "active_goals": sum(1 for s in self.goals.values() if s.alive()),
                "completed": dict(self.completed),
                "score_p50": statistics.median(scores) if scores else 0.0,
                "score_p99": (sorted(scores)[max(0, int(len(scores) * 0.99) - 1)]
                              if scores else 0.0),
                "intents_emitted": self.intents_emitted_total,
                "replans": self.replans_total,
                "stability_p50": statistics.median(stab) if stab else 1.0,
                "kill_switch_armed": self.kill.is_set(),
                "cpu_pct_1s": round(self.cpu_pct_1s, 2),
                "policy": self.policy,
            }

    # -- internals: goal construction ----------------------------------------
    def _build_goal(self, spec: dict) -> Goal:
        name = spec.get("name", "")
        if not isinstance(name, str) or "." not in name:
            raise ValueError("goal.name required (dot-namespaced)")
        deadline = spec.get("deadline_s")
        if deadline is not None:
            deadline = float(deadline)
        priority = spec.get("priority", "normal")
        if priority not in PRIORITY_ORDER:
            raise ValueError(f"bad priority {priority!r}")
        # clamp by policy
        max_pri = self.policy.get("max_priority", "high")
        if PRIORITY_ORDER[priority] > PRIORITY_ORDER[max_pri]:
            priority = max_pri

        def _preds(seq):
            out = []
            for p in seq or ():
                if not isinstance(p, dict):
                    continue
                out.append(Predicate(
                    kind=p.get("kind", "state_eq"),
                    path=p.get("path", "/"),
                    value=p.get("value"),
                    tolerance=float(p.get("tolerance", 0.0) or 0.0),
                ))
            return tuple(out)

        return Goal(
            id=spec.get("id") or str(uuid.uuid4()),
            name=name,
            params=dict(spec.get("params") or {}),
            success=_preds(spec.get("success")),
            failure=_preds(spec.get("failure")),
            deadline_s=deadline,
            max_replans=int(spec.get("max_replans", 8)),
            priority=priority,
            parent_id=spec.get("parent_id"),
            created_ts=time.time(),
        )

    def _has_cycle(self, goal: Goal) -> bool:
        seen = {goal.id}
        cur = goal.parent_id
        while cur:
            if cur in seen:
                return True
            seen.add(cur)
            with self._lock:
                parent = self.goals.get(cur)
            cur = parent.goal.parent_id if parent else None
        return False

    # -- planner (§2) ---------------------------------------------------------
    def _plan(self, goal_id: str) -> None:
        with self._lock:
            st = self.goals.get(goal_id)
            if not st or not st.alive():
                return
            st.status = "planning"
            goal = st.goal
        recipe_fn = PLAN_LIBRARY.get(goal.name)
        try:
            proto = recipe_fn(goal) if recipe_fn else heuristic_decompose(goal)
        except Exception as e:                          # noqa: BLE001
            _LOG(f"[autonomy] planner error '{goal.name}': {e}", level="ERROR")
            proto = []
        # cap fan-out (§2.2)
        proto = list(proto)[:16]
        # safety: lifetime cap (§10.1)
        if st.intents_emitted + len(proto) > RUNAWAY_HARD_CAP:
            with self._lock:
                self._finalize(st, "G_RUNAWAY")
            return
        with self._lock:
            st.plan = proto
            st.replans += 1
            self.replans_total += 1
            st.status = "executing" if proto else "planning"
        if self.logger:
            self.logger.write("plan", goal_id=goal.id, steps=len(proto),
                              mode=("recipe" if recipe_fn else "heuristic"))
        if not proto:
            # empty plan twice in a row => infeasible
            if st.replans >= 2:
                with self._lock:
                    self._finalize(st, "G_INFEASIBLE")

    # -- executor (§6 role) ---------------------------------------------------
    def _executor_tick(self) -> None:
        if self.kill.is_set() or _BROKER is None:
            return
        # global token bucket
        now = time.time()
        while self._global_intents_window and now - self._global_intents_window[0] > 1.0:
            self._global_intents_window.popleft()
        budget_global = MAX_INTENTS_GLOBAL_PER_S - len(self._global_intents_window)
        if budget_global <= 0:
            return

        candidates: list[tuple[GoalState, dict, int]] = []
        with self._lock:
            for st in self.goals.values():
                if not st.alive() or not st.plan:
                    continue
                # per-goal rate
                if st.intents_emitted and st.intents_emitted >= RUNAWAY_HARD_CAP:
                    continue
                idx = len(st.in_flight) + 0  # counter for hashing
                proto = st.plan.pop(0)
                candidates.append((st, proto, idx))
                if len(candidates) >= budget_global:
                    break

        for st, proto, idx in candidates:
            envelope = self.compiler.compile(st.goal, idx, proto)
            if envelope is None:
                if self.logger:
                    self.logger.write("step_reject", goal_id=st.goal.id,
                                      proto=proto)
                continue
            delivered, drops = _BROKER.publish(envelope)
            with self._lock:
                st.intents_emitted += 1
                self.intents_emitted_total += 1
                st.in_flight[envelope["id"]] = {
                    "name": envelope["name"],
                    "ts": time.time(),
                    "delivered": delivered,
                }
            self._global_intents_window.append(time.time())
            if self.logger:
                self.logger.write("step_emit",
                                  goal_id=st.goal.id,
                                  intent_id=envelope["id"],
                                  name=envelope["name"],
                                  delivered=delivered,
                                  drops=drops)

    # -- critic (§4) ----------------------------------------------------------
    def _critic_tick(self) -> None:
        if _BROKER is None:
            return
        world = self.world_view() or {}
        now = time.time()
        with self._lock:
            for st in list(self.goals.values()):
                if not st.alive():
                    continue
                # collect results for in-flight intents
                results: dict[str, dict] = {}
                completed_ids: list[str] = []
                for iid in list(st.in_flight):
                    res = _BROKER.query_result(iid)
                    if res is not None:
                        results[iid] = res
                        completed_ids.append(iid)
                for iid in completed_ids:
                    st.in_flight.pop(iid, None)
                    if self.logger:
                        self.logger.write("step_result",
                                          goal_id=st.goal.id,
                                          intent_id=iid,
                                          ok=results[iid].get("ok"))

                # predicate evaluation
                fail_hit = any(_eval_predicate(p, world, results)
                               for p in st.goal.failure)
                if fail_hit:
                    self._finalize(st, "failure")
                    continue
                succ_total = len(st.goal.success) or 1
                succ_hit = sum(1 for p in st.goal.success
                               if _eval_predicate(p, world, results))
                coverage = succ_hit / succ_total
                ok_count = sum(1 for r in results.values() if r.get("ok"))
                step_ok_rate = (ok_count / len(results)) if results else 0.0
                score = 0.4 * step_ok_rate + 0.6 * coverage
                st.last_score = score
                st.score_history.append(score)

                # success?
                if succ_hit == succ_total and score >= 0.95:
                    self._finalize(st, "success")
                    continue
                # deadline?
                if st.goal.deadline_s is not None and \
                   (now - st.goal.created_ts) > st.goal.deadline_s:
                    self._finalize(st, "G_TIMEOUT")
                    continue
                # replan cap?
                if st.replans >= st.goal.max_replans and not st.plan and not st.in_flight:
                    self._finalize(st, "G_REPLAN_CAP")
                    continue
                # need replan?
                if not st.plan and not st.in_flight and st.alive():
                    # release lock briefly to call planner
                    pass
        # planning calls reacquire the lock; do them outside the loop.
        with self._lock:
            need_replan = [s.goal.id for s in self.goals.values()
                           if s.alive() and not s.plan and not s.in_flight]
        for gid in need_replan:
            self._plan(gid)

    # -- stabilizer (§5.2) ----------------------------------------------------
    def _stabilizer_tick(self) -> None:
        # CPU accounting
        cpu_now = time.process_time()
        wall_now = time.time()
        dt_wall = max(wall_now - self._last_wall_t, 1e-6)
        dt_cpu = max(cpu_now - self._last_cpu_t, 0.0)
        self.cpu_pct_1s = 100.0 * dt_cpu / dt_wall
        self._last_cpu_t, self._last_wall_t = cpu_now, wall_now
        if self.cpu_pct_1s > CPU_KILL_PCT:
            self._cpu_over_windows += 1
        else:
            self._cpu_over_windows = 0
        if self._cpu_over_windows >= CPU_KILL_WINDOWS:
            _LOG(f"[autonomy] CPU runaway ({self.cpu_pct_1s:.1f}%) — kill", level="ERROR")
            self.kill_all()
            self._cpu_over_windows = 0

        # divergence check (§5.2)
        with self._lock:
            for st in self.goals.values():
                if not st.alive() or len(st.score_history) < 4:
                    continue
                recent = list(st.score_history)[-4:]
                if max(recent) - min(recent) < 0.05 and st.last_score < 0.5:
                    # demote
                    st.goal = Goal(**{**st.goal.__dict__, "priority": "low"})
                    if self.logger:
                        self.logger.write("stabilize_demote", goal_id=st.goal.id)

    # -- finalize -------------------------------------------------------------
    def _finalize(self, st: GoalState, status: str) -> None:
        if not st.alive():
            return
        st.status = status
        st.finalized_ts = time.time()
        bucket = ("success" if status == "success"
                  else "killed" if status.startswith("G_KILLED")
                  else "failure")
        self.completed[bucket] = self.completed.get(bucket, 0) + 1
        if self.logger:
            self.logger.write("goal_finalize",
                              goal_id=st.goal.id,
                              status=status,
                              score=st.last_score,
                              intents=st.intents_emitted)

    # -- helpers --------------------------------------------------------------
    def _stability(self, st: GoalState) -> float:
        if len(st.score_history) < 2:
            return 1.0
        try:
            v = statistics.pvariance(st.score_history)
        except statistics.StatisticsError:
            return 1.0
        return max(0.0, 1.0 - v)

    def _serialize(self, st: GoalState) -> dict:
        g = st.goal
        return {
            "id": g.id,
            "name": g.name,
            "status": st.status,
            "priority": g.priority,
            "replans": st.replans,
            "intents_emitted": st.intents_emitted,
            "in_flight": len(st.in_flight),
            "queued_steps": len(st.plan),
            "score": round(st.last_score, 3),
            "created_ts": g.created_ts,
            "finalized_ts": st.finalized_ts,
            "error": st.error,
        }


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------
AUTONOMY = Autonomy()
