"""
QRATUM Arbitration Layer — control authority resolution.

Implements QC_CORE_OS_ARBITRATION_V1.md.

Pure deterministic function of (intent, lock-state) →
verdict(win|lose, reason, evicted_lock?).  Late-bound to the launcher
the same way `autonomy.py` and `console.py` are.

Public surface:
    ARBITER             - singleton ControlArbiter
    ARBITER.bind(...)
    ARBITER.tag(envelope) -> envelope     # ensures authority/scope/priority
    ARBITER.evaluate(envelope) -> Verdict # call from BROKER.publish gate
    ARBITER.snapshot()  -> dict           # for launcher.json
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

# --- ranking tables ---------------------------------------------------------
AUTHORITY_RANK = {"system": 4, "console": 3, "external": 2, "autonomy": 1}
SCOPE_RANK = {"system": 3, "world": 2, "actor": 1}
PRIORITY_RANK = {"override": 4, "high": 3, "normal": 2, "low": 1}

DEFAULT_AUTHORITY = "external"
DEFAULT_SCOPE = "actor"
DEFAULT_PRIORITY = "normal"

DOMINANCE_S = {
    "system": 0.0,
    "console": 5.0,
    "external": 1.0,
    "autonomy": 0.25,
}

OVERRIDE_THRASH_WINDOW_S = 5.0
OVERRIDE_THRASH_LIMIT = 50

_BROKER = None  # not used directly but kept symmetrical
_LOG: Callable[..., None] = print


@dataclass
class Lock:
    key: str
    holder_authority: str
    holder_id: str
    acquired_ts: float
    expires_ts: float
    priority: str
    refresh_count: int = 0


@dataclass(frozen=True)
class Verdict:
    win: bool
    reason: str
    lock_key: str
    holder_authority: str
    intent_id: str


def _intent_authority(env: dict) -> str:
    qm = (env.get("params") or {}).get("_qratum") or {}
    if qm.get("source") == "ui_console":
        return "console"
    if qm.get("goal_id"):
        return "autonomy"
    if qm.get("system"):
        return "system"
    return env.get("authority") or DEFAULT_AUTHORITY


def _intent_scope(env: dict) -> str:
    return env.get("scope") or DEFAULT_SCOPE


def _intent_priority(env: dict, authority: str) -> str:
    """Arbitration priority: derived from envelope priority + override flag.

    The wire envelope's `priority` is the broker queue priority
    (high|normal|low). Arbitration "override" is signalled via
    `params._qratum.override = true` and only honored for system/console.
    """
    qm = (env.get("params") or {}).get("_qratum") or {}
    if qm.get("override") and authority in ("system", "console"):
        return "override"
    return env.get("priority") or DEFAULT_PRIORITY


def _lock_key(env: dict) -> str:
    scope = _intent_scope(env)
    params = env.get("params") or {}
    if scope == "system":
        return "system"
    if scope == "world":
        return f"world:{params.get('region') or 'all'}"
    target = params.get("target") or params.get("actor") or params.get("id")
    if target:
        return f"actor:{target}"
    ns = (env.get("name") or "").split(".", 1)[0] or "_"
    return f"{ns}:_default"


def _sort_key(env: dict) -> tuple:
    a = _intent_authority(env)
    s = _intent_scope(env)
    p = _intent_priority(env, a)
    return (
        -AUTHORITY_RANK.get(a, 0),
        -PRIORITY_RANK.get(p, 0),
        SCOPE_RANK.get(s, 1),
        float(env.get("ts") or 0.0),
        str(env.get("id") or ""),
    )


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------
class ControlArbiter:
    def __init__(self) -> None:
        self.locks: dict[str, Lock] = {}
        self._lock = threading.RLock()
        # counters
        self.evaluated = 0
        self.winners = 0
        self.losers = 0
        self.conflicts = 0
        self.override_thrash = 0
        self.by_winner_authority: dict[str, int] = dict.fromkeys(AUTHORITY_RANK, 0)
        self.by_loser_authority: dict[str, int] = dict.fromkeys(AUTHORITY_RANK, 0)
        # thrash detection: per-(authority,key) recent acquisition timestamps
        self._acquisitions: dict[tuple[str, str], deque[float]] = {}
        # journal
        self._fp = None
        self._fp_lock = threading.Lock()

    def bind(self, broker, log_fn, journal_path) -> None:
        global _BROKER, _LOG
        _BROKER = broker
        _LOG = log_fn
        try:
            self._fp = open(journal_path, "a", encoding="utf-8", buffering=1)
        except OSError:
            self._fp = None

    def _journal(self, kind: str, **fields) -> None:
        if self._fp is None:
            return
        rec = {"ts": time.time(), "kind": kind, **fields}
        with self._fp_lock:
            try:
                self._fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
            except OSError:
                pass

    # -- tagging (Agent 3) ---------------------------------------------------
    def tag(self, env: dict) -> dict:
        """Inject authority/scope defaults; preserve envelope priority.

        Note: envelope `priority` is broker-lane (high|normal|low). The
        arbitration priority (which may be `override`) lives only in the
        verdict computation, not in the envelope.
        """
        authority = _intent_authority(env)
        scope = _intent_scope(env)
        env["authority"] = authority
        env["scope"] = scope
        if "ts" not in env:
            env["ts"] = time.time()
        return env

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, env: dict) -> Verdict:
        """Return verdict; on win, acquire/refresh lock."""
        env = self.tag(env)
        key = _lock_key(env)
        a = env["authority"]
        a_rank = AUTHORITY_RANK.get(a, 0)
        arb_priority = _intent_priority(env, a)
        now = time.time()

        with self._lock:
            self.evaluated += 1
            held = self.locks.get(key)
            # expire stale lock
            if held and held.expires_ts <= now:
                self._journal(
                    "lock_expire",
                    key=key,
                    holder=held.holder_authority,
                    held_for=now - held.acquired_ts,
                )
                held = None
                self.locks.pop(key, None)

            verdict_win: bool
            reason: str

            if held is None:
                verdict_win = True
                reason = "free"
            else:
                held_rank = AUTHORITY_RANK.get(held.holder_authority, 0)
                if a_rank > held_rank:
                    verdict_win = True
                    reason = "system_override" if a == "system" else "authority"
                elif a_rank == held_rank:
                    # same authority class: arbitrate by sort key vs an
                    # implicit "current holder" intent. Holder wins ties
                    # iff acquired earlier (already true since now > acquired).
                    new_key = _sort_key(env)
                    held_proxy = (
                        -AUTHORITY_RANK[held.holder_authority],
                        -PRIORITY_RANK.get(held.priority, 2),
                        1,
                        held.acquired_ts,
                        held.holder_id,
                    )
                    if new_key < held_proxy:
                        verdict_win = True
                        reason = "priority"
                    else:
                        verdict_win = False
                        reason = (
                            "console_dominance"
                            if a == "console"
                            else ("autonomy_blocked" if a == "autonomy" else "timestamp")
                        )
                else:
                    verdict_win = False
                    reason = "autonomy_blocked" if a == "autonomy" else "authority"

            if verdict_win:
                # thrash check
                if self._is_thrashing(a, key, now):
                    verdict_win = False
                    reason = "override_thrash"
                    self.override_thrash += 1
                else:
                    self._acquire(env, key, a, now)
                    self.winners += 1
                    self.by_winner_authority[a] = self.by_winner_authority.get(a, 0) + 1
            if not verdict_win:
                self.losers += 1
                self.conflicts += 1
                self.by_loser_authority[a] = self.by_loser_authority.get(a, 0) + 1

            holder_auth = self.locks[key].holder_authority if key in self.locks else a

            verdict = Verdict(
                win=verdict_win,
                reason=reason,
                lock_key=key,
                holder_authority=holder_auth,
                intent_id=str(env.get("id") or ""),
            )

        self._journal(
            "intent_arbitration",
            intent_id=verdict.intent_id,
            win=verdict.win,
            reason=reason,
            authority=a,
            scope=env["scope"],
            priority=arb_priority,
            lock_key=key,
            holder=holder_auth,
            name=env.get("name"),
        )
        return verdict

    # -- internals -----------------------------------------------------------
    def _is_thrashing(self, authority: str, key: str, now: float) -> bool:
        bucket = self._acquisitions.setdefault((authority, key), deque())
        cutoff = now - OVERRIDE_THRASH_WINDOW_S
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return len(bucket) >= OVERRIDE_THRASH_LIMIT

    def _acquire(self, env: dict, key: str, authority: str, now: float) -> None:
        prev = self.locks.get(key)
        window = DOMINANCE_S.get(authority, 0.25)
        if window <= 0:
            # system intents do not hold; only evict.
            if prev:
                self._journal(
                    "lock_evict", key=key, evicted=prev.holder_authority, new_holder=authority
                )
                self.locks.pop(key, None)
            return
        qm = (env.get("params") or {}).get("_qratum") or {}
        holder_id = qm.get("client_id") or qm.get("goal_id") or env.get("client_id") or "external"
        if prev and prev.holder_authority == authority and prev.holder_id == holder_id:
            prev.expires_ts = now + window
            prev.refresh_count += 1
            prev.priority = env["priority"]
            self.locks[key] = prev
        else:
            if prev:
                self._journal(
                    "lock_evict", key=key, evicted=prev.holder_authority, new_holder=authority
                )
            self.locks[key] = Lock(
                key=key,
                holder_authority=authority,
                holder_id=holder_id,
                acquired_ts=now,
                expires_ts=now + window,
                priority=env["priority"],
            )
            self._journal(
                "lock_acquire", key=key, holder=authority, holder_id=holder_id, expires_in_s=window
            )
        self._acquisitions.setdefault((authority, key), deque()).append(now)

    # -- maintenance ---------------------------------------------------------
    def sweep(self) -> int:
        """Drop expired locks. Return count expired. Cheap; safe to call ofen."""
        now = time.time()
        n = 0
        with self._lock:
            for k in list(self.locks):
                if self.locks[k].expires_ts <= now:
                    held = self.locks.pop(k)
                    self._journal(
                        "lock_expire",
                        key=k,
                        holder=held.holder_authority,
                        held_for=now - held.acquired_ts,
                    )
                    n += 1
        return n

    # -- snapshot ------------------------------------------------------------
    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            return {
                "evaluated": self.evaluated,
                "winners": self.winners,
                "losers": self.losers,
                "conflicts": self.conflicts,
                "by_winner_authority": dict(self.by_winner_authority),
                "by_loser_authority": dict(self.by_loser_authority),
                "active_locks": len(self.locks),
                "override_thrash": self.override_thrash,
                "locks": [
                    {
                        "key": L.key,
                        "holder": L.holder_authority,
                        "holder_id": L.holder_id,
                        "priority": L.priority,
                        "expires_in_s": round(max(0.0, L.expires_ts - now), 3),
                    }
                    for L in self.locks.values()
                ],
            }


ARBITER = ControlArbiter()
