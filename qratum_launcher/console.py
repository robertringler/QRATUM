"""
QRATUM Console Hub — v1.2 real-time text command surface.

Implements QC_CORE_OS_SPEC_V1_2.md sections 1-3, 7, 8.

Pure-Python, no third-party deps. Late-bound to the launcher
(broker / validator / replay logger / log function) the same way
`autonomy.py` is, to avoid any import cycle.

Public surface:
    parse_command(raw: str) -> ProtoIntent | tuple[bool, str]
    HUB                       - singleton ConsoleHub
    HUB.bind(...)
    HUB.attach(conn, client_id) / detach(...)
    HUB.submit(client_id, raw, allow_debug) -> dict (server reply)
    HUB.on_result(result)     - hooked from BROKER.store_result()
    HUB.snapshot()            - dict for launcher.json
"""

from __future__ import annotations

import json
import re
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

# --- late-bound globals (set by HUB.bind) -----------------------------------
_BROKER = None
_VALIDATE: Callable[[dict], dict] | None = None
_VALIDATION_ERROR: type[BaseException] = Exception
_LOG: Callable[..., None] = print
_PROTOCOL_VERSION = "1.0"

# --- limits (§7) -------------------------------------------------------------
PER_CLIENT_RATE_PER_S = 20
PER_CLIENT_BURST = 5
GLOBAL_RATE_PER_S = 200
MAX_CONSOLE_CLIENTS = 8
MAX_RAW_LEN = 2048
SCROLLBACK_LATENCY = 256  # how many recent samples we keep

PRIORITY_TOKENS = {"!high": "high", "!normal": "normal", "!low": "low"}
RESERVED_PARAM_KEYS = ("_qratum",)  # client cannot set these


# ---------------------------------------------------------------------------
# Parser (§1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtoIntent:
    name: str
    params: dict
    priority: str


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _tokenize(raw: str) -> list[str]:
    """Whitespace-split, but keep "..."-quoted runs intact."""
    out: list[str] = []
    buf: list[str] = []
    in_q = False
    for ch in raw:
        if ch == '"':
            in_q = not in_q
            buf.append(ch)
            continue
        if ch.isspace() and not in_q:
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _coerce_value(tok: str) -> Any:
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok[1:-1]
    low = tok.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    if "," in tok:
        parts = tok.split(",")
        try:
            return [float(p) if "." in p else int(p) for p in parts]
        except ValueError:
            return tok
    try:
        if "." in tok or "e" in low:
            return float(tok)
        return int(tok)
    except ValueError:
        return tok


def parse_command(raw: str) -> tuple[bool, ProtoIntent | str]:
    """Returns (True, ProtoIntent) or (False, error_string)."""
    if not isinstance(raw, str):
        return False, "raw must be a string"
    raw = raw.strip()
    if not raw:
        return False, "empty command"
    if len(raw) > MAX_RAW_LEN:
        return False, "command too long"

    tokens = _tokenize(raw)
    if not tokens:
        return False, "empty command"

    name = tokens[0]
    if not _NAME_RE.match(name):
        return False, f"bad name: {name!r}"

    priority = "normal"
    params: dict[str, Any] = {}
    for tok in tokens[1:]:
        if tok in PRIORITY_TOKENS:
            priority = PRIORITY_TOKENS[tok]
            continue
        if "=" not in tok:
            return False, f"expected key=value: {tok!r}"
        k, v = tok.split("=", 1)
        if not _KEY_RE.match(k):
            return False, f"bad key: {k!r}"
        if k in RESERVED_PARAM_KEYS:
            return False, f"reserved key: {k!r}"
        params[k] = _coerce_value(v)

    return True, ProtoIntent(name=name, params=params, priority=priority)


# ---------------------------------------------------------------------------
# Per-client token bucket
# ---------------------------------------------------------------------------
@dataclass
class _Bucket:
    timestamps: deque[float] = field(default_factory=deque)

    def allow(self, now: float, rate: int) -> bool:
        # sliding 1 s window
        while self.timestamps and now - self.timestamps[0] > 1.0:
            self.timestamps.popleft()
        if len(self.timestamps) >= rate:
            return False
        self.timestamps.append(now)
        return True


# ---------------------------------------------------------------------------
# Console hub
# ---------------------------------------------------------------------------
@dataclass
class _Client:
    client_id: str
    sender: Callable[[dict], bool]  # writes a JSON line to socket; returns ok
    bucket: _Bucket = field(default_factory=_Bucket)
    submitted: int = 0
    rejected: int = 0
    results_routed: int = 0


class ConsoleHub:
    def __init__(self) -> None:
        self.clients: dict[str, _Client] = {}
        # intent_id -> client_id (for routing results back)
        self.routes: dict[str, str] = {}
        self._lock = threading.RLock()
        self._global_bucket = _Bucket()
        # latency tracking (ms) — submit -> first result
        self._submit_ts: dict[str, float] = {}
        self._latencies: deque[float] = deque(maxlen=SCROLLBACK_LATENCY)
        self.rate_limited = 0
        self._log_path = None
        self._log_fp = None
        self._log_lock = threading.Lock()

    # -- bind -----------------------------------------------------------------
    def bind(
        self,
        broker,
        validate_fn,
        validation_err,
        log_fn,
        console_log_path,
        protocol_version: str = "1.0",
    ) -> None:
        global _BROKER, _VALIDATE, _VALIDATION_ERROR, _LOG, _PROTOCOL_VERSION
        _BROKER = broker
        _VALIDATE = validate_fn
        _VALIDATION_ERROR = validation_err
        _LOG = log_fn
        _PROTOCOL_VERSION = protocol_version
        try:
            self._log_path = console_log_path
            self._log_fp = open(console_log_path, "a", encoding="utf-8", buffering=1)
        except OSError:
            self._log_fp = None

    # -- logging --------------------------------------------------------------
    def _journal(self, kind: str, **fields) -> None:
        if self._log_fp is None:
            return
        rec = {"ts": time.time(), "kind": kind, **fields}
        with self._log_lock:
            try:
                self._log_fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
            except OSError:
                pass

    # -- client lifecycle -----------------------------------------------------
    def attach(
        self, sender: Callable[[dict], bool], client_id: str | None = None
    ) -> tuple[bool, str]:
        with self._lock:
            if len(self.clients) >= MAX_CONSOLE_CLIENTS:
                return False, ""
            cid = client_id or uuid.uuid4().hex[:12]
            if cid in self.clients:
                cid = uuid.uuid4().hex[:12]
            self.clients[cid] = _Client(client_id=cid, sender=sender)
        self._journal("attach", client_id=cid)
        return True, cid

    def detach(self, client_id: str) -> None:
        with self._lock:
            self.clients.pop(client_id, None)
            # drop pending routes for this client
            stale = [k for k, v in self.routes.items() if v == client_id]
            for k in stale:
                self.routes.pop(k, None)
                self._submit_ts.pop(k, None)
        self._journal("detach", client_id=client_id)

    # -- submit (§3.1) --------------------------------------------------------
    def submit(self, client_id: str, raw: str, allow_debug: bool = False) -> dict:
        if _BROKER is None or _VALIDATE is None:
            return {"ok": False, "error": "E_NOT_BOUND"}
        with self._lock:
            cli = self.clients.get(client_id)
        if cli is None:
            return {"ok": False, "error": "E_NO_CLIENT"}

        now = time.time()
        # rate limit
        if not self._global_bucket.allow(now, GLOBAL_RATE_PER_S) or not cli.bucket.allow(
            now, PER_CLIENT_RATE_PER_S
        ):
            with self._lock:
                cli.rejected += 1
                self.rate_limited += 1
            self._journal("rate_limit", client_id=client_id, raw=raw)
            return {"ok": False, "error": "E_RATE"}

        ok, parsed = parse_command(raw)
        if not ok:
            with self._lock:
                cli.rejected += 1
            self._send(
                cli, {"type": "console_error", "error": "E_PARSE", "detail": parsed, "raw": raw}
            )
            self._journal("parse_error", client_id=client_id, raw=raw, detail=parsed)
            return {"ok": False, "error": "E_PARSE", "detail": parsed}

        proto: ProtoIntent = parsed  # type: ignore[assignment]

        # debug-namespace gate (§7)
        ns = proto.name.split(".", 1)[0]
        if ns == "debug" and not allow_debug:
            with self._lock:
                cli.rejected += 1
            self._send(
                cli,
                {
                    "type": "console_error",
                    "error": "E_DENIED",
                    "detail": "debug namespace requires --allow-debug",
                    "raw": raw,
                },
            )
            return {"ok": False, "error": "E_DENIED"}

        # build envelope
        params = dict(proto.params)
        params["_qratum"] = {"source": "ui_console", "client_id": client_id}
        envelope = {
            "v": _PROTOCOL_VERSION,
            "type": "intent",
            "id": uuid.uuid4().hex,
            "name": proto.name,
            "params": params,
            "priority": proto.priority,
        }
        try:
            envelope = _VALIDATE(envelope)
        except _VALIDATION_ERROR as e:
            with self._lock:
                cli.rejected += 1
            self._send(
                cli, {"type": "console_error", "error": "E_VALIDATE", "detail": str(e), "raw": raw}
            )
            self._journal("validate_error", client_id=client_id, raw=raw, detail=str(e))
            return {"ok": False, "error": "E_VALIDATE", "detail": str(e)}

        intent_id = envelope["id"]
        with self._lock:
            cli.submitted += 1
            self.routes[intent_id] = client_id
            self._submit_ts[intent_id] = now

        delivered, _drops = _BROKER.publish(envelope)
        # echo to originating client
        self._send(cli, {"type": "console_intent", "intent": envelope, "delivered": delivered})
        self._journal(
            "submit",
            client_id=client_id,
            intent_id=intent_id,
            name=proto.name,
            priority=proto.priority,
            delivered=delivered,
        )
        return {"ok": True, "id": intent_id, "delivered": delivered}

    # -- result hook (§3.2) ---------------------------------------------------
    def on_result(self, result: dict) -> None:
        """Called by the launcher after BROKER.store_result()."""
        rid = result.get("id")
        if not rid:
            return
        with self._lock:
            cid = self.routes.pop(rid, None)
            ts0 = self._submit_ts.pop(rid, None)
            if cid is None:
                return
            cli = self.clients.get(cid)
            if cli is None:
                return
            if ts0 is not None:
                latency_ms = (time.time() - ts0) * 1000.0
                self._latencies.append(latency_ms)
            cli.results_routed += 1

        rtype = result.get("type", "result")
        frame_kind = "console_ack" if rtype == "ack" else "console_result"
        self._send(cli, {"type": frame_kind, "result": result})
        self._journal("result", client_id=cid, intent_id=rid, ok=result.get("ok"))

    # -- broadcast log --------------------------------------------------------
    def broadcast_log(self, level: str, msg: str) -> None:
        with self._lock:
            clients = list(self.clients.values())
        for c in clients:
            self._send(c, {"type": "console_log", "level": level, "msg": msg})

    # -- snapshot (§8) --------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            submitted = sum(c.submitted for c in self.clients.values())
            rejected = sum(c.rejected for c in self.clients.values())
            routed = sum(c.results_routed for c in self.clients.values())
            lat = list(self._latencies)
            return {
                "clients": len(self.clients),
                "submitted": submitted,
                "rejected": rejected,
                "results_routed": routed,
                "p50_latency_ms": (round(statistics.median(lat), 2) if lat else 0.0),
                "p99_latency_ms": (
                    round(sorted(lat)[max(0, int(len(lat) * 0.99) - 1)], 2) if lat else 0.0
                ),
                "rate_limited": self.rate_limited,
            }

    # -- internals ------------------------------------------------------------
    def _send(self, cli: _Client, frame: dict) -> None:
        try:
            cli.sender(frame)
        except Exception as e:  # noqa: BLE001
            _LOG(f"[console] sender error for {cli.client_id}: {e}", level="WARN")


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------
HUB = ConsoleHub()
