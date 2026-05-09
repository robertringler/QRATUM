"""
QRATUM Launcher v2 — supervisor + control hub for Unreal Engine.

Responsibilities (non-negotiable):
  1. Supervise Unreal: start / detect crash / restart / log.
  2. Control interface: localhost JSON-line socket on 127.0.0.1:5555.
  3. Failsafe: Ctrl+Alt+Q kills QRATUM and restores Windows shell.
  4. State awareness: writes launcher status into the QRATUM bridge dir
     (%LOCALAPPDATA%/QRATUM/bridge/launcher.json) so the desktop GUI
     can read UE5 lifecycle state without polling processes.

Integrates with:
  - existing kernel state at %LOCALAPPDATA%/QRATUM/bridge/state.json
  - SoiGame project at <repo>/soi/unreal_bridge/SoiGame.uproject
  - UE 5.7 default install path

CLI:
    python qratum_launcher.py             # launch + supervise
    python qratum_launcher.py --headless  # nullrhi mode (no graphical UE)
    python qratum_launcher.py --no-ue     # supervisor only, no UE spawn
    python qratum_launcher.py --port 5555

Optional deps:
    pip install psutil keyboard
  - psutil:   richer health detection (memory, freeze heuristic)
  - keyboard: global Ctrl+Alt+Q hotkey on Windows (admin not required for user session)
Both are optional; the launcher degrades cleanly without them.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Optional dependencies
# ----------------------------------------------------------------------------
try:
    import psutil  # type: ignore
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    import keyboard  # type: ignore
    HAVE_KEYBOARD = True
except ImportError:
    HAVE_KEYBOARD = False


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
UE_PROJECT_DEFAULT = REPO_ROOT / "soi" / "unreal_bridge" / "IntentOS.uproject"
UE_CMD_DEFAULT = Path(r"C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor-Cmd.exe")


def bridge_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    d = base / "QRATUM" / "bridge"
    d.mkdir(parents=True, exist_ok=True)
    return d


LOG_FILE = bridge_dir() / "launcher.log"
LAUNCHER_STATE_FILE = bridge_dir() / "launcher.json"


# ----------------------------------------------------------------------------
# Logging (thread-safe)
# ----------------------------------------------------------------------------
_log_lock = threading.Lock()


def log(msg: str, *, level: str = "INFO") -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}"
    with _log_lock:
        print(line, flush=True)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Supervisor
# ----------------------------------------------------------------------------
class UnrealSupervisor:
    """Owns the Unreal child process: start, monitor, restart, kill."""

    STATES = ("idle", "starting", "running", "frozen", "crashed", "stopped")

    def __init__(self, editor: Path, project: Path, *, headless: bool, no_ue: bool) -> None:
        self.editor = editor
        self.project = project
        self.headless = headless
        self.no_ue = no_ue
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.restart_count = 0
        self.last_state = "idle"
        self._lock = threading.Lock()

    # -- process lifecycle ---------------------------------------------------
    def launch(self) -> bool:
        if self.no_ue:
            self.last_state = "stopped"
            log("UE5 launch skipped (--no-ue).")
            return False

        if not self.editor.exists():
            log(f"Unreal editor not found: {self.editor}", level="ERROR")
            self.last_state = "crashed"
            return False
        if not self.project.exists():
            log(f"UE project not found: {self.project}", level="ERROR")
            self.last_state = "crashed"
            return False

        argv = [str(self.editor), str(self.project), "-NoSplash", "-log"]
        if self.headless:
            argv += ["-unattended", "-nopause", "-nullrhi"]
        else:
            argv += ["-fullscreen"]

        log(f"Launching Unreal: {' '.join(argv)}")
        try:
            with self._lock:
                self.proc = subprocess.Popen(
                    argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if sys.platform == "win32"
                        else 0
                    ),
                )
                self.started_at = time.time()
                self.restart_count += 1
                self.last_state = "starting"
            return True
        except OSError as e:
            log(f"Failed to launch Unreal: {e}", level="ERROR")
            self.last_state = "crashed"
            return False

    def is_running(self) -> bool:
        with self._lock:
            return self.proc is not None and self.proc.poll() is None

    def kill(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            if not self.proc:
                return
            log("Stopping Unreal...")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log("Unreal didn't terminate; killing.", level="WARN")
                try:
                    self.proc.kill()
                except OSError:
                    pass
            except OSError:
                pass
            self.proc = None
            self.last_state = "stopped"

    # -- health --------------------------------------------------------------
    def state(self) -> str:
        if self.no_ue:
            return "stopped"
        if not self.is_running():
            # crashed if we ever started
            return "crashed" if self.started_at else "idle"
        # Up; classify starting vs running vs frozen.
        age = time.time() - (self.started_at or time.time())
        if age < 10.0:
            return "starting"
        if HAVE_PSUTIL and self.proc is not None:
            try:
                p = psutil.Process(self.proc.pid)
                # Crude freeze heuristic: process exists but uses ~0 CPU and has
                # been up >30 s. UE engine never sits at 0% in normal operation.
                cpu = p.cpu_percent(interval=0.2)
                if age > 30.0 and cpu < 0.1:
                    return "frozen"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return "crashed"
        return "running"

    def status_dict(self) -> dict:
        with self._lock:
            pid = self.proc.pid if self.proc else None
        st = self.state()
        self.last_state = st
        return {
            "state": st,
            "pid": pid,
            "started_at": self.started_at,
            "uptime_s": (time.time() - self.started_at) if self.started_at else 0.0,
            "restart_count": self.restart_count,
            "headless": self.headless,
            "no_ue": self.no_ue,
            "project": str(self.project),
        }


# ----------------------------------------------------------------------------
# State publisher — writes launcher.json atomically every second
# ----------------------------------------------------------------------------
def state_publisher(sup: UnrealSupervisor, stop: threading.Event) -> None:
    target = LAUNCHER_STATE_FILE
    tmp = target.with_suffix(".json.tmp")
    while not stop.is_set():
        try:
            from autonomy import AUTONOMY
            autonomy_snap = AUTONOMY.snapshot()
        except Exception:                               # noqa: BLE001
            autonomy_snap = {}
        try:
            from console import HUB as _CONSOLE_HUB
            console_snap = _CONSOLE_HUB.snapshot()
        except Exception:                               # noqa: BLE001
            console_snap = {}
        try:
            from arbitration import ARBITER as _ARBITER
            _ARBITER.sweep()
            arbitration_snap = _ARBITER.snapshot()
        except Exception:                               # noqa: BLE001
            arbitration_snap = {}
        payload = {
            "v": PROTOCOL_VERSION,
            "phase": "RUN",
            "wall_clock": time.time(),
            "supervisor": sup.status_dict(),
            "control_port": CONTROL_PORT,
            "log_file": str(LOG_FILE),
            "intent_broker": {
                "subscribers": BROKER.subscriber_count(),
                "published":   BROKER.published_count,
                "delivered":   BROKER.delivered_count,
                "dropped":     dict(BROKER.dropped),
                "queue_depth": BROKER.queue_depth(),
                "replay_log_bytes": REPLAY.size_bytes(),
            },
            "autonomy": autonomy_snap,
            "console": console_snap,
            "arbitration": arbitration_snap,
        }
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as e:
            log(f"state publish error: {e}", level="WARN")
        stop.wait(1.0)
    # final HALT marker
    try:
        target.write_text(
            json.dumps({"phase": "HALT", "supervisor": sup.status_dict()}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Watchdog — restarts Unreal on crash; gives up after MAX_RESTARTS within window
# ----------------------------------------------------------------------------
MAX_RESTARTS = 5
RESTART_WINDOW_S = 60.0


def watchdog(sup: UnrealSupervisor, stop: threading.Event) -> None:
    restart_times: list[float] = []
    while not stop.is_set():
        if sup.no_ue:
            stop.wait(2.0)
            continue
        if not sup.is_running():
            now = time.time()
            restart_times = [t for t in restart_times if now - t < RESTART_WINDOW_S]
            if len(restart_times) >= MAX_RESTARTS:
                log(
                    f"crash loop: {MAX_RESTARTS} restarts in {RESTART_WINDOW_S}s — "
                    "watchdog disengaging",
                    level="ERROR",
                )
                stop.wait(30.0)
                restart_times.clear()
                continue
            log("Unreal not running — restarting.", level="WARN")
            sup.launch()
            restart_times.append(now)
        stop.wait(3.0)


# ----------------------------------------------------------------------------
# QRATUM CORE OS v1.0 — protocol, validation, replay, broker
# Refer to QC_CORE_OS_SPEC.md for the canonical spec.
# ----------------------------------------------------------------------------
import queue as _queue
import re
import uuid
from collections import deque

PROTOCOL_VERSION = "1.0"

# §2.1 schema
INTENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,4}$")
ALLOWED_NAMESPACES = {
    "sim", "actor", "vehicle", "ai", "ui", "param", "qratum", "debug",
}
MAX_LINE_BYTES   = 65_536
MAX_PARAMS_DEPTH = 8
MAX_PARAMS_KEYS  = 256
MAX_STRING_LEN   = 8 * 1024
ALLOWED_PARAM_TYPES = (str, int, float, bool, type(None), list, dict)

# §4 priority lanes (per subscriber)
LANE_CAPS = {"high": 64, "normal": 512, "low": 256}
LANE_RATIO = (("high", 4), ("normal", 2), ("low", 1))

# §3 result cache
RESULT_TTL_S    = 600.0
RESULT_CAP      = 4096

# §1 heartbeat + staleness
HEARTBEAT_PERIOD_S   = 5.0
SUBSCRIBER_STALE_S   = 15.0


class ValidationError(Exception):
    """Raised by IntentValidator. Carries a stable error code."""
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class IntentValidator:
    """§2 — strict, non-mutating intent validation."""

    @staticmethod
    def _check_value(v, depth: int = 0) -> None:
        if depth > MAX_PARAMS_DEPTH:
            raise ValidationError("E_PARAMS", "params nesting too deep")
        if not isinstance(v, ALLOWED_PARAM_TYPES):
            raise ValidationError("E_PARAMS", f"forbidden type {type(v).__name__}")
        if isinstance(v, str) and len(v) > MAX_STRING_LEN:
            raise ValidationError("E_PARAMS", "string too long")
        if isinstance(v, dict):
            if len(v) > MAX_PARAMS_KEYS:
                raise ValidationError("E_PARAMS", "too many keys")
            for k, sub in v.items():
                if not isinstance(k, str):
                    raise ValidationError("E_PARAMS", "non-string key")
                IntentValidator._check_value(sub, depth + 1)
        elif isinstance(v, list):
            if len(v) > MAX_PARAMS_KEYS:
                raise ValidationError("E_PARAMS", "list too long")
            for sub in v:
                IntentValidator._check_value(sub, depth + 1)

    @classmethod
    def validate(cls, intent: dict) -> dict:
        """Return a new, frozen-style dict; raise ValidationError on reject."""
        if not isinstance(intent, dict):
            raise ValidationError("E_SCHEMA", "intent must be object")
        v = intent.get("v", PROTOCOL_VERSION)
        if v != PROTOCOL_VERSION:
            raise ValidationError("E_VERSION", f"unsupported version {v}")
        name = intent.get("name")
        if not isinstance(name, str) or not INTENT_NAME_RE.match(name):
            raise ValidationError("E_NAME", f"bad name: {name!r}")
        ns = name.split(".", 1)[0]
        if ns not in ALLOWED_NAMESPACES:
            raise ValidationError("E_NAME", f"unknown namespace {ns!r}")
        if ns == "debug" and os.environ.get("QRATUM_ALLOW_DEBUG_INTENTS") != "1":
            raise ValidationError("E_NAME", "debug.* intents disabled")

        params = intent.get("params", {})
        if not isinstance(params, dict):
            raise ValidationError("E_PARAMS", "params must be object")
        cls._check_value(params)

        priority = intent.get("priority", "normal")
        if priority not in LANE_CAPS:
            raise ValidationError("E_SCHEMA", f"bad priority {priority!r}")

        # Build canonical envelope.
        return {
            "v": PROTOCOL_VERSION,
            "type": "intent",
            "id": intent.get("id") or str(uuid.uuid4()),
            "ts": time.time(),
            "name": name,
            "params": params,
            "priority": priority,
            "epoch": int(intent.get("epoch", 0) or 0),
        }


class ReplayLogger:
    """§5 — append-only JSONL log of every intent + ack + result + drop."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._fp = None
        self._open()

    def _open(self) -> None:
        try:
            self._fp = self.path.open("a", encoding="utf-8", buffering=1)
        except OSError as e:
            log(f"replay log open failed: {e}", level="ERROR")
            self._fp = None

    def write(self, record: dict) -> None:
        if self._fp is None:
            return
        record.setdefault("ts", time.time())
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                self._fp.write(line)
            except OSError as e:
                log(f"replay log write failed: {e}", level="WARN")

    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                finally:
                    self._fp = None


REPLAY = ReplayLogger(bridge_dir() / "replay.jsonl")


class _Subscriber:
    """A connected UE (or other) client with three priority lanes."""

    __slots__ = ("id", "lanes", "last_seen", "delivered")

    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.lanes: dict[str, deque] = {k: deque() for k in LANE_CAPS}
        self.last_seen = time.time()
        self.delivered = 0

    def enqueue(self, lane: str, line: str) -> tuple[bool, str | None]:
        """Returns (accepted, drop_reason). Implements §4.2."""
        dq = self.lanes[lane]
        cap = LANE_CAPS[lane]
        if len(dq) < cap:
            dq.append(line)
            return True, None
        if lane == "high":
            return False, "E_OVERFLOW"
        if lane == "normal":
            dq.popleft()  # drop oldest
            dq.append(line)
            return True, "dropped_oldest_normal"
        # low: drop newest
        return True, "dropped_newest_low"

    def pop(self) -> str | None:
        """Round-robin drain weighted by §4.1 ratio."""
        for lane, weight in LANE_RATIO:
            for _ in range(weight):
                dq = self.lanes[lane]
                if dq:
                    return dq.popleft()
        return None

    def depth(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.lanes.items()}


class IntentBroker:
    """§4 — priority fan-out + replay logging + drop accounting."""

    def __init__(self) -> None:
        self._subs: list[_Subscriber] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._seq = 0
        self.published_count = 0
        self.delivered_count = 0
        self.dropped = {"high": 0, "normal": 0, "low": 0}

        # §3 result cache (id -> (ts, result_dict))
        self._results: dict[str, tuple[float, dict]] = {}
        self._result_order: deque[str] = deque()

    # -- subscriber lifecycle -------------------------------------------------
    def subscribe(self) -> _Subscriber:
        s = _Subscriber()
        with self._cv:
            self._subs.append(s)
        return s

    def unsubscribe(self, s: _Subscriber) -> None:
        with self._cv:
            try:
                self._subs.remove(s)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def queue_depth(self) -> dict[str, int]:
        agg = {"high": 0, "normal": 0, "low": 0}
        with self._lock:
            for s in self._subs:
                for k, v in s.depth().items():
                    agg[k] += v
        return agg

    # -- publish --------------------------------------------------------------
    def publish(self, validated: dict) -> tuple[int, list[str]]:
        """Fan-out a *validated* intent. Returns (delivered, drop_codes)."""
        with self._cv:
            self._seq += 1
            validated = dict(validated)
            validated["seq"] = self._seq

        # ---- arbitration gate (QC_CORE_OS_ARBITRATION_V1.md) -----------
        try:
            from arbitration import ARBITER as _ARBITER
            validated = _ARBITER.tag(validated)
            verdict = _ARBITER.evaluate(validated)
        except Exception:                               # noqa: BLE001
            verdict = None

        line = json.dumps(validated, separators=(",", ":")) + "\n"
        priority = validated.get("priority", "normal")

        REPLAY.write({"kind": "intent", **validated})

        if verdict is not None and not verdict.win:
            # Losing intents do not reach UE. Synthesize a result so the
            # originator (console / autonomy critic) always sees a verdict.
            REPLAY.write({"kind": "arbitration",
                          "id": validated["id"],
                          "win": False,
                          "reason": verdict.reason,
                          "lock_key": verdict.lock_key,
                          "holder": verdict.holder_authority})
            self.published_count += 1
            self.store_result({
                "v": validated.get("v", "1.0"),
                "type": "result",
                "id": validated["id"],
                "ok": False,
                "error": "E_ARBITRATION_LOSS",
                "data": {"reason": verdict.reason,
                         "lock_key": verdict.lock_key,
                         "holder": verdict.holder_authority},
                "dispatch_ms": 0.0,
                "ts": time.time(),
            })
            return 0, ["arbitration_loss"]

        drops: list[str] = []
        delivered = 0
        with self._cv:
            for s in self._subs:
                ok, reason = s.enqueue(priority, line)
                if not ok:
                    drops.append(reason or "")
                    self.dropped[priority] += 1
                    REPLAY.write({"kind": "drop", "id": validated["id"],
                                  "priority": priority, "reason": reason})
                    continue
                delivered += 1
                s.delivered += 1
                self.delivered_count += 1
                if reason and reason.startswith("dropped"):
                    # accepted but evicted an older entry — count + log it.
                    self.dropped[priority] += 1
                    REPLAY.write({"kind": "drop", "id": validated["id"],
                                  "priority": priority, "reason": reason})
            self._cv.notify_all()
        self.published_count += 1
        return delivered, drops

    # -- writer-side helpers (called from subscribe handler) ------------------
    def wait_for_message(self, sub: _Subscriber, timeout: float) -> str | None:
        deadline = time.time() + timeout
        with self._cv:
            while True:
                line = sub.pop()
                if line is not None:
                    return line
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)

    # -- result cache ---------------------------------------------------------
    def store_result(self, result: dict) -> None:
        rid = result.get("id")
        if not rid:
            return
        REPLAY.write({"kind": "result", **result})
        now = time.time()
        with self._lock:
            # evict expired
            cutoff = now - RESULT_TTL_S
            while self._result_order:
                head = self._result_order[0]
                rec = self._results.get(head)
                if rec and rec[0] >= cutoff and len(self._results) <= RESULT_CAP:
                    break
                self._result_order.popleft()
                self._results.pop(head, None)
            self._results[rid] = (now, result)
            self._result_order.append(rid)
            while len(self._results) > RESULT_CAP:
                old = self._result_order.popleft()
                self._results.pop(old, None)
        # v1.2 — fan out to console hub if present (decoupled, non-blocking).
        try:
            from console import HUB as _CONSOLE_HUB
            _CONSOLE_HUB.on_result(result)
        except Exception:                               # noqa: BLE001
            pass

    def query_result(self, rid: str) -> dict | None:
        with self._lock:
            rec = self._results.get(rid)
            return rec[1] if rec else None


BROKER = IntentBroker()


def heartbeat_publisher(stop: threading.Event) -> None:
    """§1 — emits heartbeat lines into every subscriber's high-priority lane."""
    while not stop.is_set():
        beat = json.dumps({
            "v": PROTOCOL_VERSION,
            "type": "heartbeat",
            "ts": time.time(),
        }, separators=(",", ":")) + "\n"
        with BROKER._cv:  # type: ignore[attr-defined]
            for s in BROKER._subs:  # type: ignore[attr-defined]
                # Heartbeats bypass drop accounting; if the lane is full we skip.
                dq = s.lanes["high"]
                if len(dq) < LANE_CAPS["high"]:
                    dq.append(beat)
            BROKER._cv.notify_all()  # type: ignore[attr-defined]
        stop.wait(HEARTBEAT_PERIOD_S)


# ----------------------------------------------------------------------------
# Control socket (JSON-line protocol on localhost)
# ----------------------------------------------------------------------------
# Protocol: each connection sends one line of JSON, receives one line of JSON.
#   request  -> {"cmd": "<name>", ...args}
#   response -> {"ok": bool, ...payload}
# Commands: status | restart | shutdown | kill | launch | log_tail | ping
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 5555


def _send_line(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def handle_client(conn: socket.socket, sup: UnrealSupervisor, stop: threading.Event) -> None:
    try:
        conn.settimeout(5.0)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
            if len(buf) > 65536:
                _send_line(conn, {"ok": False, "error": "request too large"})
                return
        try:
            req = json.loads(buf.decode("utf-8").strip())
        except json.JSONDecodeError:
            _send_line(conn, {"ok": False, "error": "invalid json"})
            return

        cmd = req.get("cmd", "").lower()
        log(f"control cmd: {cmd}")

        if cmd == "ping":
            _send_line(conn, {"ok": True, "pong": time.time()})

        elif cmd == "status":
            _send_line(conn, {"ok": True, "supervisor": sup.status_dict()})

        elif cmd == "launch":
            if sup.is_running():
                _send_line(conn, {"ok": False, "error": "already running"})
            else:
                ok = sup.launch()
                _send_line(conn, {"ok": ok})

        elif cmd == "restart":
            sup.kill()
            time.sleep(1.0)
            ok = sup.launch()
            _send_line(conn, {"ok": ok})

        elif cmd == "kill":
            sup.kill()
            _send_line(conn, {"ok": True})

        elif cmd == "shutdown":
            sup.kill()
            stop.set()
            _send_line(conn, {"ok": True, "msg": "QRATUM launcher shutting down"})

        elif cmd == "log_tail":
            n = int(req.get("lines", 50))
            try:
                with LOG_FILE.open("r", encoding="utf-8") as f:
                    lines = f.readlines()[-n:]
                _send_line(conn, {"ok": True, "lines": lines})
            except OSError as e:
                _send_line(conn, {"ok": False, "error": str(e)})

        elif cmd == "push_intent":
            # §2 — validate first, then fan-out.
            intent = req.get("intent")
            if not isinstance(intent, dict):
                name = req.get("name")
                if not name:
                    _send_line(conn, {"ok": False, "error": "E_SCHEMA",
                                      "detail": "missing intent.name"})
                    return
                intent = {
                    "v": PROTOCOL_VERSION,
                    "type": "intent",
                    "name": name,
                    "params": req.get("params", {}),
                    "id": req.get("id", ""),
                    "epoch": req.get("epoch", 0),
                    "priority": req.get("priority", "normal"),
                }
            try:
                validated = IntentValidator.validate(intent)
            except ValidationError as ve:
                _send_line(conn, {"ok": False, "error": ve.code, "detail": ve.detail})
                return
            delivered, drops = BROKER.publish(validated)
            _send_line(conn, {
                "ok": True,
                "id": validated["id"],
                "seq": validated["seq"],
                "delivered": delivered,
                "drops": drops,
                "subscribers": BROKER.subscriber_count(),
            })

        elif cmd == "query_result":
            rid = req.get("id", "")
            res = BROKER.query_result(rid) if rid else None
            if res is None:
                _send_line(conn, {"ok": False, "error": "E_PENDING",
                                  "detail": "no result yet"})
            else:
                _send_line(conn, {"ok": True, "result": res})

        elif cmd == "submit_goal":
            from autonomy import AUTONOMY
            spec = req.get("goal") if isinstance(req.get("goal"), dict) else req
            ok, gid, err = AUTONOMY.submit(spec)
            if ok:
                _send_line(conn, {"ok": True, "goal_id": gid})
            else:
                _send_line(conn, {"ok": False, "error": err or "submit_failed"})

        elif cmd == "goal_status":
            from autonomy import AUTONOMY
            gid = req.get("goal_id") or req.get("id")
            _send_line(conn, AUTONOMY.status(gid))

        elif cmd == "cancel_goal":
            from autonomy import AUTONOMY
            gid = req.get("goal_id") or req.get("id", "")
            ok = AUTONOMY.cancel(gid) if gid else False
            _send_line(conn, {"ok": ok})

        elif cmd == "kill_autonomy":
            from autonomy import AUTONOMY
            n = AUTONOMY.kill_all()
            _send_line(conn, {"ok": True, "killed": n})

        elif cmd == "console_send":
            from console import HUB as CONSOLE_HUB
            cid = req.get("client_id", "")
            raw = req.get("raw", "")
            allow_dbg = bool(req.get("allow_debug", False))
            if not cid:
                _send_line(conn, {"ok": False, "error": "E_NO_CLIENT"})
            else:
                _send_line(conn, CONSOLE_HUB.submit(cid, raw, allow_dbg))

        elif cmd == "console_subscribe":
            from console import HUB as CONSOLE_HUB
            client_v = req.get("v", PROTOCOL_VERSION)
            if client_v != PROTOCOL_VERSION:
                _send_line(conn, {"ok": False, "error": "version_mismatch",
                                  "detail": f"server={PROTOCOL_VERSION}"})
                return

            send_lock = threading.Lock()

            def _sender(frame: dict) -> bool:
                line = json.dumps(frame, separators=(",", ":")) + "\n"
                with send_lock:
                    try:
                        conn.sendall(line.encode("utf-8"))
                        return True
                    except OSError:
                        return False

            ok, cid = CONSOLE_HUB.attach(_sender, req.get("client_id"))
            if not ok:
                _send_line(conn, {"ok": False, "error": "E_BUSY",
                                  "detail": "max console clients"})
                return
            _send_line(conn, {"ok": True, "client_id": cid,
                              "v": PROTOCOL_VERSION})
            log(f"console client {cid} attached")

            # Reader: accept inline {"cmd":"console_send"...} on the same socket
            # so the UE widget can multiplex.
            try:
                conn.settimeout(1.0)
                buf = b""
                while not stop.is_set():
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if msg.get("cmd") == "console_send":
                            CONSOLE_HUB.submit(cid, msg.get("raw", ""),
                                               bool(msg.get("allow_debug", False)))
            finally:
                CONSOLE_HUB.detach(cid)
                log(f"console client {cid} detached")
            return  # subscribe owns the connection

        elif cmd == "result":
            # UE → launcher: acknowledgement or final result frame.
            payload = req.get("result") or req
            if not isinstance(payload, dict) or "id" not in payload:
                _send_line(conn, {"ok": False, "error": "E_SCHEMA"})
                return
            BROKER.store_result({
                "v": PROTOCOL_VERSION,
                "type": payload.get("type", "result"),
                "id": payload["id"],
                "ok": bool(payload.get("ok", True)),
                "error": payload.get("error", ""),
                "data": payload.get("data", {}),
                "dispatch_ms": float(payload.get("dispatch_ms", 0.0) or 0.0),
                "ts": time.time(),
            })
            _send_line(conn, {"ok": True})

        elif cmd == "subscribe":
            # §1 — long-lived bidirectional channel. Writer drains broker;
            # a reader thread parses inbound ack/result frames.
            client_v = req.get("v", PROTOCOL_VERSION)
            if client_v != PROTOCOL_VERSION:
                _send_line(conn, {"ok": False, "error": "version_mismatch",
                                  "detail": f"server={PROTOCOL_VERSION}"})
                return
            _send_line(conn, {"ok": True, "msg": "subscribed",
                              "v": PROTOCOL_VERSION,
                              "subscribers": BROKER.subscriber_count() + 1})
            sub = BROKER.subscribe()
            log(f"subscriber {sub.id[:8]} attached")

            # Reader thread: parse ack/result coming back from this client.
            reader_stop = threading.Event()

            def _reader() -> None:
                buf = b""
                conn.settimeout(1.0)
                while not reader_stop.is_set() and not stop.is_set():
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        sub.last_seen = time.time()
                        mtype = msg.get("type")
                        if mtype in ("ack", "result"):
                            BROKER.store_result({
                                "v": PROTOCOL_VERSION,
                                "type": mtype,
                                "id": msg.get("id", ""),
                                "ok": bool(msg.get("ok", True)),
                                "error": msg.get("error", ""),
                                "data": msg.get("data", {}),
                                "dispatch_ms": float(msg.get("dispatch_ms", 0.0) or 0.0),
                                "ts": time.time(),
                            })
                        # heartbeats just refresh last_seen.

            rt = threading.Thread(target=_reader, daemon=True,
                                  name=f"sub-reader-{sub.id[:8]}")
            rt.start()
            try:
                while not stop.is_set():
                    line = BROKER.wait_for_message(sub, timeout=1.0)
                    if line is None:
                        # staleness check
                        if time.time() - sub.last_seen > SUBSCRIBER_STALE_S:
                            log(f"subscriber {sub.id[:8]} stale; closing",
                                level="WARN")
                            break
                        continue
                    try:
                        conn.sendall(line.encode("utf-8"))
                    except OSError:
                        break
            finally:
                reader_stop.set()
                BROKER.unsubscribe(sub)
                log(f"subscriber {sub.id[:8]} detached")
            return  # subscribe owns the connection through close

        else:
            _send_line(conn, {"ok": False, "error": f"unknown cmd: {cmd}"})

    except (OSError, socket.timeout) as e:
        log(f"control socket error: {e}", level="WARN")
    finally:
        try:
            conn.close()
        except OSError:
            pass


def control_server(sup: UnrealSupervisor, stop: threading.Event, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((CONTROL_HOST, port))
    except OSError as e:
        log(f"cannot bind control socket {CONTROL_HOST}:{port} — {e}", level="ERROR")
        return
    srv.listen(8)
    srv.settimeout(1.0)
    log(f"control server listening on {CONTROL_HOST}:{port}")
    while not stop.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=handle_client, args=(conn, sup, stop), daemon=True).start()
    srv.close()
    log("control server stopped.")


# ----------------------------------------------------------------------------
# Failsafe — Ctrl+Alt+Q kills QRATUM and restores Windows shell
# ----------------------------------------------------------------------------
def failsafe_listener(sup: UnrealSupervisor, stop: threading.Event) -> None:
    if not HAVE_KEYBOARD:
        log("`keyboard` module not installed — failsafe hotkey disabled. "
            "Install with `pip install keyboard` for Ctrl+Alt+Q.",
            level="WARN")
        return

    def _trigger():
        log("Failsafe triggered — restoring Windows shell.", level="WARN")
        try:
            from autonomy import AUTONOMY
            AUTONOMY.kill_all()
        except Exception:                               # noqa: BLE001
            pass
        sup.kill()
        if sys.platform == "win32":
            try:
                subprocess.Popen(["explorer.exe"])
            except OSError:
                pass
        stop.set()
        # Hard-exit so the supervisor can't be re-pinned.
        os._exit(0)

    try:
        keyboard.add_hotkey("ctrl+alt+q", _trigger)
        log("Failsafe armed: Ctrl+Alt+Q to exit & restore shell.")
    except (ImportError, ValueError, OSError) as e:
        log(f"failsafe hotkey registration failed: {e}", level="WARN")
        return

    while not stop.is_set():
        stop.wait(1.0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QRATUM Launcher v2")
    p.add_argument("--editor", type=Path, default=UE_EDITOR_DEFAULT,
                   help="path to UnrealEditor.exe")
    p.add_argument("--project", type=Path, default=UE_PROJECT_DEFAULT,
                   help="path to .uproject")
    p.add_argument("--headless", action="store_true",
                   help="run UE with -nullrhi (no graphics)")
    p.add_argument("--no-ue", action="store_true",
                   help="don't spawn UE; just run supervisor + control socket")
    p.add_argument("--port", type=int, default=CONTROL_PORT,
                   help="control socket port (default 5555)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    log("=" * 60)
    log("QRATUM CORE LAUNCHER v2")
    log(f"  bridge_dir = {bridge_dir()}")
    log(f"  project    = {args.project}")
    log(f"  editor     = {args.editor}")
    log(f"  psutil     = {HAVE_PSUTIL}")
    log(f"  keyboard   = {HAVE_KEYBOARD}")
    log("=" * 60)

    sup = UnrealSupervisor(args.editor, args.project,
                           headless=args.headless, no_ue=args.no_ue)
    stop = threading.Event()

    def _on_signal(signum, _frame):
        log(f"signal {signum} — shutting down")
        sup.kill()
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    sup.launch()

    # ----- v1.1 autonomy plane (QC_CORE_OS_SPEC_V2.md) ----------------------
    from autonomy import AUTONOMY  # local import: late-bound globals

    def _world_view() -> dict:
        # Compose what predicates can match against.
        try:
            state_path = bridge_dir() / "state.json"
            world = json.loads(state_path.read_text("utf-8")) if state_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            world = {}
        # Inject launcher-side facts under /ue and /broker
        world.setdefault("ue", {})["connected"] = BROKER.subscriber_count() > 0
        world["broker"] = {
            "subscribers": BROKER.subscriber_count(),
            "queue_depth": BROKER.queue_depth(),
        }
        return world

    AUTONOMY.bind(BROKER, IntentValidator.validate, ValidationError, log,
                  bridge_dir() / "autonomy.jsonl", _world_view)
    AUTONOMY.start()

    # ----- v1.2 console hub (QC_CORE_OS_SPEC_V1_2.md) -----------------------
    from console import HUB as CONSOLE_HUB
    CONSOLE_HUB.bind(BROKER, IntentValidator.validate, ValidationError, log,
                     bridge_dir() / "console.jsonl", PROTOCOL_VERSION)

    # ----- arbitration v1.0 (QC_CORE_OS_ARBITRATION_V1.md) ------------------
    from arbitration import ARBITER
    ARBITER.bind(BROKER, log, bridge_dir() / "arbitration.jsonl")

    threads = [
        threading.Thread(target=watchdog, args=(sup, stop),
                         daemon=True, name="watchdog"),
        threading.Thread(target=control_server, args=(sup, stop, args.port),
                         daemon=True, name="control"),
        threading.Thread(target=state_publisher, args=(sup, stop),
                         daemon=True, name="publisher"),
        threading.Thread(target=failsafe_listener, args=(sup, stop),
                         daemon=True, name="failsafe"),
        threading.Thread(target=heartbeat_publisher, args=(stop,),
                         daemon=True, name="heartbeat"),
    ]
    for t in threads:
        t.start()

    try:
        while not stop.is_set():
            stop.wait(1.0)
    except KeyboardInterrupt:
        pass

    log("launcher exiting.")
    AUTONOMY.shutdown()
    sup.kill()
    stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
