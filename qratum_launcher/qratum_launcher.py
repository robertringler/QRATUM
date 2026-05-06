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
REPO_ROOT = Path(__file__).resolve().parent.parent
UE_PROJECT_DEFAULT = REPO_ROOT / "soi" / "unreal_bridge" / "SoiGame.uproject"
UE_EDITOR_DEFAULT = Path(r"C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe")
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
        payload = {
            "phase": "RUN",
            "wall_clock": time.time(),
            "supervisor": sup.status_dict(),
            "control_port": CONTROL_PORT,
            "log_file": str(LOG_FILE),
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

    threads = [
        threading.Thread(target=watchdog, args=(sup, stop),
                         daemon=True, name="watchdog"),
        threading.Thread(target=control_server, args=(sup, stop, args.port),
                         daemon=True, name="control"),
        threading.Thread(target=state_publisher, args=(sup, stop),
                         daemon=True, name="publisher"),
        threading.Thread(target=failsafe_listener, args=(sup, stop),
                         daemon=True, name="failsafe"),
    ]
    for t in threads:
        t.start()

    try:
        while not stop.is_set():
            stop.wait(1.0)
    except KeyboardInterrupt:
        pass

    log("launcher exiting.")
    sup.kill()
    stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
