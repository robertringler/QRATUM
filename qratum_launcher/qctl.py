"""
qctl — tiny CLI client for QRATUM Launcher v2 control socket.

Usage:
    python qctl.py status
    python qctl.py restart
    python qctl.py kill
    python qctl.py launch
    python qctl.py shutdown
    python qctl.py log_tail --lines 100
    python qctl.py ping
    python qctl.py push --name spawn_vehicle --params '{"location":[0,0,100]}'
"""

from __future__ import annotations

import argparse
import json
import socket
import sys


def send(host: str, port: int, payload: dict, *, timeout: float = 5.0) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((host, port))
    s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode("utf-8").strip()) if buf else {"ok": False, "error": "no response"}


def _run_console_repl(host: str, port: int) -> int:
    """v1.2 — interactive QRATUM console REPL.

    Opens a console_subscribe long-lived socket; spawns a reader
    thread that prints inbound frames; main thread reads stdin and
    pushes console_send frames over the same socket.
    """
    import threading

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
    except OSError as e:
        print(f"connect error: {e}", file=sys.stderr)
        return 2
    s.sendall((json.dumps({"cmd": "console_subscribe", "v": "1.0"}) + "\n").encode("utf-8"))

    stop = threading.Event()

    def _reader() -> None:
        buf = b""
        while not stop.is_set():
            try:
                chunk = s.recv(4096)
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
                t = msg.get("type") or (
                    "ack" if msg.get("ok") is not None and "client_id" in msg else "?"
                )
                if t == "console_intent":
                    env = msg.get("intent", {})
                    print(
                        f"  >> {env.get('name')}  id={env.get('id','')[:8]}  "
                        f"delivered={msg.get('delivered')}"
                    )
                elif t == "console_ack":
                    r = msg.get("result", {})
                    print(
                        f"  ack {r.get('id','')[:8]}  ok={r.get('ok')}  "
                        f"{r.get('dispatch_ms', 0):.1f} ms"
                    )
                elif t == "console_result":
                    r = msg.get("result", {})
                    print(
                        f"  result {r.get('id','')[:8]}  ok={r.get('ok')}  "
                        f"{r.get('dispatch_ms', 0):.1f} ms"
                    )
                    if r.get("error"):
                        print(f"    error: {r['error']}")
                    if r.get("data"):
                        print(f"    data: {json.dumps(r['data'])[:200]}")
                elif t == "console_error":
                    print(f"  ! {msg.get('error')}: {msg.get('detail')}")
                elif t == "console_log":
                    print(f"  [{msg.get('level','INFO')}] {msg.get('msg')}")
                else:
                    print(f"  {json.dumps(msg)[:200]}")

    rt = threading.Thread(target=_reader, daemon=True, name="console-reader")
    rt.start()

    print("QRATUM console — type :help, :quit to exit, or any intent line.")
    try:
        while True:
            try:
                line = input("qratum> ")
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line:
                continue
            if line in (":quit", ":q", ":exit"):
                break
            if line == ":help":
                print("  examples:")
                print("    sim.tick")
                print("    actor.spawn type=truck location=0,0,100")
                print("    vehicle.set_throttle value=0.7 !high")
                print("  meta: :clear  :quit")
                continue
            if line == ":clear":
                # cross-platform clear
                print("\033[2J\033[H", end="")
                continue
            try:
                s.sendall((json.dumps({"cmd": "console_send", "raw": line}) + "\n").encode("utf-8"))
            except OSError as e:
                print(f"send error: {e}", file=sys.stderr)
                break
    finally:
        stop.set()
        try:
            s.close()
        except OSError:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QRATUM launcher control client")
    p.add_argument(
        "cmd",
        choices=[
            "status",
            "restart",
            "kill",
            "launch",
            "shutdown",
            "log_tail",
            "ping",
            "push",
            "query",
            "goal",
            "goals",
            "cancel",
            "kill_autonomy",
            "console",
        ],
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--lines", type=int, default=50, help="lines to retrieve for log_tail")
    p.add_argument("--name", help="intent or goal name (push/goal)")
    p.add_argument("--params", default="{}", help="params as JSON string (push/goal)")
    p.add_argument("--id", default="", help="intent/goal correlation id")
    p.add_argument("--epoch", type=int, default=0, help="kernel epoch (push)")
    p.add_argument(
        "--priority", choices=["high", "normal", "low"], default="normal", help="priority lane"
    )
    p.add_argument("--deadline", type=float, default=None, help="goal deadline in seconds")
    p.add_argument("--max-replans", type=int, default=8, help="goal replan budget")
    p.add_argument("--success", default="[]", help="goal success predicates as JSON array")
    p.add_argument("--failure", default="[]", help="goal failure predicates as JSON array")
    args = p.parse_args(argv)

    payload: dict = {"cmd": args.cmd}
    if args.cmd == "log_tail":
        payload["lines"] = args.lines
    elif args.cmd == "query":
        if not args.id:
            print("error: --id is required for query", file=sys.stderr)
            return 2
        payload = {"cmd": "query_result", "id": args.id}
    elif args.cmd == "goals":
        payload = {"cmd": "goal_status"}
    elif args.cmd == "cancel":
        if not args.id:
            print("error: --id required for cancel", file=sys.stderr)
            return 2
        payload = {"cmd": "cancel_goal", "goal_id": args.id}
    elif args.cmd == "kill_autonomy":
        payload = {"cmd": "kill_autonomy"}
    elif args.cmd == "goal":
        if not args.name:
            print("error: --name required for goal", file=sys.stderr)
            return 2
        try:
            params = json.loads(args.params)
            success = json.loads(args.success)
            failure = json.loads(args.failure)
        except json.JSONDecodeError as e:
            print(f"error: bad JSON: {e}", file=sys.stderr)
            return 2
        spec = {
            "name": args.name,
            "params": params,
            "priority": args.priority,
            "max_replans": args.max_replans,
            "success": success,
            "failure": failure,
        }
        if args.deadline is not None:
            spec["deadline_s"] = args.deadline
        if args.id:
            spec["id"] = args.id
        payload = {"cmd": "submit_goal", "goal": spec}
    elif args.cmd == "push":
        if not args.name:
            print("error: --name is required for push", file=sys.stderr)
            return 2
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"error: --params is not valid JSON: {e}", file=sys.stderr)
            return 2
        payload = {
            "cmd": "push_intent",
            "intent": {
                "v": "1.0",
                "type": "intent",
                "name": args.name,
                "params": params,
                "id": args.id,
                "epoch": args.epoch,
                "priority": args.priority,
            },
        }

    if args.cmd == "console":
        return _run_console_repl(args.host, args.port)

    try:
        resp = send(args.host, args.port, payload)
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "log_tail" and resp.get("ok"):
        for line in resp.get("lines", []):
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
    else:
        print(json.dumps(resp, indent=2))
    return 0 if resp.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
