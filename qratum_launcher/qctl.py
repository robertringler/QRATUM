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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QRATUM launcher control client")
    p.add_argument("cmd", choices=["status", "restart", "kill", "launch",
                                    "shutdown", "log_tail", "ping"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--lines", type=int, default=50,
                   help="lines to retrieve for log_tail")
    args = p.parse_args(argv)

    payload: dict = {"cmd": args.cmd}
    if args.cmd == "log_tail":
        payload["lines"] = args.lines

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
