#!/usr/bin/env python3
"""QRATUM Kernel Daemon — Φ=2 headless boot path.

Mirrors the UE5 ``UQRATUMBridgeSubsystem`` JSON contract so the desktop GUI
can attach to QRATUM without the Unreal simulation running. Implements the
deterministic kernels for CIIR / CRS / RIC / QuaSim / QuBIC / RMHD / MVRI /
RWB / CGL using only the Python standard library + numpy.

Boot sequence (matches §8 dependency order):
    POST → RWB → MVRI → CIIR → CRS → CGL → QuaSim → QuBIC → RMHD → RIC → RUN

State publish: ``%LOCALAPPDATA%/QRATUM/bridge/state.json`` (Windows) or
``$HOME/.local/share/QRATUM/bridge/state.json`` (Unix). Atomic write via
temp + rename.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    import numpy as np
except ImportError:  # numpy missing → minimal fallback
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def bridge_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "QRATUM" / "bridge"


def boot_log_path() -> Path:
    return bridge_dir() / "boot.log"


def state_path() -> Path:
    return bridge_dir() / "state.json"


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@dataclass
class CIIRKernel:
    """φ_{n+1} = T(φ_n) contraction on R^4. [Ch.22, Thm 22.1]."""

    A: list = field(
        default_factory=lambda: [
            [0.55, -0.10, 0.05, 0.00],
            [0.10, 0.50, -0.05, 0.00],
            [0.00, 0.05, 0.45, 0.10],
            [0.00, 0.00, -0.10, 0.40],
        ]
    )
    b: list = field(default_factory=lambda: [0.30, -0.20, 0.15, 0.05])
    phi_star: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    phi: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    epoch: int = 0
    delta: float = 0.0

    def apply(self, x: list) -> list:
        return [self.b[i] + sum(self.A[i][j] * x[j] for j in range(4)) for i in range(4)]

    def initialize(self) -> None:
        x = [0.0, 0.0, 0.0, 0.0]
        for _ in range(1000):
            xn = self.apply(x)
            d = math.sqrt(sum((xn[i] - x[i]) ** 2 for i in range(4)))
            x = xn
            if d < 1e-6:
                break
        self.phi_star = list(x)
        self.phi = list(x)

    def step(self) -> None:
        decay = 0.99
        new_phi = [decay * self.phi[i] + (1 - decay) * self.phi_star[i] for i in range(4)]
        self.delta = math.sqrt(sum((new_phi[i] - self.phi[i]) ** 2 for i in range(4)))
        self.phi = new_phi
        self.epoch += 1

    @property
    def phi_norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.phi))


@dataclass
class CRSKernel:
    active_stratum: int = 0
    stratum_count: int = 4
    iss_residual: float = 0.0
    accumulated_time: float = 0.0

    def step(self, dt: float) -> None:
        self.iss_residual = max(0.0, self.iss_residual * math.exp(-0.5 * dt))
        self.accumulated_time += dt
        if self.accumulated_time >= 2.0:
            self.accumulated_time = 0.0
            self.active_stratum = (self.active_stratum + 1) % self.stratum_count


@dataclass
class RICKernel:
    weights: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    last_write_tick: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    tick: int = 0

    def step(self) -> None:
        self.tick += 1
        k_bias = 0.05
        s = 0.0
        new_w = []
        for i, w in enumerate(self.weights):
            age = self.tick - self.last_write_tick[i]
            corrected = w / (1.0 + k_bias * age)
            new_w.append(corrected)
            s += corrected
        if s > 1e-9:
            self.weights = [w / s for w in new_w]


@dataclass
class QuaSimKernel:
    """CHSH on the singlet. E(a,b) = -cos(a-b). S = 2√2 [Ch.10, Thm 10.1]."""

    s: float = 0.0
    samples: int = 0
    bound: bool = False
    tick_counter: int = 0

    def step(self) -> None:
        self.tick_counter += 1
        if self.tick_counter % 100 != 0:
            return
        a, ap, b, bp = 0.0, math.pi / 2, math.pi / 4, 3 * math.pi / 4
        e_ab = -math.cos(a - b)
        e_abp = -math.cos(a - bp)
        e_apb = -math.cos(ap - b)
        e_apbp = -math.cos(ap - bp)
        s = abs(e_ab - e_abp + e_apb + e_apbp)
        self.s = s
        self.samples += 1
        self.bound = 2.0 < s <= 2.0 * math.sqrt(2.0) + 1e-4


@dataclass
class QuBICKernel:
    n: int = 64
    weights: List[float] = field(default_factory=list)
    epoch: int = 0

    def initialize(self) -> None:
        self.weights = [1.0 if i == 0 else 0.0 for i in range(self.n)]

    def step(self) -> None:
        nxt = [0.0] * self.n
        for i in range(self.n):
            nxt[(i + 1) % self.n] += 0.5 * self.weights[i]
            nxt[i] += 0.5 * self.weights[i]
        s = sum(nxt)
        if s > 1e-9:
            nxt = [v / s for v in nxt]
        self.weights = nxt
        self.epoch += 1


@dataclass
class RMHDKernel:
    rate: float = 0.0
    psi: float = 1.0

    def initialize(self) -> None:
        # Sweet–Parker [Ch.13, Thm 13.2]: V_in/V_A ≈ S^{-1/2}, S = 1e6.
        self.rate = 1.0 / math.sqrt(1.0e6)

    def step(self, dt: float) -> None:
        self.psi = max(0.0, self.psi - self.rate * dt)


@dataclass
class MVRIKernel:
    ema: float = 0.0
    gate_mask: int = 0x1F
    status: str = "Accepted"

    def ingest(self, sensor: List[float]) -> None:
        if not sensor:
            self.status = "RejectedInv"
            return
        n = len(sensor)
        mean = sum(sensor) / n
        var = max(0.0, sum(v * v for v in sensor) / n - mean * mean)
        self.ema = 0.9 * self.ema + 0.1 * mean
        if abs(mean) > 10.0:
            self.gate_mask &= ~0x01
            self.status = "RejectedGate"
        elif var > 10.0:
            self.gate_mask &= ~0x02
            self.status = "RejectedBound"
        else:
            self.gate_mask = 0x1F
            self.status = "Accepted"


@dataclass
class RWBKernel:
    sequence: int = 0
    accumulated_time: float = 0.0

    def step(self, dt: float) -> List[float]:
        self.sequence += 1
        self.accumulated_time += dt
        return [math.sin(self.accumulated_time + 0.25 * i) for i in range(8)]


@dataclass
class CGLKernel:
    lambda_min: float = 1.0
    rank_deficient: bool = False
    accumulated_time: float = 0.0

    def step(self, dt: float) -> None:
        self.accumulated_time += dt
        self.lambda_min = 0.5 + 0.5 * math.cos(self.accumulated_time * 0.5)
        self.rank_deficient = self.lambda_min < 0.05


# ---------------------------------------------------------------------------
# Boot orchestrator
# ---------------------------------------------------------------------------


class Kernel:
    def __init__(self, hz: float = 60.0) -> None:
        self.hz = hz
        self.dt = 1.0 / hz
        self.ciir = CIIRKernel()
        self.crs = CRSKernel()
        self.cgl = CGLKernel()
        self.quasim = QuaSimKernel()
        self.qubic = QuBICKernel()
        self.rmhd = RMHDKernel()
        self.mvri = MVRIKernel()
        self.rwb = RWBKernel()
        self.ric = RICKernel()
        self.epoch_tick = 0
        self.phase = "POST"

    # Boot order matches §8: RWB→MVRI→CIIR→CRS→CGL→QuaSim→QuBIC→RMHD→RIC.
    def boot(self, log) -> None:
        steps = [
            ("RWB", None),
            ("MVRI", None),
            ("CIIR", self.ciir.initialize),
            ("CRS", None),
            ("CGL", None),
            ("QuaSim", None),
            ("QuBIC", self.qubic.initialize),
            ("RMHD", self.rmhd.initialize),
            ("RIC", None),
        ]
        self.phase = "INIT"
        log("[QRATUM] POST OK")
        for name, fn in steps:
            t0 = time.perf_counter()
            if fn is not None:
                fn()
            log(f"[QRATUM] init {name:6s} ok  ({(time.perf_counter()-t0)*1000:.2f} ms)")
        self.phase = "RUN"
        log("[QRATUM] kernel ready — entering RUN phase")

    def step(self) -> None:
        sensor = self.rwb.step(self.dt)
        self.mvri.ingest(sensor)
        self.ciir.step()
        self.crs.step(self.dt)
        self.cgl.step(self.dt)
        self.quasim.step()
        self.qubic.step()
        self.rmhd.step(self.dt)
        self.ric.step()
        self.epoch_tick += 1

    def snapshot(self) -> dict:
        return {
            "phase": self.phase,
            "epoch_tick": self.epoch_tick,
            "wall_clock_seconds": time.time(),
            "ciir": {
                "phi": self.ciir.phi_norm,
                "delta_w_norm": self.ciir.delta,
                "active_constraints": 4,
            },
            "crs": {
                "active_stratum": self.crs.active_stratum,
                "iss_residual": self.crs.iss_residual,
                "stratum_count": self.crs.stratum_count,
            },
            "ric_dispatch_weights": list(self.ric.weights),
            "quasim": {
                "s": self.quasim.s,
                "quantum_bound": self.quasim.bound,
                "samples": self.quasim.samples,
            },
            "qubic": {"epoch": self.qubic.epoch, "nodes": self.qubic.n},
            "rmhd": {"reconnection_rate": self.rmhd.rate, "psi_flux": self.rmhd.psi},
            "mvri": {
                "ema": self.mvri.ema,
                "gate_mask": self.mvri.gate_mask,
                "status": self.mvri.status,
            },
            "rwb": {"sequence": self.rwb.sequence},
            "cgl": {"lambda_min": self.cgl.lambda_min, "rank_deficient": self.cgl.rank_deficient},
        }


def write_state_atomic(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QRATUM Kernel Daemon")
    p.add_argument("--hz", type=float, default=60.0, help="Tick rate (Hz)")
    p.add_argument("--write", type=float, default=0.1, help="State publish interval (s)")
    p.add_argument("--steps", type=int, default=0, help="Stop after N steps (0 = run forever)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    sp = state_path()
    bp = boot_log_path()
    bp.parent.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(msg: str) -> None:
        log_lines.append(f"{time.strftime('%H:%M:%S')} {msg}")
        if not args.quiet:
            print(msg, flush=True)
        bp.write_text("\n".join(log_lines), encoding="utf-8")

    log(f"[QRATUM] boot daemon pid={os.getpid()} state={sp}")
    k = Kernel(hz=args.hz)
    k.boot(log)
    write_state_atomic(k.snapshot(), sp)

    halted = {"flag": False}

    def handle_sigint(_sig, _frm):
        halted["flag"] = True

    try:
        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)
    except (AttributeError, ValueError):
        pass

    write_every = max(1, int(args.write * args.hz))
    next_tick = time.perf_counter()
    step_count = 0
    while not halted["flag"]:
        k.step()
        step_count += 1
        if step_count % write_every == 0:
            write_state_atomic(k.snapshot(), sp)
        if args.steps > 0 and step_count >= args.steps:
            break
        next_tick += k.dt
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()

    k.phase = "HALT"
    write_state_atomic(k.snapshot(), sp)
    log(f"[QRATUM] halted after {step_count} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
