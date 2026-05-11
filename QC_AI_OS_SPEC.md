# QRATUM CORE AI-NATIVE OS (QC-AI-OS) — v1 Specification

> **Status:** canonical design, synthesized by the 12-agent swarm.
> **Posture:** _orchestration shell_, not a kernel replacement. QC-AI-OS rides on top of the host OS (Windows 10/11, Linux 5.15+) and **owns the user session**: from logon onward, the user never sees the host desktop shell; they see QRATUM.
> **Determinism contract:** every action that mutates simulation state passes through CIIR → RIC → Spine and is appended to the cryptographic event chain. The host OS becomes a HAL.

---

## 0. Reality check (what's already built)

| Component | Path | Status |
|---|---|---|
| File-IPC bridge contract (`state.json`) | `%LOCALAPPDATA%/QRATUM/bridge/state.json` | live |
| Headless kernel daemon (Python, Φ=2) | [qratum_kernel/qratum_kernel_daemon.py](qratum_kernel/qratum_kernel_daemon.py) | live |
| UE5 swarm (9 subsystems + Bridge) | [soi/unreal_bridge/Source/QRATUM/](soi/unreal_bridge/Source/QRATUM/) | code complete, not yet built |
| Tauri desktop shell | [qratum_desktop/](qratum_desktop/) | live |
| Boot orchestrator | [qratum-boot.cmd](qratum-boot.cmd) / [qratum-boot.sh](qratum-boot.sh) | live |
| Logon-time autostart | [qratum-boot-install.ps1](qratum-boot-install.ps1) | live |
| UE5 launcher | [soi/unreal_bridge/Run.bat](soi/unreal_bridge/Run.bat) | live (build pending) |

The 12-agent spec below extends these pieces; nothing here is greenfield fantasy.

---

## AGENT 1 — System Architecture

### Decisions

- **Three-plane model**
  - **Substrate plane** — host OS (Windows kernel / Linux kernel + systemd). Provides drivers, scheduler, GPU stack. Untouched.
  - **Control plane** — `qratum-kernel` daemon. Runs the Φ=2 math (CIIR/CRS/RIC/QuaSim/QuBIC/RMHD/MVRI/RWB/CGL), publishes `state.json`, accepts intents on a Unix domain socket / Windows named pipe.
  - **Presentation plane** — UE5 SoiGame (graphical world) + Tauri shell (HUD, settings, fallback). Both attach to `state.json` as **read-only consumers** + write intents back via the control socket.
- **Single source of truth:** the `state.json` snapshot (atomic temp+rename, Merkle-chained per epoch). Whoever holds the `kernel.lock` is the authoritative publisher; UE5 takes the lock when launched, Python falls back to read-only follower.
- **Boot flow:**

  ```
  POWER → host kernel → host login → QC-AI-OS bootloader (logon task)
   └─ POST  : sanity (python, paths, GPU, lockfile)
   └─ INIT  : spawn qratum-kernel (control plane)
   └─ SHELL : spawn UE5 SoiGame fullscreen + Tauri HUD overlay
   └─ READY : input router live, host shell suppressed
  ```

- **Session ownership:** a `qratum-session` process becomes the user's `Userinit`/`Shell` replacement (Win) or replaces `gnome-session` (Linux). On crash, watchdog restores host shell.

### Interfaces

| Interface | Producer | Consumer | Transport |
|---|---|---|---|
| `BridgeState` | kernel | UE5, Tauri | `state.json` (atomic) |
| `Intent` | UE5/Tauri/AI | kernel | named pipe / unix socket, JSON-RPC 2.0 |
| `EventChain` | kernel | audit | append-only `events.log` (SHA-256 chained) |
| `SessionControl` | session-mgr | host OS | Win32 `Userinit` / PAM session |

### Risks

- Suppressing the host shell is reversible only if a watchdog daemon stays alive — must be a separate process with hardware watchdog timer.
- GPU contention between UE5 and any host compositor remnants: solved by running UE5 in DXGI exclusive fullscreen and disabling DWM on the active monitor.

---

## AGENT 2 — Bootstrap & Init

### Decisions

- **Tier-0 bootloader:** scheduled task (Windows) / systemd unit (Linux), already implemented for kernel-only mode. Extend to run a `qratum-session.exe` that owns the lifecycle.
- **Stages mirror BIOS→OS metaphor:**

  | Stage | Duration | Failure → |
  |---|---|---|
  | POST | <100 ms | abort, emit `boot.fail` |
  | INIT | <2 s | retry 3×, then host-shell fallback |
  | SHELL | <8 s | UE5 fail → Tauri-only mode |
  | READY | – | watchdog active |

- **Fail-safe rollback:** `qratum-session.exe --panic` re-enables `explorer.exe` (Win) or `gnome-shell` (Linux) and writes a panic dump to `%LOCALAPPDATA%/QRATUM/panic/`.
- **Boot phase published in `state.json`** as `phase ∈ {POST, INIT, RUN, HALT, PANIC}`.

### Pseudocode

```rust
// qratum-session/src/main.rs
fn main() -> ! {
    post()?;                            // python, paths, GPU, lock
    let kernel = spawn_kernel()?;       // Python or UE5 publisher
    wait_for_state(Duration::from_secs(5))?;
    let ue   = spawn_ue5_shell()?;
    let hud  = spawn_tauri_overlay()?;
    suppress_host_shell();              // Win: HKCU\...\Winlogon\Shell
    Watchdog::supervise(vec![kernel, ue, hud]).run();   // never returns
}
```

### Risks

- Suppressing Windows Explorer requires editing `HKCU\Software\Microsoft\Windows NT\CurrentVersion\Winlogon\Shell` — must be reverted by panic handler or user is locked out. Mitigation: keep a hardware-keyed escape (`Ctrl+Alt+Shift+Q` → spawn `cmd.exe` directly).

---

## AGENT 3 — Unreal Engine Integration

### Decisions

- **Embedding model:** UE5 runs as a **child process**, not embedded as a library. Communication via the bridge (state) + a UE5 plugin `QRATUMComm` exposing JSON-RPC over named pipe.
- **Two UE5 modes:**
  - **Graphical** — `UnrealEditor.exe SoiGame.uproject -game -fullscreen` (production)
  - **Headless** — `UnrealEditor-Cmd.exe ... -nullrhi -unattended` (CI / server)
- **Plugin layout** (already coded as `QRATUM` module, just needs build): each subsystem is a `UTickableWorldSubsystem`. Tick rate locked to 60 Hz via `FApp::SetFixedDeltaTime(1.0/60.0)`.
- **Bridge ownership protocol:**
  1. UE5 `UQRATUMBridgeSubsystem::Initialize()` tries to acquire `kernel.lock` (atomic create).
  2. If acquired → UE5 publishes; Python daemon detects lock and downgrades to follower mode (no writes, just consumes intents).
  3. If UE5 exits, lock is released, Python promotes itself.

### Interfaces

```cpp
// QRATUMComm plugin — JSON-RPC methods
qratum.intent.submit(intent: Intent) -> ContractHash
qratum.subsystem.poll(name: string) -> SubsystemState
qratum.event.append(evt: Event) -> EpochTick
```

### Risks

- Cooked vs editor build: shipping needs `BuildCookRun.bat` packaging — separate v1.1 task.
- Subsystem tick order: must respect §8 boot order (RWB→MVRI→CIIR→CRS→CGL→QuaSim→QuBIC→RMHD→RIC) via `GetDependencies()`.

---

## AGENT 4 — SoiGame Integration

### Decisions

- **SoiGame is the "first world"** — the equivalent of a desktop wallpaper, but interactive and AI-driven.
- World init order, gated on `phase == RUN`:
  1. `WorldSubsystem::OnWorldBeginPlay` waits for `state.json.phase == RUN`
  2. `SwarmManager` reads `bridge.qubic.nodes` and instantiates 64 `AAgentNode` actors
  3. CIIR's φ-fixed-point seeds initial topology
  4. Player pawn (or AI camera) spawned at deterministic seed `hash(epoch_tick)`
- **Game state ↔ kernel state binding:** every `AAgentNode::Tick` reads its row from `bridge.qubic.graph_diffusion[i]`; mutations require submitting an `Intent` to the kernel, which authorizes via QCore and broadcasts the new contract back.
- **Determinism rule:** UE5 randomness (`FMath::Rand`) is forbidden inside swarm logic; use `FRandomStream(epoch_tick)` exclusively.

### Risks

- UE5's GC may evict actors mid-tick → keep `AgentNode` references in a `UPROPERTY` array on `USwarmManager`.

---

## AGENT 5 — CIIR / RIC Control

### Decisions

- **Authority pipeline** (mirrors QCore architecture freeze):

  ```
  LLM/UE5/User → Intent (QIL string)
              → Parser (qil/) → AST
              → QCore.authorize() → either AuthorizationError or Contract
              → Spine.dispatch(Contract) → Adapter → Event
              → EventChain.append(Event)  [SHA-256 chained]
  ```

- **CIIR fixed-point** is the arbitration kernel: any intent whose post-state would push `|Δφ| > ε_max` (default 1e-3) is rejected as destabilizing. Stops runaway LLM behaviors.
- **RIC bias-correction** weights every dispatch by `score_c = score_r / (1 + k·age)` to prevent stale-priority lock-in.
- **Rollback:** every Contract carries a `parent_hash`. If a downstream adapter fails, the `EventChain` records `Reverted(parent_hash)` and CIIR rolls φ back to the parent fixed point.

### Pseudocode

```python
def submit_intent(intent: Intent) -> Contract | AuthorizationError:
    ast        = parse(intent.qil)
    candidate  = ciir.simulate_step(ast)         # what would φ become?
    if abs(candidate.phi - ciir.phi).norm() > EPS_MAX:
        raise AuthorizationError("would destabilize CIIR")
    contract = qcore.issue(ast, candidate)        # frozen dataclass
    spine.dispatch(contract)                      # async, returns event
    return contract
```

### Risks

- Single point of failure on QCore → run two QCore instances in **active-passive** with Raft log replication; passive promotes on missed heartbeats.

---

## AGENT 6 — AI Orchestration

### Decisions

- **In-OS agent layer = a small fixed swarm**, not freeform LLM autonomy:

  | Agent | Role | Memory |
  |---|---|---|
  | `Perceiver` | reads state.json + UE5 sensor stream | ring buffer 1024 epochs |
  | `Planner` | decomposes user goals into QIL intents | episodic, on disk |
  | `Critic` | rejects unsafe intents pre-CIIR | rules + small LM |
  | `Executor` | submits intents to QCore | none (stateless) |
  | `Narrator` | renders explanations to HUD | last 30 s |

- **Memory persistence:** `~/.qratum/memory/` with three tiers — working (RAM, lost on halt), episodic (sqlite, per session), semantic (vector store, persistent).
- **LLM access:** off by default. When enabled, every LLM call is wrapped: response → parsed as QIL → routed through CIIR. The LLM can never directly mutate state.

### Risks

- Vector store size growth → eviction policy: drop episodes whose `score_c < 0.05` after 7 days.

---

## AGENT 7 — Security & Isolation

### Decisions

- **Process boundaries:** kernel daemon runs as the user (no admin); UE5 runs as the user; QCore runs in its own process with a **distinct named-pipe DACL** restricting write access to the kernel binary's signed handle.
- **Sandbox:** Windows AppContainer (low-IL token) for the Tauri shell — it can read state.json but cannot spawn processes, only submit intents.
- **GPU ACL:** UE5 holds GPU exclusivity; AI agents using CUDA must request a `GPULease` contract.
- **Anti-corruption:** every `state.json` write includes `state_hash = sha256(canonical_json(payload))`; readers verify before accepting. Mismatched hash → reader stalls and emits `IntegrityViolation` event.
- **Network posture:** loopback only by default. External calls require an `EgressIntent` contract that names the destination and rate cap.

### Threat model (top 3)

| Threat | Mitigation |
|---|---|
| Malicious LLM-generated intent | CIIR ε-gate + Critic agent |
| Disk-tampered `state.json` | SHA-256 + Merkle chain in `events.log` |
| Process injection into kernel | code-signed binary + ACL'd named pipe |

---

## AGENT 8 — Graphics & Render Pipeline

### Decisions

- **UE5 owns the framebuffer.** It runs DXGI exclusive fullscreen on the primary monitor; the host compositor (DWM) is unloaded for that adapter via `IDXGIOutput::TakeOwnership`.
- **Tauri HUD** is rendered as a **textured quad inside UE5** (via a `UMediaTexture` fed from a Chromium offscreen buffer through the `QRATUMComm` plugin). This gives one composited frame, no z-fighting between two windows.
- **Latency targets:** AI-decision → UE-action ≤ 2 frames at 60 Hz (= 33 ms). Achieved by keeping the kernel→UE pipe non-blocking and submitting intents in UE's `PreActorTick`.
- **VR mode:** OpenXR runtime piggybacks on existing `ciir_vr_simulation/` — same shaders, same heatmaps; HUD reprojected per-eye.

### Risks

- Disabling DWM is hostile to multi-monitor users → only do so on the monitor UE5 owns; leave DWM up on others.

---

## AGENT 9 — Data & Memory

### Decisions

- **State storage hierarchy:**

  ```
  hot     → RAM (state.json, kernel struct)
  warm    → ~/.qratum/state/  (snapshots every 60 s, ring of 16)
  cold    → ~/.qratum/events/ (append-only Merkle log, rotated daily)
  archive → optional S3/B2, encrypted with user's libsodium keypair
  ```

- **Snapshot/restore:** `qratumctl snapshot create <tag>` freezes kernel for ≤50 ms, copies struct + last 1024 events. `qratumctl snapshot restore <tag>` reseeds kernel and replays events past the snapshot tick.
- **Versioned memory graph:** semantic store keyed by `(epoch_tick, contract_hash)`; queries return causally-consistent views.
- **Schema:** all on-disk state uses the same frozen-dataclass JSON contracts that travel through `state.json`, so there is one schema, versioned by `bridge.schema_version`.

### Risks

- Disk write amplification at 60 Hz → snapshots are sampled at 1 Hz; hot path stays in RAM only.

---

## AGENT 10 — DevOps & Deployment

### Decisions

- **Single installer per OS:**
  - Windows: MSIX bundle containing `qratum-session.exe`, kernel daemon, Tauri shell, UE5 cooked SoiGame, and a post-install hook running `qratum-boot-install.ps1`.
  - Linux: deb/rpm + systemd user unit (`~/.config/systemd/user/qratum-session.service`).
- **UE5 packaging:** `Run.bat /package` invokes `BuildCookRun` → produces `dist/SoiGame-Win64-Shipping/` (~3 GB). Bundled as MSIX optional feature so kernel-only installs stay small.
- **Dependency strategy:**

  | Dep | Strategy |
  |---|---|
  | Python 3.11 | embedded (PyOxidizer) — never relies on system Python |
  | UE5 runtime | shipped as cooked binary, no engine install needed |
  | VC++ runtime | static-linked Tauri/UE5 binaries |

- **CI:** existing `.github/workflows/` adds a `qc-ai-os-build` job: lint → kernel pytest → UE5 cook (Linux runner with UE5 docker image) → MSIX pack → SBOM (CycloneDX) → sign.

### Risks

- UE5 binary redistribution requires Epic's EULA acceptance per user — installer must show the EULA on first run.

---

## AGENT 11 — Performance & Optimization

### Decisions

- **Budgets (per 16.67 ms frame):**

  | Stage | Budget |
  |---|---|
  | UE5 render | 8 ms |
  | UE5 game tick (subsystems) | 3 ms |
  | Bridge IPC (intent + state) | 1 ms |
  | Kernel CIIR/CRS/etc. | 2 ms |
  | Headroom | 2.67 ms |

- **Threading:** kernel uses 1 thread for the `Tick()` loop + 1 for IPC; UE5 keeps default task-graph; AI agents in their own process pool, throttled to ≤30 % of one core.
- **GPU:** UE5 owns it; kernel/AI use CPU only by default. Optional CUDA path for QuaSim large-S quantum probes — gated behind a `GPULease` contract.
- **Hot-path optimizations:**
  - `state.json` is double-buffered (`state.A.json`/`state.B.json` + atomic symlink swap) → reader never blocks writer.
  - Subsystems share a single `BridgeStateCache` SoA struct, not per-subsystem JSON serialization.

### Risks

- File-IPC scales to one publisher; for distributed (multi-node) operation v1.1 swaps to NATS or shared-memory.

---

## AGENT 12 — Synthesis & Roadmap

### Resolved contradictions

| Conflict | Resolution |
|---|---|
| Agent 3 wanted UE5 embedded as lib; Agent 7 wanted process isolation | **Process isolation wins** (security > convenience). |
| Agent 8 wanted DXGI exclusive; Agent 10 wanted multi-monitor friendly | UE5 takes only the monitor it spawns on. |
| Agent 6 wanted free LLM autonomy; Agent 5 demanded CIIR gating | All LLM output → QIL → CIIR ε-gate. No exceptions. |
| Agent 4 wanted FMath::Rand for variety; Agent 5 demanded determinism | `FRandomStream(epoch_tick)` only. |

### v1 implementation roadmap (12 weeks, 4 milestones)

**M1 — Foundation (already done, weeks 0)**

- ✅ Bridge contract + Python kernel + Tauri HUD + boot scripts + scheduled tasks

**M2 — UE5 wired (weeks 1–4)**

- Build the QRATUM module via [Run.bat /build](soi/unreal_bridge/Run.bat)
- Implement `kernel.lock` ownership protocol (UE5 wins, Python follows)
- `QRATUMComm` JSON-RPC plugin
- Cooked Shipping build of SoiGame
- Bridge automation tests (≥ 1 per subsystem, §4 of v3 prompt)

**M3 — Session takeover (weeks 5–8)**

- `qratum-session.exe` Rust binary (replaces `Userinit\Shell`)
- HUD-as-UE5-texture compositing
- Watchdog + panic restore
- AppContainer sandbox for Tauri
- AI swarm (Perceiver/Planner/Critic/Executor/Narrator) with sqlite memory

**M4 — Ship (weeks 9–12)**

- MSIX/deb/rpm installers
- CycloneDX SBOM
- Signed binaries
- EULA flow
- DO-178C-aligned audit log export tool (`qratumctl audit export`)

### Acceptance criteria (definition of done)

1. Cold reboot → user logs in → sees QRATUM, never the host desktop.
2. `state.json.phase` reaches `RUN` within 8 s of logon.
3. Closing UE5 Alt+F4 does **not** drop the user to host shell — Tauri-only fallback engages, watchdog respawns UE5.
4. `qratumctl audit verify` confirms unbroken Merkle chain over 24 h of uptime.
5. Panic key (`Ctrl+Alt+Shift+Q`) restores host shell within 2 s.
6. CHSH probe holds `S = 2√2 ± 1e-4` across the entire session.
7. CIIR rejects an injected destabilizing intent (regression test) with a logged `AuthorizationError`.

---

## Appendix A — File map of v1 deliverables

| New artifact | Path |
|---|---|
| Session orchestrator | `qratum_session/src/main.rs` |
| Watchdog | `qratum_session/src/watchdog.rs` |
| Lock protocol | `qratum_kernel/lock.py` + UE5 `Bridge/LockOwnership.cpp` |
| UE5 RPC plugin | `soi/unreal_bridge/Plugins/QRATUMComm/` |
| AI swarm | `qratum_agents/{perceiver,planner,critic,executor,narrator}.py` |
| Installer | `installer/qratum-{win,linux}.{msix,deb,rpm}` |
| Audit tool | `tools/qratumctl/` |

## Appendix B — Non-goals (for v1)

- Replacing the host kernel.
- Multi-user concurrency (single user per session).
- Mobile or web targets.
- Generic Linux distro shell (only GNOME/KDE session-replace tested).
- Cluster federation (deferred to v2 with NATS/QUIC bus).

---

_This specification is implementation-ready and binds the existing code (Φ=2 swarm + bridge + Tauri shell + boot scripts) into a coherent OS shell layer. Further detail is per-agent: open the corresponding source folder._
