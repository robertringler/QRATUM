# QRATUM CORE OS — Boot Kernel Transformation v1.0

**Status:** ratified by 12-agent systems-engineering swarm
**Scope:** transforms QRATUM from Windows-hosted Python control plane into
          a bootable, kernel-level, intent-arbitrating operating system.
**Predecessors (preserved as user-space contracts):**
  v1.0 wire · v1.1 autonomy · v1.2 console · arbitration v1.0
**Authority:** Agent 12 (System Synthesis Engineer)

---

## 0. Doctrine

QRATUM CORE OS is **not a Linux distribution and not a microkernel research
toy**. It is a deterministic, AI-aware operating system whose central
abstraction is the **intent**, not the process. Every CPU cycle is the
result of an arbitrated, logged, replayable intent. Traditional concepts
(threads, files, syscalls) exist only as low-level mechanisms supporting
that abstraction.

```
                 ┌─────────────────────────────────────────────┐
                 │           UEFI firmware                     │
                 └────────────────┬────────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────────┐
                 │      qratum-boot.efi  (Stage 1)             │
                 │  load kernel ELF, fetch memory map,         │
                 │  ExitBootServices, jump to kernel_main      │
                 └────────────────┬────────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────────┐
                 │      qratum-kernel  (Ring 0, Rust no_std)   │
                 │  ┌────────────┬────────────┬────────────┐   │
                 │  │  Arbiter   │  Scheduler │  Replay    │   │
                 │  ├────────────┴────────────┴────────────┤   │
                 │  │   Intent Syscall Dispatcher           │   │
                 │  ├────────────────────────────────────────┤   │
                 │  │ Memory · IRQ · Timer · IO · Net · GPU  │   │
                 │  └────────────────────────────────────────┘   │
                 └────────────────┬────────────────────────────┘
                                  ▼
                 ┌─────────────────────────────────────────────┐
                 │  User space (Ring 3)                        │
                 │   /svc/autonomy   /svc/replay               │
                 │   /bin/qsh        /bin/qctl                 │
                 │   /opt/ue5  (optional renderer)             │
                 └─────────────────────────────────────────────┘
```

Target architecture: **x86_64 UEFI** (Phase 0). ARM64 is a Phase 2 port.

---

## 1. Invariants

| # | Invariant | Owner |
|---|-----------|-------|
| K1 | Every state mutation in the system is the result of an intent dispatched through the kernel arbiter | A8, A4 |
| K2 | The arbiter is a pure function of `(intent, lock_table, monotonic_tick)`; replay reproduces every verdict bit-for-bit | A8, A9 |
| K3 | The kernel never allocates after boot in arbitration paths — all per-intent work uses fixed slabs | A2, A5 |
| K4 | Autonomy runs in Ring 3 inside a containment sandbox; it can *propose* but never *commit* without arbiter approval | A10, A7 |
| K5 | Every intent, verdict, and lock event is appended to the kernel audit log before user-visible side-effects | A9 |
| K6 | The console (qsh) is just another arbitrated client — no special path, identical syscall ABI | A11 |
| K7 | Boot is deterministic: same firmware + same disk image + same `replay.qaudit` → identical kernel state at hand-off | A1, A9, A12 |
| K8 | UE5 and any renderer are user-space; pulling the GPU driver or killing UE must not affect arbitration | A6, A7 |

---

## 2. Boot Sequence (Agent 1)

### 2.1 Stages

```
[UEFI POST]
   │
   ▼
[Stage 1: qratum-boot.efi]      ← UEFI application, PE/COFF
   • Locate \EFI\QRATUM\kernel.elf on ESP
   • Validate SHA-256 against \EFI\QRATUM\kernel.hash
   • Load kernel ELF segments into identity-mapped memory
   • Acquire UEFI memory map
   • Acquire framebuffer descriptor (GOP)
   • Acquire RNG seed (EFI_RNG_PROTOCOL) for non-deterministic ops only
   • Build BootInfo struct (see §2.3)
   • ExitBootServices()
   • Switch CR0/CR4, enable long-mode paging (already on under UEFI)
   • Jump to kernel_main(BootInfo*)
   │
   ▼
[Stage 2: qratum-kernel]
```

### 2.2 Memory layout (post hand-off)

| Range                                  | Purpose                       |
|----------------------------------------|-------------------------------|
| `0x0000_0000_0010_0000` … `0x0000_0000_07FF_FFFF` | Kernel image + early heap (≤128 MB)      |
| `0xFFFF_8000_0000_0000` … `0xFFFF_8000_3FFF_FFFF` | Direct-map of physical RAM (HHDM, 1 GB pages) |
| `0xFFFF_FF00_0000_0000` …              | Kernel heap (slab allocator) |
| Userspace: `0x0000_0000_0040_0000` …   | Per-process address space    |

### 2.3 BootInfo contract

```rust
#[repr(C)]
pub struct BootInfo {
    pub magic: u64,              // = 0x51_52_41_54_55_4D_42_4F  ("QRATUMBO")
    pub mem_map_ptr: *const E820Entry,
    pub mem_map_len: usize,
    pub framebuffer: FrameBuffer,
    pub rng_seed: [u8; 32],
    pub audit_log_lba: u64,      // ESP location of replay.qaudit
    pub audit_log_size: u64,
    pub command_line: [u8; 256], // e.g. "replay=1 hz=60 console=ttyS0"
}
```

---

## 3. Kernel Core (Agent 2)

### 3.1 Language & runtime

* **Rust 1.78+, `no_std`, `alloc` after slab init.**
  Chosen for memory safety, ownership-typed lock primitives, and zero-cost
  abstractions over hardware. C is permitted only in the bootloader stub
  and a handful of inline-asm shims (`cli`, `sti`, `cr3`, `lgdt`, etc.).
* **No dynamic allocation** in the arbitration hot path.
* `panic = "abort"`. Kernel panic produces a final audit-log frame and
  halts CPU (`hlt` in IPI-broadcast loop).

### 3.2 Module map

```
kernel/
├── boot/        // entry, GDT/IDT/TSS, paging bootstrap
├── arch/x86_64/ // CPU specifics, MSRs, APIC, ACPI
├── mm/          // physical frame allocator, virtual memory, slabs
├── irq/         // IDT vectors, IPIs, syscall MSR setup
├── time/        // TSC + HPET monotonic ticker, deterministic clock
├── proc/        // process/thread/scheduler state, context switch
├── arbiter/     // §6 — kernel-resident arbitration core
├── intent/      // §4 — syscall ABI, validator, dispatcher
├── audit/       // §8 — append-only ring + on-disk log
├── dev/         // §5 — drivers (kbd, blk, net, gpu, console UART)
├── svc/         // §7 — kernel services (autonomy proxy, replay, time)
└── main.rs      // kernel_main entry
```

### 3.3 IDT / syscall

* **`SYSCALL`/`SYSRET`** path on x86_64 (MSR_LSTAR → `intent_syscall_entry`).
* Legacy `int 0x80` retained only for early debug builds; removed in
  release.
* IDT vectors 0–31 architectural, 32–63 IRQs (legacy PIC remapped),
  64–127 IPIs, 128 reserved for `int $QRATUM_DEBUG` traps.

---

## 4. Intent System Calls (Agent 3)

### 4.1 The single syscall

QRATUM exposes **exactly one user-visible syscall**: `intent_submit`.
Everything else (open, read, write, mmap, fork) is a kernel-internal
intent issued by trusted services.

```
RAX = SYS_INTENT_SUBMIT (= 1)
RDI = ptr to IntentEnvelope (user)
RSI = sizeof IntentEnvelope
RDX = ptr to ResultEnvelope (user, kernel writes back)
RCX = sizeof ResultEnvelope
return:
  RAX = 0       → submitted; consult ResultEnvelope.verdict
  RAX = -EINVAL → schema rejection
  RAX = -EFAULT → bad pointer
  RAX = -EBUSY  → queue full (back-pressure)
```

### 4.2 Binary envelope (kernel-canonical)

```rust
#[repr(C)] pub struct IntentEnvelope {
    pub magic: u32,         // = 0x49_4E_54_31  ("INT1")
    pub version: u16,       // = 1
    pub flags: u16,         // bit0 = override, bit1 = system
    pub id: [u8; 16],       // 128-bit ULID, monotonic prefix
    pub ts: u64,            // kernel monotonic tick at submission
    pub authority: u8,      // 1=autonomy 2=external 3=console 4=system
    pub scope: u8,          // 1=actor 2=world 3=system
    pub priority: u8,       // 1=low 2=normal 3=high 4=override
    pub _pad0: u8,
    pub name_len: u16,
    pub params_len: u32,
    pub name: [u8; 64],     // dotted, e.g. b"vehicle.set_throttle\0..."
    pub params_ptr: u64,    // user pointer to CBOR/raw blob, kernel copies
}
#[repr(C)] pub struct ResultEnvelope {
    pub id: [u8; 16],       // matches request
    pub verdict: u8,        // 0=accepted 1=denied
    pub reason: u8,         // arbitration reason code (§6.4)
    pub holder_authority: u8,
    pub _pad: u8,
    pub error_code: i32,    // 0 on accept, errno-style on deny
    pub data_ptr: u64,      // optional payload back to caller
    pub data_len: u32,
    pub completed_ts: u64,
}
```

### 4.3 Dispatcher

```
intent_syscall_entry:
    swapgs
    save_user_context
    rdi/rsi/rdx/rcx = args
    call intent::dispatch
    restore_user_context
    swapgs
    sysretq
```

`intent::dispatch`:

1. Copy envelope into kernel slab `IntentSlot` (fixed 4 KiB pool).
2. **Validate** schema, name, params length, namespace allow-list.
3. **Tag** authority/scope (compiler-style), reject illegal `flags` for
   the caller's privilege ring.
4. Append `INTENT_SUBMIT` audit frame.
5. Hand to arbiter (§6).
6. On accept, enqueue on the target subsystem run-queue (§5).
7. On deny, write synthetic result, append `INTENT_DENY` frame, return.

---

## 5. Scheduler (Agent 4)

### 5.1 Scheduling classes

| Class       | Sources              | Quantum | Preemptable by |
|-------------|----------------------|---------|----------------|
| `SCHED_SYS` | system intents, IRQs | ∞       | none           |
| `SCHED_CON` | console (qsh)        | 5 ms    | SYS            |
| `SCHED_EXT` | external clients     | 2 ms    | SYS, CON       |
| `SCHED_AUT` | autonomy service     | 1 ms    | SYS, CON, EXT  |
| `SCHED_BG`  | renderer, telemetry  | best-effort | all       |

Scheduling is **not fair** — it is deterministic-by-authority. Within a
class, ordering is the arbitration tuple `(priority, scope, ts, id)`.

### 5.2 Run queues

Per-CPU, per-class, MPMC bounded ring buffers. No global lock; lock-free
SPSC for per-CPU enqueue, work-stealing across CPUs only for `SCHED_BG`.

### 5.3 Resource locks (kernel embodiment of §arbitration v1.0)

```rust
pub struct ResLock {
    pub key: LockKey,            // hashed canonical lock_key
    pub holder: AuthorityId,
    pub holder_id: HolderHandle,
    pub acquired_tick: u64,
    pub expires_tick: u64,
    pub priority: Priority,
    pub refresh_count: u32,
}
```

A flat hash table sized at boot from BootInfo (default 65 536 slots).
Eviction follows §6.

---

## 6. Arbitration Kernel Core (Agent 8)

The arbiter from arbitration v1.0 moves into Ring 0 verbatim. The
algorithm is unchanged; the data path is rewired:

```
intent::dispatch
   └── arbiter::evaluate(envelope, &mut lock_table, tick) -> Verdict
```

### 6.1 Determinism

* `tick` is a monotonic counter incremented at every `LOC` (Lock-Order
  Commit) point, **not** wall-clock. Replay deterministically advances
  `tick` from the audit log.
* RNG (if any policy needs it) is drawn from a per-boot ChaCha20 seeded
  from `BootInfo.rng_seed` and audit-logged on every consumption.

### 6.2 Sort key (kernel)

`(-authority_rank, -priority_rank, +scope_rank, +ts, +id_lex)` — same as
user-space arbitration v1.0. Verified by replay test in CI.

### 6.3 Dominance windows (in ticks at default 1000 Hz)

system 0 · console 5000 · external 1000 · autonomy 250.

### 6.4 Reason codes (kernel ABI; stable)

```
0 ACCEPT_FREE          5 DENY_AUTONOMY_BLOCKED
1 ACCEPT_REFRESH       6 DENY_TIMESTAMP
2 ACCEPT_AUTHORITY     7 DENY_OVERRIDE_THRASH
3 ACCEPT_SYSTEM_OVR    8 DENY_QUEUE_FULL
4 DENY_CONSOLE_DOMIN   9 DENY_SCHEMA
```

---

## 7. Process Model (Agent 7)

### 7.1 Process struct

```rust
pub struct Proc {
    pid: Pid,
    parent: Pid,
    authority_grant: AuthorityMask, // which authorities this proc may claim
    address_space: Cr3,
    sched_class: SchedClass,
    state: ProcState,               // Running | Ready | Blocked | Zombie
    intent_quota: TokenBucket,
    audit_cursor: AuditOffset,
}
```

### 7.2 Boot processes (PID 1..N, hard-coded)

| PID | Image                      | Authority   | Purpose                |
|-----|----------------------------|-------------|------------------------|
| 1   | `/svc/init`                | SYSTEM      | mount fs, spawn rest   |
| 2   | `/svc/audit`               | SYSTEM      | flush audit ring → disk|
| 3   | `/svc/replay` (if `replay=1`) | SYSTEM   | feeds historical intents |
| 4   | `/svc/autonomy`            | AUTONOMY    | planner/critic v1.1    |
| 5   | `/bin/qsh` on console UART | CONSOLE     | shell                  |
| 6+  | optional `/opt/ue5/QRATUM` | EXTERNAL    | renderer (sandboxed)   |

PID 1 cannot be killed; its death triggers `KERNEL_PANIC_INIT_LOST`.

### 7.3 No `fork`

Process creation is its own intent: `proc.spawn{image, authority,
sched_class}` — only PID 1 may issue it. This eliminates ambient
authority and fork-bomb classes.

---

## 8. Replay & Audit (Agent 9)

### 8.1 Audit log layout

A typed, append-only ring of fixed-size frames (64 B header + variable
body) backed by a dedicated **GPT partition** with type GUID
`51524154-554D-4155-4449-540000000001`.

Frame types:

```
0x01 INTENT_SUBMIT
0x02 INTENT_VERDICT   (arbiter result)
0x03 LOCK_ACQUIRE
0x04 LOCK_REFRESH
0x05 LOCK_EVICT
0x06 LOCK_EXPIRE
0x10 PROC_SPAWN
0x11 PROC_EXIT
0x20 IRQ_DELIVER       (only when policy-relevant)
0x80 RNG_DRAW
0xFE TICK_ADVANCE      (boundary marker every N ticks)
0xFF KERNEL_PANIC
```

Each frame includes a Blake3 hash of the previous frame → tamper-evident
chain. The kernel mirrors hot frames in a 16 MiB ring; a flusher core
DMAs them to disk.

### 8.2 Replay boot mode

`command_line = "replay=1 audit=/EFI/QRATUM/golden.qaudit"`

Kernel boots normally up to scheduler init, then:

1. Disables wall-clock derived advances.
2. `/svc/replay` is the **only** producer of intents until EOF.
3. Each frame replays in order; arbitration verdicts must match
   recorded verdicts byte-for-byte. Mismatch → `KERNEL_PANIC_REPLAY_DRIFT`.
4. On EOF the kernel either halts (`replay=halt`) or transitions to
   live mode (`replay=resume`).

This gives full state reconstruction from any audit log to the exact
tick.

---

## 9. Memory Management (Agent 5)

* **Physical frame allocator:** buddy, 4 KiB / 2 MiB / 1 GiB orders.
* **Kernel heap:** segregated slab allocator with size classes
  {16, 32, 64, 128, 256, 512, 1024, 2048, 4096} bytes; per-CPU magazines.
* **Userspace:** demand-paged, copy-on-write disabled (no fork → no need),
  mmap-style allocations issued via `mem.map` intents.
* **Lock-aware regions:** memory associated with a `lock_key` carries a
  back-reference; eviction of the lock invalidates write capability via
  page-table flag flip + TLB shootdown.

---

## 10. Devices (Agent 6)

| Class      | Phase 0 driver                      | Notes                               |
|------------|-------------------------------------|-------------------------------------|
| Console    | 16550 UART + GOP framebuffer text   | `/dev/console`                      |
| Keyboard   | PS/2 + USB HID (xHCI later)         | translates to `input.key` intents   |
| Block      | NVMe (PCIe MMIO)                    | audit log + rootfs                  |
| Net        | virtio-net (for VM bring-up first)  | external-client transport           |
| GPU        | none in kernel; UE5 talks to        | Phase 0: no GPU; Phase 1: VirtIO-GPU|
| Timer      | TSC deadline + HPET fallback        | drives tick                         |

All drivers issue intents (`disk.read`, `net.tx`) and are themselves
arbitrated — even hardware access is governed.

---

## 11. Safety & Isolation (Agent 10)

* **Authority masks** are baked into each process at spawn; raising the
  mask requires a `system` intent signed by a kernel-known key (boot
  measurement of `kernel.hash`).
* **Autonomy sandbox:** Ring 3, no MMIO map, syscall budget rate-limited
  (`intent_quota`). Every denied intent decrements goal priority (v1.2 §6).
* **Lock starvation:** anti-starvation drop after `2 × dominance_window`
  with no winning intent (v1.0 arbitration §6).
* **Override thrash:** ≥ 50 acquisitions of the same key in 5 s blocks
  further acquisitions for 5 s; logged as `DENY_OVERRIDE_THRASH`.
* **Panic surface:** divide-by-zero, page-fault in arbiter path,
  audit-log write failure, replay drift, init death → all
  `kernel_panic` with final audit frame + halt.

---

## 12. Userspace Shell (Agent 11)

`qsh` is a minimal REPL implemented over `intent_submit`:

```
qsh> v.set_throttle truck_17 0.5
[ARB] verdict=ACCEPTED  reason=ACCEPT_FREE  lock=actor:truck_17  by=console
qsh> .locks
key                 holder   priority  expires_in
actor:truck_17      console  normal    4.8s
qsh> .replay tail 5
…
```

Built-in dot-commands (`.locks`, `.replay`, `.proc`, `.audit`, `.q`) map
to system-namespace intents the kernel understands. No external
executables are needed for v1.

---

## 13. Synthesis (Agent 12)

### 13.1 Subsystem init order

```
1.  GDT/IDT/TSS, basic paging  (Stage 1 hand-off)
2.  Physical frame allocator   (parse memory map)
3.  Kernel heap (slabs)
4.  Audit ring + early disk driver  (so frames flow before scheduler)
5.  Arbiter + lock table        (still no preemption)
6.  Scheduler classes & run queues
7.  Timer + IRQ enable          (preemption begins)
8.  Drivers (UART, kbd, NVMe, virtio-net)
9.  PID 1 spawn  (init)
10. PID 2 (audit flusher), 3 (replay if requested), 4 (autonomy), 5 (qsh)
11. Optional renderer
```

Steps 1–4 must complete before any intent enters the arbiter; otherwise
audit determinism is violated.

### 13.2 Determinism guarantees collapsed

| Source of nondeterminism      | Neutralization                      |
|-------------------------------|-------------------------------------|
| Wall-clock skew               | tick-only ordering                  |
| Multi-core arrival order      | per-CPU MPSC + arbiter linearization|
| RNG                           | seeded ChaCha20, audited            |
| Memory layout randomness      | KASLR off in replay mode            |
| IRQ timing                    | IRQs only mark ticks, do not order  |
| User input typing speed       | input → intents arbitrate normally  |

---

## 14. Build & Distribution

```
artifacts/
├── qratum-boot.efi              (UEFI app, ~80 KiB)
├── kernel.elf                   (kernel image, ~2 MiB stripped)
├── kernel.hash                  (sha256)
├── initrd.qfs                   (services + qsh, ~1 MiB)
├── qratum-os-x86_64.iso         (bootable ISO: ESP + GPT data parts)
└── qratum-os-x86_64.img         (USB / disk image, dd-able)
```

Build host: cross-compiles via Cargo `x86_64-unknown-none` target +
`cargo-xbuild`; bootloader via `uefi-rs`. CI gate: `kernel.hash` is
reproducible (deterministic build flags).

---

## 15. Phase Plan

| Phase | Deliverable                                                   | Gate                     |
|-------|---------------------------------------------------------------|--------------------------|
| P0.0  | UEFI Hello World, audit ring stub, halt                       | boots in QEMU            |
| P0.1  | Memory + slabs + IDT + serial console                         | kernel-panic visible     |
| P0.2  | Intent syscall ABI + arbiter port from arbitration v1.0       | qsh-on-UART issues intents|
| P0.3  | Scheduler + procs + audit on NVMe                             | replay round-trips       |
| P0.4  | Autonomy as Ring 3 service (port `autonomy.py` to Rust)       | v1.1 parity              |
| P0.5  | Console hub parity (port `console.py`)                        | v1.2 parity              |
| P0.6  | virtio-net for external clients                               | qctl-over-network        |
| P1.0  | VirtIO-GPU + UE5 user-space app                               | renderer optional         |
| P1.1  | ARM64 port                                                    | boots on Pi 5             |
| P2.0  | Verified boot (TPM-attested kernel.hash)                      | secure boot chain         |

Phase 0 alone is the equivalent of a small-team kernel. Estimate:
several thousand lines of Rust, primarily in `kernel/arbiter`,
`kernel/intent`, `kernel/proc`, and `kernel/audit`.

---

## 16. Acceptance (boot-OS scope)

1. Cold boot of `qratum-os-x86_64.iso` in QEMU UEFI lands at a `qsh>`
   prompt over serial within 2 s on a modern host.
2. `intent_submit` issued from `qsh` produces an arbitration verdict
   identical to user-space arbitration v1.0 for an equivalent input.
3. Capturing the audit log of a 60 s session and re-booting with
   `replay=1` reproduces the final lock table byte-for-byte.
4. Killing autonomy (PID 4) leaves console + arbiter alive; respawn is
   automatic and journaled.
5. `/svc/audit` flushes to NVMe at ≥ 50 MB/s without dropping frames
   under a 100 k intents/s burst.
6. UE5 attached as PID 6 has zero ability to acquire `system`-authority
   locks, verified by negative test.
7. Removing the GPU / killing UE does not produce any arbiter event
   beyond a `PROC_EXIT` frame.

---

## 17. What this **is not** (honest scope)

* Not a microkernel research project: drivers live in-kernel for v1.
* Not POSIX-compatible: there is no `open/read/write/fork`.
* Not a Linux fork: zero Linux source code is used.
* Not yet a security-hardened OS: TPM attestation, secure boot, and
  formal verification are Phase 2.
* Not a Python runtime: the existing `qratum_launcher.py` becomes a
  *reference implementation* used to generate and validate the kernel
  arbiter test corpus, not a runtime dependency.

---

## 18. Migration of existing code

| Current artefact                       | Fate                                            |
|----------------------------------------|-------------------------------------------------|
| `qratum_launcher/qratum_launcher.py`   | reference impl + golden corpus generator (CI)   |
| `qratum_launcher/arbitration.py`       | spec-equivalent kernel `arbiter` crate          |
| `qratum_launcher/autonomy.py`          | port to Rust as `/svc/autonomy`                 |
| `qratum_launcher/console.py`           | port to Rust as part of `qsh`                   |
| UE bridge (file-IPC)                   | replaced by `intent_submit` over virtio-vsock   |
| Windows scheduled tasks                | obsolete on bare metal                          |
| Tauri desktop                          | retained only as remote viewer (network client) |

The user-space Python stack remains usable on Windows/Linux as a
*development simulator* of the same arbitration semantics.

---

**End of QC_CORE_OS_BOOT_KERNEL_V1.md**
