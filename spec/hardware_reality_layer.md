# P0.7 — Hardware Reality Layer Specification

**Status**: Scaffolding shipped. Bring-up dormant (not yet wired into `kmain`)
pending QEMU + bare-metal validation.
**Exit criterion (verbatim from directive)**: *"Kernel boots on physical
machine with multi-core scheduling active."*

This document satisfies the per-step deliverable format:
1. subsystem → 2. real hardware dependency → 3. implementation order →
4. failure modes under real hardware → 5. test for physical validation.

---

## Subsystem map

| # | Subsystem | File | HW dependency | Failure modes (real) | Validation |
|---|-----------|------|---------------|----------------------|------------|
| 1 | CPUID detect | `kernel/hw/cpuid.rs` | `cpuid` insn (CPL0, always available) | None — runs on any x86_64 CPU | Boot log: vendor + flags |
| 2 | RDMSR/WRMSR | `kernel/hw/msr.rs` | MSR mechanism (Pentium+) | `#GP` if MSR unimplemented (caller must gate by CPUID) | Read IA32_APIC_BASE; expect non-zero |
| 3 | ACPI walker | `kernel/hw/acpi.rs` | Firmware-installed RSDP via UEFI ConfigurationTable | Bad checksum; missing MADT (legacy systems w/o ACPI) | Boot log: LAPIC base + LAPIC count match `nproc` |
| 4 | xAPIC MMIO | `kernel/hw/lapic_mmio.rs` | LAPIC at MADT-advertised PA, mapped UC | Spurious IPIs if SIVR not enabled; wrong base → silent reads of 0xFF | Send self-IPI; verify EOI clears ISR |
| 5 | x2APIC MSR | `kernel/hw/x2apic_msr.rs` | CPUID.01h:ECX[21] = 1 + IA32_APIC_BASE.X2APIC_EN | `#GP` if x2APIC bit set on a CPU lacking it | LAPIC ID via MSR 0x802 must match BSP CPU index |
| 6 | APIC backend selector | `kernel/hw/apic_real.rs` | Composes #4/#5 | Mis-detection → IPIs ignored | Active backend in `last_report()` |
| 7 | HPET driver | `kernel/hw/hpet.rs` | ACPI HPET table; MMIO at advertised base | Counter freeze if main CONFIG_ENABLE not set | `counter()` strictly monotonic over 1 ms wall clock |
| 8 | Frame allocator | `kernel/hw/frame_alloc.rs` | UEFI memory map staged by stage1 | Range collision with UEFI runtime → `#PF` on first dereference | Allocate 16 frames; verify uniqueness; free; re-allocate |
| 9 | Kernel paging | `kernel/hw/kernel_paging.rs` | CR3 writable post-handoff; current code/stack within identity 1 GiB | `#PF` mid-`take_ownership` if PML4 entry 0 not pre-populated → triple fault | After CR3 swap: read kernel symbol; expect identical bytes |
| 10 | INIT/SIPI driver | `kernel/hw/init_sipi.rs` | Real LAPIC #6, working timer #7 | AP triple-faults if trampoline frame corrupt; AP misses SIPI window if delays wrong | AP increments `ap_boot::AP_COUNT`; receipt INV-T emitted |
| 11 | AP trampoline blob | `kernel/hw/ap_trampoline.rs` | <1 MiB physical frame, identity-mapped, USER+RWX | Wrong vector → AP enters reset code; missing CLI → triple fault on first IRQ | AP halts cleanly (HLT loop); no triple-fault telemetry from chipset |

**Determinism boundary**: items 1–11 are inherently non-deterministic (real
silicon timing). Per the directive, deterministic envelopes (`runtime_model`,
audit, ahtc_k) **do not bound this layer** — they consume its events.

---

## Implementation order (followed in this slice)

```
msr + cpuid                      ✅ shipped
  → lapic_mmio + x2apic_msr      ✅ shipped
    → apic_real                  ✅ shipped
      → acpi                     ✅ shipped
        → hpet                   ✅ shipped
          → frame_alloc          ✅ shipped
            → kernel_paging      ✅ shipped
              → ap_trampoline    ✅ shipped (real-mode HLT stub)
                → init_sipi      ✅ shipped (full INIT/SIPI/SIPI sequence)
```

**Gating wire-up into `kmain`**: `pub mod hw;` declared (build-clean), but
`hw::init()` is *not* yet called from `kmain::start()`. It is the next slice's
single edit — held until QEMU SMP boot is captured on this machine and the
bare-metal procedure (§Validation) succeeds at least once.

---

## Failure modes under real hardware

The table above lists per-subsystem failure modes. Three classes dominate:

* **Triple-fault on CR3 swap** (#9). Mitigation: PML4 entry 0 maps a 1 GiB
  huge page covering all current kernel code/stack/data before the write.
  If the kernel image, stacks, or UEFI runtime sit above 1 GiB on the target
  hardware, `take_ownership` must be widened first.
* **AP startup window miss** (#10). Intel SDM mandates 10 ms after INIT and
  200 µs after each SIPI. Timing is HPET-driven when present; the busy-loop
  fallback (`3000 cycles/µs`) is calibrated for ≥3 GHz CPUs and may
  under-wait on slower silicon.
* **MSR `#GP`** on x2APIC writes (#5). Backend selector gates by CPUID; an
  AP whose CPUID disagrees with the BSP (rare but seen on heterogeneous
  hybrid CPUs) would raise `#GP` on entering long mode. Detection: the
  AP-side init must re-read CPUID before touching x2APIC MSRs.

---

## Validation

### Model-level (CI / QEMU)

* `tools/hardware/qemu_smp_boot.ps1` — boots `qratum.efi` under OVMF with
  `-smp 4`, captures serial console, asserts the boot banner reaches
  `BootReady` and (once `hw::init` is wired) `aps_started == 3`.

### Bare-metal (user-executed)

* `tools/hardware/baremetal_test_procedure.md` — verbatim runnable steps for
  silicon validation. Required because the agent cannot drive physical USB
  imaging or serial capture.

---

## Out of scope for P0.7

* IO-APIC programming (MADT entries collected but not yet consumed).
* TSC-deadline LAPIC timer (CPUID gate present; programming deferred).
* Demand paging on `#PF` (handler logs only; no map-on-fault yet).
* Real long-mode bridge in the AP trampoline (current blob halts the AP
  cleanly; long-mode jump is the immediately-next slice).

These are tracked for P0.7-followup and **must complete before declaring
P0.7 exit**, since exit demands "multi-core scheduling **active**", which
requires APs to reach `ap_entry` in long mode and pull from
`smp::deterministic_work_steal`.
