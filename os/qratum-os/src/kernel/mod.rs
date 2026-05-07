//! Stage-2 kernel — runs after `ExitBootServices`.
//!
//! From here onwards UEFI services are unavailable. All I/O is direct
//! hardware: 16550 UART for console, port I/O for keyboard. The audit
//! ring lives in kernel BSS; no disk yet (P0.3).

pub mod ahtc_k;
pub mod audit;
pub mod audit_persist;
// P0.6 — receipt anchor + replay checkpoint extensions.
pub mod audit_receipt_anchor;
pub mod audit_replay_checkpoint;
// P0.6 — cryptographic anchors.
pub mod crypto;
pub mod apic;
pub mod gdt;
pub mod heap;
pub mod heartbeat;
// P0.7 — Hardware Reality Layer (real APIC / ACPI / HPET / SMP bring-up).
pub mod hw;
pub mod idt;
pub mod intent;
pub mod interrupts;
pub mod invariants;
pub mod instrumentation;
pub mod kmain;
pub mod paging;
pub mod panic;
pub mod pic;
pub mod pit;
pub mod preempt;
pub mod profiler;
pub mod proof_gate;
pub mod qsh;
pub mod runtime_model;
pub mod sched;
pub mod serial;
pub mod smp;
pub mod sysc;
pub mod syscall;
pub mod tick;
