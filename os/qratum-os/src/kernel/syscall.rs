//! Syscall dispatcher (Agent 3, Agent 6, Agent 8).
//!
//! P0.1 entry path: `int 0x80`. The IDT vector has DPL=3 so user code can
//! invoke it (when ring 3 lands in P0.1.1). For now the only caller is
//! kernel demo code (`userland::run_demo`) that prepares an envelope and
//! triggers the gate.
//!
//! ABI (volatile, P0.1):
//!   * RDI = pointer to `IntentEnvelope`
//!   * RSI = pointer to caller-supplied `ResultEnvelope` slot
//!   * RAX (out) = 0 on success, negative errno on dispatch failure
//!
//! The dispatcher itself is called from the `int 0x80` handler in
//! `interrupts.rs`. P0.2 will move to `syscall`/`sysret` MSR-driven entry.

use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::shared::{IntentEnvelope, ResultEnvelope, Verdict, Reason};

use super::audit::{self, FrameKind};
use super::intent;

// Shared slot for the kernel-side trampoline. P0.2 reads RDI/RSI from the
// saved register frame; for P0.1 we have a single in-kernel caller, so we
// stage the envelope/result through atomic pointers.
static IN_ENV:  AtomicPtr<IntentEnvelope>  = AtomicPtr::new(core::ptr::null_mut());
static OUT_RES: AtomicPtr<ResultEnvelope>  = AtomicPtr::new(core::ptr::null_mut());
static CALL_NR: AtomicUsize                = AtomicUsize::new(0);

/// Stage a syscall and trigger the gate. Synchronous in P0.1.
pub fn invoke(env: &IntentEnvelope, out: &mut ResultEnvelope) {
    IN_ENV.store(env as *const _ as *mut _, Ordering::SeqCst);
    OUT_RES.store(out as *mut _, Ordering::SeqCst);
    CALL_NR.store(1, Ordering::SeqCst); // 1 = SYS_INTENT_SUBMIT

    unsafe { core::arch::asm!("int 0x80", options(nostack, preserves_flags)); }
}

/// Called from the `int 0x80` IDT entry. Reads the staged envelope,
/// runs validation + arbitration, writes the result.
pub fn dispatch() {
    let nr = CALL_NR.swap(0, Ordering::SeqCst);
    let env_ptr = IN_ENV.swap(core::ptr::null_mut(), Ordering::SeqCst);
    let out_ptr = OUT_RES.swap(core::ptr::null_mut(), Ordering::SeqCst);

    if nr != 1 || env_ptr.is_null() || out_ptr.is_null() {
        audit::append(FrameKind::SyscallExit, b"bad-args");
        return;
    }

    // Safety: pointers are kernel-owned for the duration of `invoke`.
    let env = unsafe { &*env_ptr };
    let out = unsafe { &mut *out_ptr };

    let r = intent::submit(env);
    *out = r;
    let _ = (Verdict::Accepted, Reason::AcceptFree); // keep enum reachable
}
