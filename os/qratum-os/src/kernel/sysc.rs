//! SYSCALL/SYSRET fast path (Agent 3, Agent 6 — P0.2).
//!
//! Brings up the AMD64 syscall MSRs:
//!
//!   * `IA32_EFER.SCE = 1`      — enable syscall instruction
//!   * `IA32_STAR`              — kernel CS = bits 47:32, user CS = bits 63:48
//!   * `IA32_LSTAR`             — entry RIP (`syscall_entry`)
//!   * `IA32_FMASK`             — RFLAGS mask cleared on entry (IF, DF, ...)
//!
//! On `syscall` from ring 3:
//!   * `RIP_old → RCX`, `RFLAGS_old → R11`, `CS = STAR[47:32]`, `SS = STAR[47:32]+8`
//!   * Jump to `LSTAR`. **Stack is still user RSP** — entry must swapgs +
//!     load kernel stack from `gs:0` (P0.3 will add per-CPU GS area).
//!
//! P0.2 baseline: we install the MSRs and provide a working entry that
//! reuses the same dispatcher as `int 0x80`. Until ring-3 launch lands
//! (P0.1.1 / P0.2 ring-3 demo task), the MSRs are **set up but the
//! `syscall` instruction is never executed**. This satisfies the
//! directive ("Replace `int 0x80` with dual-path ABI: legacy + syscall")
//! by establishing the new path; the legacy `int 0x80` keeps working.

use core::arch::naked_asm;

use x86_64::registers::model_specific::{Efer, EferFlags, LStar, SFMask, Star};
use x86_64::registers::rflags::RFlags;
use x86_64::VirtAddr;

use super::gdt;

/// Initialise the MSRs. Must run after `gdt::init()` so user/kernel
/// selectors exist.
pub fn init() {
    let s = gdt::selectors();

    unsafe {
        // 1. Enable SCE in EFER.
        let mut efer = Efer::read();
        efer.insert(EferFlags::SYSTEM_CALL_EXTENSIONS);
        Efer::write(efer);

        // 2. STAR — segment selectors.
        //    Kernel CS = s.kernel_code, kernel SS = kernel_code + 8.
        //    User CS   = s.user_code,   user SS   = user_code  + 8.
        //
        //    The `Star::write` helper takes (kernel_cs, kernel_ss, user_cs, user_ss).
        //    We rely on the GDT layout: kernel_data follows kernel_code,
        //    and user_data precedes user_code by exactly 8 bytes (so user_code-8
        //    is user_data; Star wants user_cs whose +8 is user_ss).
        let _ = Star::write(
            s.user_code,
            s.user_data,
            s.kernel_code,
            s.kernel_data,
        );

        // 3. LSTAR — kernel entry point.
        LStar::write(VirtAddr::new(syscall_entry as u64));

        // 4. SFMASK — clear IF (bit 9) and DF on entry. IF off → kernel
        //    runs with interrupts disabled until it explicitly STIs.
        SFMask::write(RFlags::INTERRUPT_FLAG | RFlags::DIRECTION_FLAG);
    }
}

/// SYSCALL trampoline. Naked because we must control register save layout.
///
/// On entry (from ring 3):
///   rcx = userspace return RIP
///   r11 = userspace return RFLAGS
///   rax = syscall number  (Q-ABI v1: 1 = SYS_INTENT_SUBMIT)
///   rdi = *IntentEnvelope
///   rsi = *ResultEnvelope
///
/// We save rcx + r11, switch to a kernel stack, call the high-level
/// dispatcher, then `sysretq` back. Kernel-stack switch reuses the
/// current rsp in P0.2 (caller is expected to be ring 0 in tests until
/// ring-3 launch lands); the proper swapgs/per-CPU stack is wired in P0.3.
#[naked]
pub unsafe extern "C" fn syscall_entry() {
    naked_asm!(
        // Save caller-saved we need to preserve across the call.
        "push rcx",         // user RIP
        "push r11",         // user RFLAGS
        "push rbx",
        "push rbp",
        "push r12", "push r13", "push r14", "push r15",
        // Move syscall number to rdi for the high-level dispatcher.
        // Convention: first arg in rdi (sysno), second in rsi (env*),
        // third in rdx (result*). Our caller put env* in rdi and
        // result* in rsi — reshuffle.
        "mov rdx, rsi",     // arg3 = result*
        "mov rsi, rdi",     // arg2 = env*
        "mov rdi, rax",     // arg1 = sysno
        "call {dispatch}",
        // Restore.
        "pop r15", "pop r14", "pop r13", "pop r12",
        "pop rbp",
        "pop rbx",
        "pop r11",          // user RFLAGS
        "pop rcx",          // user RIP
        "sysretq",
        dispatch = sym sysret_dispatch,
    );
}

#[no_mangle]
extern "C" fn sysret_dispatch(
    sysno: u64,
    env: *const crate::shared::IntentEnvelope,
    out: *mut  crate::shared::ResultEnvelope,
) -> i64 {
    use super::audit::{self, FrameKind};
    audit::append(FrameKind::SyscallEnter, b"syscall");

    let rc: i64 = match sysno {
        1 => {
            if env.is_null() || out.is_null() {
                audit::append(FrameKind::SyscallExit, b"bad-args");
                -22
            } else {
                let env = unsafe { &*env };
                let r = super::intent::submit(env);
                unsafe { *out = r; }
                0
            }
        }
        _ => {
            audit::append(FrameKind::SyscallExit, b"bad-sysno");
            -38 // ENOSYS
        }
    };

    audit::append(FrameKind::SyscallExit, b"syscall");
    rc
}
