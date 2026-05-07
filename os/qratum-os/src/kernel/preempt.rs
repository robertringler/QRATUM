//! Preemptive context switch — naked IRQ0 trampoline (P0.2).
//!
//! Path:
//!   1. CPU enters timer IRQ — pushes RIP/CS/RFLAGS/RSP/SS (iretq frame).
//!   2. Trampoline pushes all 15 GPRs (rax..r15) onto the same stack.
//!   3. Saves the resulting `rsp` into the running task's `Tcb.rsp`.
//!   4. Calls the Rust picker which returns the next task's saved `rsp`.
//!   5. Loads it, pops GPRs, `iretq`s.
//!
//! Tasks marked `sched_class = 1` use this path. Cooperative tasks
//! (`sched_class = 0`) keep the `ret`-compatible stack layout used by
//! `sched::context_switch`; the timer falls back to setting
//! `NEED_RESCHED` for those.
//!
//! **Honest scope:** the trampoline is implemented and live, but
//! heartbeat (the demonstrably preemptable task) is migrated to
//! preemptive class only after a "warm-up" tick. qsh stays cooperative
//! so that interactive REPL paths never enter naked-asm territory while
//! holding spinlocks.

#![allow(dead_code)]

use core::arch::naked_asm;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::audit::{self, FrameKind};
use super::sched::{TaskState, MAX_TASKS, SCHED};
use crate::shared::Verdict;

static PREEMPT_COUNT: AtomicUsize = AtomicUsize::new(0);
pub fn count() -> usize { PREEMPT_COUNT.load(Ordering::Relaxed) }

/// Picker called from the trampoline. Returns the next task's saved
/// `rsp` so the trampoline can switch stacks. If no preemptive task is
/// runnable, returns the input `rsp` (no-op switch).
///
/// # Safety
/// Called only from the naked trampoline with valid scheduler state.
#[no_mangle]
unsafe extern "C" fn preempt_pick(prev_rsp: u64) -> u64 {
    let mut s = SCHED.lock();
    let cur = s.current;

    // Refuse to preempt cooperative tasks — they cannot resume from an
    // iretq frame because their stack layout is `ret`-compatible.
    if s.tasks[cur].sched_class != 1 {
        return prev_rsp;
    }

    // Pick highest-priority preemptive ready task other than self.
    let mut best: Option<usize> = None;
    let mut best_prio: i32 = -1;
    for offset in 1..=MAX_TASKS {
        let i = (cur + offset) % MAX_TASKS;
        let t = &s.tasks[i];
        if t.state == TaskState::Ready
            && t.verdict == Verdict::Accepted
            && t.sched_class == 1
        {
            if (t.prio as i32) > best_prio {
                best = Some(i);
                best_prio = t.prio as i32;
            }
        }
    }
    let next = match best { Some(n) => n, None => return prev_rsp };
    if next == cur { return prev_rsp; }

    // Save current rsp; transition states.
    s.tasks[cur].rsp = prev_rsp;
    if s.tasks[cur].state == TaskState::Running {
        s.tasks[cur].state = TaskState::Ready;
    }
    s.tasks[next].state = TaskState::Running;
    s.current = next;
    let next_rsp = s.tasks[next].rsp;

    audit::append(FrameKind::Preempt, &[cur as u8, next as u8]);
    audit::append(FrameKind::ContextSwitch, &[cur as u8, next as u8]);
    PREEMPT_COUNT.fetch_add(1, Ordering::Relaxed);
    next_rsp
}

/// Naked IRQ0 entry — saves full GPR set, calls picker, restores, iretq.
///
/// Installed by `idt::init_preemptive_timer()` as the IDT vector for
/// IRQ_TIMER, replacing the cooperative `extern "x86-interrupt" fn timer`
/// once the kernel decides to enable preemption.
#[naked]
pub unsafe extern "C" fn timer_trampoline() {
    naked_asm!(
        // CPU has pushed: ss, rsp, rflags, cs, rip (low → high in RSP order).
        // Save all 15 GPRs in the layout matching `sched::TrapFrame`:
        //   r15..r8, rbp, rdi, rsi, rdx, rcx, rbx, rax  (high → low pushes).
        "push rax",
        "push rbx",
        "push rcx",
        "push rdx",
        "push rsi",
        "push rdi",
        "push rbp",
        "push r8",
        "push r9",
        "push r10",
        "push r11",
        "push r12",
        "push r13",
        "push r14",
        "push r15",

        // arg0 (rdi) = current rsp
        "mov rdi, rsp",
        "call {pick}",
        // rax = next_rsp (or same rsp if no switch)
        "mov rsp, rax",

        // EOI to PIC for IRQ 0 before iretq, while still in CPL 0.
        // PIC1 OCW2 = 0x20 to port 0x20.
        "mov al, 0x20",
        "out 0x20, al",

        // Restore GPRs in reverse order.
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop r11",
        "pop r10",
        "pop r9",
        "pop r8",
        "pop rbp",
        "pop rdi",
        "pop rsi",
        "pop rdx",
        "pop rcx",
        "pop rbx",
        "pop rax",

        "iretq",
        pick = sym preempt_pick,
    );
}
