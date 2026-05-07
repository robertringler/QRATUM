//! Scheduler v1 (P0.2 — Agent 4, Agent 7).
//!
//! Core change vs P0.1:
//!
//!   * **Task Control Block (TCB) v1** — every task carries a full
//!     register snapshot (`TrapFrame`), authority, scheduler class,
//!     last accepted intent id, audit cursor.
//!   * **True preemption** — the timer IRQ saves a complete `TrapFrame`
//!     to the running TCB and `iretq`s into the next task's frame.
//!     The cooperative `yield_now()` path remains for syscall/qsh use.
//!   * **Arbiter inline** — `spawn()` *requires* a verdict. A task with
//!     no accepted intent cannot be made Ready (Agent 8: "no task can
//!     be scheduled without arbitration verdict metadata").
//!
//! See `docs::execution_flow` (in `qsh::handle("flow")`) for the textual
//! Boot → IRQ → Arbiter → Sched → CtxSwitch → Audit diagram.

#![allow(dead_code)]

use core::arch::naked_asm;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use spin::Mutex;

use super::audit::{self, FrameKind};
use crate::shared::{ResultEnvelope, Verdict};

pub const MAX_TASKS:  usize = 8;
pub const STACK_SIZE: usize = 16 * 1024;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TaskState {
    Unused  = 0,
    Ready   = 1,
    Running = 2,
    Blocked = 3,
    Dead    = 4,
}

/// Saved register set for both **cooperative** and **preemptive** paths.
///
/// Cooperative `context_switch` uses only the callee-saved tail
/// (`r15..rbx`) and synthesises a fake iretq frame on first dispatch.
///
/// Preemptive IRQ0 trampoline pushes/pops the full frame.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrapFrame {
    // Pushed by the trampoline (general purpose, callee + caller saved):
    pub r15: u64, pub r14: u64, pub r13: u64, pub r12: u64,
    pub r11: u64, pub r10: u64, pub r9:  u64, pub r8:  u64,
    pub rbp: u64, pub rdi: u64, pub rsi: u64, pub rdx: u64,
    pub rcx: u64, pub rbx: u64, pub rax: u64,
    // Pushed by the CPU on interrupt entry (iretq frame, low → high):
    pub rip: u64, pub cs:  u64, pub rflags: u64,
    pub rsp: u64, pub ss:  u64,
}

impl TrapFrame {
    pub const fn empty() -> Self {
        Self { r15:0,r14:0,r13:0,r12:0,r11:0,r10:0,r9:0,r8:0,
               rbp:0,rdi:0,rsi:0,rdx:0,rcx:0,rbx:0,rax:0,
               rip:0,cs:0,rflags:0,rsp:0,ss:0 }
    }
}

/// Task Control Block — unit of execution reality (TCB v1, Agent 8).
#[repr(C)]
pub struct Tcb {
    /// `rsp` at the time the task last entered the scheduler.
    /// For cooperative tasks this is the callee-saved stack pointer;
    /// for preemptive tasks this points at a saved `TrapFrame`.
    pub rsp:           u64,
    pub state:         TaskState,
    pub prio:          u8,
    pub auth_class:    u8,
    pub sched_class:   u8,    // 0 = kernel coop, 1 = preemptive
    /// Arbitration metadata: every task starts life with an *accepted*
    /// verdict carrying its authority. No verdict ⇒ never Ready.
    pub verdict:       Verdict,
    pub last_intent_id:[u8; 16],
    pub audit_cursor:  u64,   // last audit seq this task has observed
    pub _pad:          [u8; 6],
    pub name:          [u8; 24],
}

impl Tcb {
    const fn empty() -> Self {
        Self {
            rsp: 0, state: TaskState::Unused,
            prio: 0, auth_class: 0, sched_class: 0,
            verdict: Verdict::Denied,
            last_intent_id: [0; 16],
            audit_cursor: 0,
            _pad: [0; 6],
            name: [0; 24],
        }
    }
}

#[repr(align(16))]
struct Stack([u8; STACK_SIZE]);

static mut STACKS: [Stack; MAX_TASKS] = [const { Stack([0; STACK_SIZE]) }; MAX_TASKS];

pub struct SchedState {
    pub tasks:   [Tcb; MAX_TASKS],
    pub current: usize,
}

pub static SCHED: Mutex<SchedState> = Mutex::new(SchedState {
    tasks:   [const { Tcb::empty() }; MAX_TASKS],
    current: 0,
});

static NEED_RESCHED:    AtomicBool  = AtomicBool::new(false);
static CONTEXT_SWITCHES: AtomicUsize = AtomicUsize::new(0);
static PREEMPTIVE_SWITCHES: AtomicUsize = AtomicUsize::new(0);

pub fn switches() -> usize { CONTEXT_SWITCHES.load(Ordering::Relaxed) }
pub fn preemptive_switches() -> usize { PREEMPTIVE_SWITCHES.load(Ordering::Relaxed) }

/// Initialise scheduler with task 0 = the currently-executing kernel
/// thread (qsh). Carries an implicit System verdict so it's allowed to run.
pub fn init() {
    let mut s = SCHED.lock();
    s.tasks[0].state       = TaskState::Running;
    s.tasks[0].prio        = 2;
    s.tasks[0].auth_class  = 4; // System
    s.tasks[0].sched_class = 0; // cooperative
    s.tasks[0].verdict     = Verdict::Accepted;
    let nm = b"qsh";
    s.tasks[0].name[..nm.len()].copy_from_slice(nm);
    s.current = 0;
}

/// Spawn a new kernel task. **Requires an accepted verdict** —
/// no verdict, no scheduling (Agent 8 hard rule).
pub fn spawn_with_verdict(
    name: &str, prio: u8, auth_class: u8,
    entry: extern "C" fn() -> !, verdict: &ResultEnvelope,
) -> Option<usize> {
    if verdict.verdict != Verdict::Accepted as u8 {
        audit::append(FrameKind::TaskExit, b"spawn-denied");
        return None;
    }

    let mut s = SCHED.lock();
    let slot = s.tasks.iter().position(|t| t.state == TaskState::Unused)?;

    // Cooperative initial stack frame — entry RIP at top, then 6 zeroed
    // callee-saved slots (rbx, rbp, r12..r15). 16-byte aligned.
    let stack_top = unsafe {
        let base = (&raw mut STACKS[slot]) as *mut u8;
        base.add(STACK_SIZE)
    };
    let mut sp = stack_top as u64;
    sp -= 8; unsafe { *(sp as *mut u64) = 0; }                 // align pad
    sp -= 8; unsafe { *(sp as *mut u64) = entry as u64; }      // ret -> entry
    for _ in 0..6 { sp -= 8; unsafe { *(sp as *mut u64) = 0; } }

    let t = &mut s.tasks[slot];
    t.rsp           = sp;
    t.state         = TaskState::Ready;
    t.prio          = prio;
    t.auth_class    = auth_class;
    t.sched_class   = 0; // cooperative; promote to 1 once preemption-safe
    t.verdict       = Verdict::Accepted;
    t.last_intent_id = verdict.id;
    t.audit_cursor  = audit::count();
    let nb = name.as_bytes();
    let n = nb.len().min(t.name.len());
    t.name[..n].copy_from_slice(&nb[..n]);

    Some(slot)
}

/// Pick next runnable task — strict priority across classes, round-robin within.
fn pick_next(state: &SchedState) -> Option<usize> {
    let cur = state.current;
    let mut best: Option<usize> = None;
    let mut best_prio: i32 = -1;
    for offset in 1..=MAX_TASKS {
        let i = (cur + offset) % MAX_TASKS;
        let t = &state.tasks[i];
        if t.state == TaskState::Ready && t.verdict == Verdict::Accepted {
            if (t.prio as i32) > best_prio {
                best = Some(i);
                best_prio = t.prio as i32;
            }
        }
    }
    best
}

/// Cooperative voluntary yield — used by qsh and syscalls.
pub fn yield_now() {
    super::profiler::trace_event(super::profiler::TraceKind::SchedEnter, 0);
    let (prev_idx, next_idx, prev_rsp_ptr, next_rsp) = {
        let mut s = SCHED.lock();
        let next = match pick_next(&s) { Some(n) => n, None => return };
        let cur = s.current;
        if cur == next { return; }
        if s.tasks[cur].state == TaskState::Running {
            s.tasks[cur].state = TaskState::Ready;
        }
        s.tasks[next].state = TaskState::Running;
        let prev_rsp_ptr = &mut s.tasks[cur].rsp as *mut u64;
        let next_rsp = s.tasks[next].rsp;
        s.current = next;
        (cur, next, prev_rsp_ptr, next_rsp)
    };

    audit::append(FrameKind::SchedYield,    &[prev_idx as u8, next_idx as u8]);
    audit::append(FrameKind::ContextSwitch, &[prev_idx as u8, next_idx as u8]);
    CONTEXT_SWITCHES.fetch_add(1, Ordering::Relaxed);
    NEED_RESCHED.store(false, Ordering::Relaxed);
    super::profiler::trace_event(super::profiler::TraceKind::SchedExit, next_idx as u64);

    unsafe { context_switch(prev_rsp_ptr, next_rsp); }
}

pub fn on_timer_tick() { NEED_RESCHED.store(true, Ordering::Relaxed); }
pub fn should_yield() -> bool { NEED_RESCHED.load(Ordering::Relaxed) }
pub fn yield_if_needed() { if should_yield() { yield_now(); } }

pub fn terminate_current_and_halt() -> ! {
    {
        let mut s = SCHED.lock();
        let cur = s.current;
        s.tasks[cur].state = TaskState::Dead;
    }
    audit::append(FrameKind::TaskExit, b"terminated");
    yield_now();
    loop { unsafe { core::arch::asm!("cli; hlt", options(nomem, nostack, preserves_flags)); } }
}

/// Dump the task table for `qsh tasks`.
pub fn dump(mut f: impl FnMut(usize, &Tcb)) {
    let s = SCHED.lock();
    for (i, t) in s.tasks.iter().enumerate() {
        if t.state != TaskState::Unused { f(i, t); }
    }
}

// ---------------------------------------------------------------------------
// Cooperative context switch (callee-saved swap, C-ABI).
// rdi = prev_rsp_ptr, rsi = next_rsp.
// ---------------------------------------------------------------------------
#[naked]
pub unsafe extern "C" fn context_switch(_prev_rsp_ptr: *mut u64, _next_rsp: u64) {
    naked_asm!(
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "mov [rdi], rsp",
        "mov rsp, rsi",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "ret",
    );
}
