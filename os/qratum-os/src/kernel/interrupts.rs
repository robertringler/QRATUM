//! CPU exception + IRQ handlers (Agent 2, Agent 5, Agent 10).
//!
//! Discipline: every handler MUST
//!   1. journal an audit frame BEFORE doing anything that can fault again,
//!   2. send EOI for hardware IRQs even on exception paths,
//!   3. never panic on the arbitration path (Agent 10 stability).
//!
//! Page-fault and GPF handlers run on dedicated IST stacks (see `gdt.rs`).

use core::sync::atomic::{AtomicU64, Ordering};

use x86_64::registers::control::Cr2;
use x86_64::structures::idt::{InterruptStackFrame, PageFaultErrorCode};

use super::audit::{self, FrameKind};
use super::sched;
use super::syscall;

/// Tick counter, advanced by IRQ 0 (PIT).
static TIMER_TICKS: AtomicU64 = AtomicU64::new(0);

pub fn timer_ticks() -> u64 { TIMER_TICKS.load(Ordering::Relaxed) }

// ---------------------------------------------------------------------------
// CPU exceptions
// ---------------------------------------------------------------------------

pub extern "x86-interrupt" fn divide_error(stack: InterruptStackFrame) {
    audit_exc(0x00, &stack, 0);
    halt_with("divide_error");
}

pub extern "x86-interrupt" fn debug(stack: InterruptStackFrame) {
    audit_exc(0x01, &stack, 0);
}

pub extern "x86-interrupt" fn nmi(stack: InterruptStackFrame) {
    audit_exc(0x02, &stack, 0);
}

pub extern "x86-interrupt" fn breakpoint(stack: InterruptStackFrame) {
    audit_exc(0x03, &stack, 0);
}

pub extern "x86-interrupt" fn overflow(stack: InterruptStackFrame) {
    audit_exc(0x04, &stack, 0);
}

pub extern "x86-interrupt" fn bound_range(stack: InterruptStackFrame) {
    audit_exc(0x05, &stack, 0);
}

pub extern "x86-interrupt" fn invalid_opcode(stack: InterruptStackFrame) {
    audit_exc(0x06, &stack, 0);
    halt_with("invalid_opcode");
}

pub extern "x86-interrupt" fn device_na(stack: InterruptStackFrame) {
    audit_exc(0x07, &stack, 0);
}

pub extern "x86-interrupt" fn double_fault(stack: InterruptStackFrame, code: u64) -> ! {
    audit_exc(0x08, &stack, code);
    audit::append(FrameKind::DoubleFault, b"#DF");
    halt_with("double_fault");
}

pub extern "x86-interrupt" fn invalid_tss(stack: InterruptStackFrame, code: u64) {
    audit_exc(0x0A, &stack, code);
    halt_with("invalid_tss");
}

pub extern "x86-interrupt" fn segment_np(stack: InterruptStackFrame, code: u64) {
    audit_exc(0x0B, &stack, code);
    halt_with("segment_not_present");
}

pub extern "x86-interrupt" fn stack_seg(stack: InterruptStackFrame, code: u64) {
    audit_exc(0x0C, &stack, code);
    halt_with("stack_segment_fault");
}

pub extern "x86-interrupt" fn gpf(stack: InterruptStackFrame, code: u64) {
    let mut payload = [0u8; 62];
    payload[0..8].copy_from_slice(&stack.instruction_pointer.as_u64().to_le_bytes());
    payload[8..16].copy_from_slice(&code.to_le_bytes());
    audit::append(FrameKind::GeneralProtFault, &payload);

    // If we faulted in user code, kill the running task and yield. P0.1
    // single-task baseline: halt for safety. No re-entry into arbiter.
    let cs = stack.code_segment.0;
    if cs & 0b11 == 3 {
        // CPL was 3 → user task — log and terminate it.
        audit::append(FrameKind::TaskExit, b"gpf-cpl3");
        sched::terminate_current_and_halt();
    } else {
        halt_with("gpf-cpl0");
    }
}

pub extern "x86-interrupt" fn page_fault(stack: InterruptStackFrame, code: PageFaultErrorCode) {
    let cr2 = Cr2::read_raw();
    let mut payload = [0u8; 62];
    payload[0..8].copy_from_slice(&cr2.to_le_bytes());
    payload[8..16].copy_from_slice(&stack.instruction_pointer.as_u64().to_le_bytes());
    payload[16..18].copy_from_slice(&(code.bits() as u16).to_le_bytes());
    audit::append(FrameKind::PageFault, &payload);

    // Hard-failure matrix (Agent 10 — non-negotiable):
    //   page fault (kernel space) → panic
    //   page fault (user space)   → task kill
    let cs = stack.code_segment.0;
    if cs & 0b11 == 3 {
        audit::append(FrameKind::TaskExit, b"pf-cpl3");
        sched::terminate_current_and_halt();
    } else {
        // Kernel-mode page fault is unrecoverable.
        panic!("#PF in kernel: cr2=0x{:016X} rip=0x{:016X} err=0x{:X}",
               cr2, stack.instruction_pointer.as_u64(), code.bits());
    }
}

pub extern "x86-interrupt" fn x87_fp(stack: InterruptStackFrame) {
    audit_exc(0x10, &stack, 0);
}

pub extern "x86-interrupt" fn alignment(stack: InterruptStackFrame, code: u64) {
    audit_exc(0x11, &stack, code);
}

pub extern "x86-interrupt" fn machine_check(stack: InterruptStackFrame) -> ! {
    audit_exc(0x12, &stack, 0);
    halt_with("machine_check");
}

pub extern "x86-interrupt" fn simd_fp(stack: InterruptStackFrame) {
    audit_exc(0x13, &stack, 0);
}

pub extern "x86-interrupt" fn virtualization(stack: InterruptStackFrame) {
    audit_exc(0x14, &stack, 0);
}

// ---------------------------------------------------------------------------
// Hardware IRQs
// ---------------------------------------------------------------------------

pub extern "x86-interrupt" fn timer(_stack: InterruptStackFrame) {
    TIMER_TICKS.fetch_add(1, Ordering::Relaxed);
    audit::append(FrameKind::IrqEnter, b"timer");
    // Preempt: hand control to the scheduler.
    sched::on_timer_tick();
    unsafe { super::apic::eoi(super::idt::IRQ_TIMER); }
}

pub extern "x86-interrupt" fn keyboard(_stack: InterruptStackFrame) {
    // P0.1 stub — drain the data port to clear the IRQ. Handler lands in P0.2.
    let mut p = x86_64::instructions::port::Port::<u8>::new(0x60);
    let _ = unsafe { p.read() };
    audit::append(FrameKind::IrqEnter, b"keyboard");
    unsafe { super::apic::eoi(super::idt::IRQ_KEYBOARD); }
}

// ---------------------------------------------------------------------------
// Syscall gate (int 0x80)
// ---------------------------------------------------------------------------

pub extern "x86-interrupt" fn syscall_int(_stack: InterruptStackFrame) {
    audit::append(FrameKind::SyscallEnter, b"int 0x80");
    syscall::dispatch();
    audit::append(FrameKind::SyscallExit, b"int 0x80");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn audit_exc(vec: u8, stack: &InterruptStackFrame, code: u64) {
    let mut p = [0u8; 62];
    p[0] = vec;
    p[1..9].copy_from_slice(&stack.instruction_pointer.as_u64().to_le_bytes());
    p[9..17].copy_from_slice(&code.to_le_bytes());
    audit::append(FrameKind::IrqEnter, &p);
}

fn halt_with(_tag: &str) -> ! {
    unsafe { core::arch::asm!("cli", options(nomem, nostack, preserves_flags)); }
    loop {
        unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)); }
    }
}
