//! Interrupt Descriptor Table (Agent 2, Agent 10).
//!
//! 32 CPU-defined exceptions hooked. IRQs 32..47 routed to PIC handlers
//! (timer = 32, keyboard = 33). Software interrupt 0x80 is the syscall
//! gate (DPL = 3 so user code can invoke it).

use spin::Lazy;
use x86_64::structures::idt::InterruptDescriptorTable;
use x86_64::PrivilegeLevel;

use super::interrupts;
use super::gdt;

/// Hardware IRQ base (after PIC remap).
pub const PIC_OFFSET: u8 = 32;
pub const IRQ_TIMER:    u8 = PIC_OFFSET + 0;
pub const IRQ_KEYBOARD: u8 = PIC_OFFSET + 1;

/// Software syscall vector.
pub const SYSCALL_VECTOR: u8 = 0x80;

static IDT: Lazy<InterruptDescriptorTable> = Lazy::new(|| {
    let mut idt = InterruptDescriptorTable::new();

    // ---- CPU exceptions --------------------------------------------------
    idt.divide_error             .set_handler_fn(interrupts::divide_error);
    idt.debug                    .set_handler_fn(interrupts::debug);
    idt.non_maskable_interrupt   .set_handler_fn(interrupts::nmi);
    idt.breakpoint               .set_handler_fn(interrupts::breakpoint);
    idt.overflow                 .set_handler_fn(interrupts::overflow);
    idt.bound_range_exceeded     .set_handler_fn(interrupts::bound_range);
    idt.invalid_opcode           .set_handler_fn(interrupts::invalid_opcode);
    idt.device_not_available     .set_handler_fn(interrupts::device_na);

    unsafe {
        idt.double_fault
            .set_handler_fn(interrupts::double_fault)
            .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
        idt.page_fault
            .set_handler_fn(interrupts::page_fault)
            .set_stack_index(gdt::PAGE_FAULT_IST_INDEX);
    }

    idt.invalid_tss              .set_handler_fn(interrupts::invalid_tss);
    idt.segment_not_present      .set_handler_fn(interrupts::segment_np);
    idt.stack_segment_fault      .set_handler_fn(interrupts::stack_seg);
    idt.general_protection_fault .set_handler_fn(interrupts::gpf);
    idt.x87_floating_point       .set_handler_fn(interrupts::x87_fp);
    idt.alignment_check          .set_handler_fn(interrupts::alignment);
    idt.machine_check            .set_handler_fn(interrupts::machine_check);
    idt.simd_floating_point      .set_handler_fn(interrupts::simd_fp);
    idt.virtualization           .set_handler_fn(interrupts::virtualization);

    // ---- IRQs ------------------------------------------------------------
    idt[IRQ_TIMER]   .set_handler_fn(interrupts::timer);
    idt[IRQ_KEYBOARD].set_handler_fn(interrupts::keyboard);

    // ---- Syscall (int 0x80, DPL = 3) ------------------------------------
    let opts = idt[SYSCALL_VECTOR].set_handler_fn(interrupts::syscall_int);
    opts.set_privilege_level(PrivilegeLevel::Ring3);

    idt
});

pub fn init() {
    IDT.load();
}
