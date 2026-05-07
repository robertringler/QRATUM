//! 8259 PIC remap (Agent 4).
//!
//! Remaps master/slave to 0x20/0x28 so they don't overlap with CPU
//! exception vectors (0..31). After init we mask all IRQs except IRQ 0
//! (timer); keyboard (IRQ 1) is unmasked when the kbd driver lands in P0.2.

use spin::Mutex;
use pic8259::ChainedPics;

const PIC1_OFFSET: u8 = 0x20;
const PIC2_OFFSET: u8 = 0x28;

pub static PICS: Mutex<ChainedPics> =
    Mutex::new(unsafe { ChainedPics::new(PIC1_OFFSET, PIC2_OFFSET) });

pub fn init() {
    unsafe {
        let mut p = PICS.lock();
        p.initialize();
        // Mask everything; we'll selectively unmask after timer init.
        p.write_masks(0xFF, 0xFF);
    }
}

/// Unmask master IRQ 0 (timer) only.
pub fn unmask_timer_only() {
    unsafe { PICS.lock().write_masks(0xFE, 0xFF); }
}

/// Send EOI to the appropriate PIC for `vec` (full IDT vector).
///
/// # Safety
/// Must only be called from an IRQ handler that observed `vec`.
pub unsafe fn end_of_interrupt(vec: u8) {
    PICS.lock().notify_end_of_interrupt(vec);
}
