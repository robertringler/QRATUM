//! P0.7 — Kernel-owned page tables.
//!
//! Up to P0.6 the kernel rode on the firmware's identity map. This
//! module constructs a kernel-owned PML4 + 1 GiB identity-mapped PDPT
//! and switches CR3 to it. The new tables retain the identity mapping
//! for the bottom 1 GiB so existing pointers stay valid.
//!
//! Once active, [`map_4k`] is the canonical entry point for adding
//! new mappings (e.g. AP trampoline frame as USER+WRITE+EXEC).

#![allow(dead_code)]

use x86_64::registers::control::{Cr3, Cr3Flags};
use x86_64::structures::paging::page_table::{PageTable, PageTableFlags as F};
use x86_64::PhysAddr;

use super::frame_alloc;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PagingError { OutOfFrames, NotInitialized, BadAddr }

static mut PML4_PA: u64 = 0;

/// Build a kernel-owned PML4 with the bottom 1 GiB identity-mapped
/// using a single 1 GiB huge page (kernel + UEFI runtime stays
/// addressable). Switch CR3.
///
/// # Safety
/// Must be called once, with IRQs off and after `frame_alloc::init`.
/// Code, stacks, and any dereferenced pointers must lie in the
/// 0..1 GiB identity range.
pub unsafe fn take_ownership() -> Result<(), PagingError> {
    let pml4_pa = frame_alloc::alloc().map_err(|_| PagingError::OutOfFrames)?;
    let pdpt_pa = frame_alloc::alloc().map_err(|_| PagingError::OutOfFrames)?;
    zero_table(pml4_pa);
    zero_table(pdpt_pa);

    let pml4 = pml4_pa as *mut PageTable;
    let pdpt = pdpt_pa as *mut PageTable;

    // Entry 0 of PML4 → PDPT.
    (*pml4)[0].set_addr(PhysAddr::new(pdpt_pa), F::PRESENT | F::WRITABLE);
    // Entry 0 of PDPT → 1 GiB huge page at 0.
    (*pdpt)[0].set_addr(PhysAddr::new(0), F::PRESENT | F::WRITABLE | F::HUGE_PAGE);

    PML4_PA = pml4_pa;
    let frame = x86_64::structures::paging::PhysFrame::containing_address(PhysAddr::new(pml4_pa));
    Cr3::write(frame, Cr3Flags::empty());
    Ok(())
}

unsafe fn zero_table(pa: u64) {
    let p = pa as *mut u64;
    for i in 0..512 { core::ptr::write_volatile(p.add(i), 0); }
}

/// Map a 4 KiB page (used for the AP trampoline frame).
///
/// # Safety
/// `pa` and `va` must be 4 KiB-aligned; PML4 must be installed.
pub unsafe fn map_4k(va: u64, pa: u64, writable: bool, exec: bool, user: bool)
    -> Result<(), PagingError>
{
    if PML4_PA == 0 { return Err(PagingError::NotInitialized); }
    if (va | pa) & 0xFFF != 0 { return Err(PagingError::BadAddr); }

    let mut flags = F::PRESENT;
    if writable { flags |= F::WRITABLE; }
    if user     { flags |= F::USER_ACCESSIBLE; }
    if !exec    { flags |= F::NO_EXECUTE; }

    let pml4 = PML4_PA as *mut PageTable;
    let pml4_idx = ((va >> 39) & 0x1FF) as usize;
    let pdpt_idx = ((va >> 30) & 0x1FF) as usize;
    let pd_idx   = ((va >> 21) & 0x1FF) as usize;
    let pt_idx   = ((va >> 12) & 0x1FF) as usize;

    let pdpt_pa = ensure_table(&mut (*pml4)[pml4_idx])?;
    let pdpt = pdpt_pa as *mut PageTable;
    let pd_pa = ensure_table(&mut (*pdpt)[pdpt_idx])?;
    let pd = pd_pa as *mut PageTable;
    let pt_pa = ensure_table(&mut (*pd)[pd_idx])?;
    let pt = pt_pa as *mut PageTable;
    (*pt)[pt_idx].set_addr(PhysAddr::new(pa), flags);

    x86_64::instructions::tlb::flush(x86_64::VirtAddr::new(va));
    Ok(())
}

unsafe fn ensure_table(entry: &mut x86_64::structures::paging::page_table::PageTableEntry)
    -> Result<u64, PagingError>
{
    if entry.flags().contains(F::PRESENT) {
        Ok(entry.addr().as_u64())
    } else {
        let pa = frame_alloc::alloc().map_err(|_| PagingError::OutOfFrames)?;
        zero_table(pa);
        entry.set_addr(PhysAddr::new(pa), F::PRESENT | F::WRITABLE | F::USER_ACCESSIBLE);
        Ok(pa)
    }
}

/// Page-fault policy: log the fault, audit it, then halt the AP that
/// took the fault. Called from `interrupts::page_fault`.
pub fn on_page_fault(addr: u64, code: u64, rip: u64) {
    use crate::kernel::audit::{append, FrameKind};
    let mut buf = [0u8; 32];
    buf[0..8].copy_from_slice(&addr.to_le_bytes());
    buf[8..16].copy_from_slice(&code.to_le_bytes());
    buf[16..24].copy_from_slice(&rip.to_le_bytes());
    append(FrameKind::KernelPanic, &buf);
}
