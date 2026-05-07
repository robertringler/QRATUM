//! Paging utilities (Agent 5).
//!
//! P0.1 keeps the firmware's identity-mapped page tables (CR3 untouched
//! after ExitBootServices). What this module provides:
//!
//!   * `walk_pte(addr)`   — locate the leaf PTE (4 KiB or 2 MiB) for a VA.
//!   * `make_user(addr)`  — set USER bit on the leaf for a page so a CPL3
//!                          task can access it.
//!   * `is_user(addr)`    — diagnostic for tests.
//!
//! No new tables are allocated in P0.1; full kernel-owned higher-half
//! mapping is the work of P0.2.

#![allow(dead_code)]

use x86_64::registers::control::Cr3;
use x86_64::structures::paging::page_table::PageTableFlags as F;
use x86_64::structures::paging::PageTable;
use x86_64::VirtAddr;

unsafe fn table_at(phys: u64) -> &'static mut PageTable {
    // Identity map: phys == virt (UEFI tables stay valid because we did not
    // free them in stage1 and have not touched the firmware's runtime pool).
    &mut *(phys as *mut PageTable)
}

/// Returns a pointer to the leaf PTE that maps `va`, or `None` if unmapped.
///
/// Stops at the first level whose entry has `HUGE_PAGE` set (2 MiB or 1 GiB
/// page) — caller must check flags to know the page size.
///
/// # Safety
/// Caller asserts CR3 still points at firmware/kernel tables and the
/// identity map is intact for those table frames. Always true post-handoff
/// in P0.1.
pub unsafe fn walk_pte(va: VirtAddr) -> Option<*mut x86_64::structures::paging::page_table::PageTableEntry> {
    let (frame, _) = Cr3::read();
    let pml4 = table_at(frame.start_address().as_u64());

    let i4 = (va.as_u64() >> 39) & 0x1FF;
    let i3 = (va.as_u64() >> 30) & 0x1FF;
    let i2 = (va.as_u64() >> 21) & 0x1FF;
    let i1 = (va.as_u64() >> 12) & 0x1FF;

    let e4 = &mut pml4[i4 as usize];
    if !e4.flags().contains(F::PRESENT) { return None; }
    let pdpt = table_at(e4.addr().as_u64());

    let e3 = &mut pdpt[i3 as usize];
    if !e3.flags().contains(F::PRESENT) { return None; }
    if e3.flags().contains(F::HUGE_PAGE) { return Some(e3 as *mut _); }
    let pd = table_at(e3.addr().as_u64());

    let e2 = &mut pd[i2 as usize];
    if !e2.flags().contains(F::PRESENT) { return None; }
    if e2.flags().contains(F::HUGE_PAGE) { return Some(e2 as *mut _); }
    let pt = table_at(e2.addr().as_u64());

    let e1 = &mut pt[i1 as usize];
    if !e1.flags().contains(F::PRESENT) { return None; }
    Some(e1 as *mut _)
}

/// Mark the leaf entry mapping `va` as user-accessible (USER=1).
/// Returns true if the entry was found and updated.
pub fn make_user(va: VirtAddr) -> bool {
    unsafe {
        if let Some(pte) = walk_pte(va) {
            let mut flags = (*pte).flags();
            flags.insert(F::USER_ACCESSIBLE);
            let addr = (*pte).addr();
            (*pte).set_addr(addr, flags);
            // Invalidate TLB for this VA.
            x86_64::instructions::tlb::flush(va);
            true
        } else {
            false
        }
    }
}

pub fn is_user(va: VirtAddr) -> bool {
    unsafe {
        match walk_pte(va) {
            Some(p) => (*p).flags().contains(F::USER_ACCESSIBLE),
            None => false,
        }
    }
}

/// Read CR2 (faulting linear address). Useful from #PF handlers, but we
/// already use `Cr2::read_raw` in `interrupts.rs`. Kept for symmetry.
pub fn faulting_address() -> u64 {
    x86_64::registers::control::Cr2::read_raw()
}
