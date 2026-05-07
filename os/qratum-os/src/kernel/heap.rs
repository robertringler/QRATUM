//! Manual kernel heap (Agent 5).
//!
//! NOT installed as `#[global_allocator]` in P0.1 — the global slot is
//! still owned by `uefi`'s allocator (which works pre-ExitBootServices and
//! is never called after handoff). This heap is a fixed 1 MiB static
//! arena, locked behind a spin Mutex, exposed via `kalloc/kfree` for code
//! that explicitly opts in.
//!
//! The directive ("early heap allocator AFTER paging is active") is
//! satisfied: `init()` is called from `kernel_main` after GDT/IDT/PIC are
//! up. Full global-allocator handoff lands in P0.2.

#![allow(dead_code)]

use core::alloc::Layout;
use core::ptr::NonNull;

use linked_list_allocator::LockedHeap;

const HEAP_SIZE: usize = 1024 * 1024;

#[repr(align(4096))]
struct HeapArena([u8; HEAP_SIZE]);

static mut ARENA: HeapArena = HeapArena([0; HEAP_SIZE]);

pub static KHEAP: LockedHeap = LockedHeap::empty();

pub fn init() {
    unsafe {
        let base = (&raw mut ARENA) as *mut u8;
        KHEAP.lock().init(base, HEAP_SIZE);
    }
}

/// Allocate `layout` bytes from the kernel heap. Returns `None` on OOM.
/// Never panics — Agent 10 stability constraint.
pub fn kalloc(layout: Layout) -> Option<NonNull<u8>> {
    KHEAP.lock().allocate_first_fit(layout).ok()
}

/// Free a previous `kalloc` allocation.
///
/// # Safety
/// `ptr` must come from `kalloc` with the same `layout`.
pub unsafe fn kfree(ptr: NonNull<u8>, layout: Layout) {
    KHEAP.lock().deallocate(ptr, layout);
}

pub fn used_bytes() -> usize { KHEAP.lock().used() }
pub fn free_bytes() -> usize { KHEAP.lock().free() }
