//! P0.7 — Bitmap physical frame allocator.
//!
//! Constructed once from the UEFI memory map staged by stage1 into a
//! kernel-owned region. Tracks 4 KiB frames as bits in a static
//! bitmap. This is intentionally simple — P0.8 will replace it with a
//! buddy allocator. The job here is "allocate the AP trampoline frame
//! and the kernel's PML4 / PDPT pages", nothing more.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use spin::Mutex;

const MAX_FRAMES: usize = 1 << 18; // 1 GiB at 4 KiB granularity
const BITS_PER_WORD: usize = 64;
const N_WORDS: usize = MAX_FRAMES / BITS_PER_WORD;

static BITMAP: Mutex<[u64; N_WORDS]> = Mutex::new([0; N_WORDS]);
static BASE_PA: AtomicU64 = AtomicU64::new(0);
static N_TRACKED: AtomicUsize = AtomicUsize::new(0);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AllocError { Uninitialized, Exhausted }

/// Initialize with a physical region `[base, base + len_bytes)` known
/// to be free of UEFI runtime data and identity-mapped. `base` must
/// be 4 KiB-aligned. Frames are marked **free** initially.
pub fn init_range(base_pa: u64, len_bytes: u64) -> Result<(), AllocError> {
    if base_pa & 0xFFF != 0 { return Err(AllocError::Exhausted); }
    let frames = (len_bytes / 4096) as usize;
    let frames = frames.min(MAX_FRAMES);
    BASE_PA.store(base_pa, Ordering::Release);
    N_TRACKED.store(frames, Ordering::Release);
    let mut bm = BITMAP.lock();
    for w in bm.iter_mut() { *w = 0; } // 0 = free
    Ok(())
}

/// Bring-up entry — looks up a default region, falls back to a 16 MiB
/// stub region just above the kernel image (caller verifies before
/// real hardware use).
pub fn init() -> Result<(), AllocError> {
    if BASE_PA.load(Ordering::Acquire) == 0 {
        // Default to 0x4000_0000..0x4100_0000 (16 MiB) — adjust per-bringup.
        init_range(0x4000_0000, 16 * 1024 * 1024)?;
    }
    Ok(())
}

/// Allocate one 4 KiB frame.
pub fn alloc() -> Result<u64, AllocError> {
    let n = N_TRACKED.load(Ordering::Acquire);
    if n == 0 { return Err(AllocError::Uninitialized); }
    let mut bm = BITMAP.lock();
    for (wi, word) in bm.iter_mut().enumerate() {
        if *word != u64::MAX {
            let b = (!*word).trailing_zeros() as usize;
            let idx = wi * BITS_PER_WORD + b;
            if idx >= n { return Err(AllocError::Exhausted); }
            *word |= 1 << b;
            return Ok(BASE_PA.load(Ordering::Acquire) + (idx as u64) * 4096);
        }
    }
    Err(AllocError::Exhausted)
}

/// Allocate a frame with `pa < limit_pa`. Required for the AP
/// trampoline (must be < 1 MiB).
pub fn alloc_below(limit_pa: u64) -> Result<u64, AllocError> {
    let base = BASE_PA.load(Ordering::Acquire);
    let n = N_TRACKED.load(Ordering::Acquire);
    if n == 0 { return Err(AllocError::Uninitialized); }
    let mut bm = BITMAP.lock();
    for idx in 0..n {
        let pa = base + (idx as u64) * 4096;
        if pa >= limit_pa { break; }
        let wi = idx / BITS_PER_WORD;
        let bi = idx % BITS_PER_WORD;
        if bm[wi] & (1 << bi) == 0 {
            bm[wi] |= 1 << bi;
            return Ok(pa);
        }
    }
    Err(AllocError::Exhausted)
}

/// Release a frame.
pub fn free(pa: u64) {
    let base = BASE_PA.load(Ordering::Acquire);
    if pa < base { return; }
    let idx = ((pa - base) / 4096) as usize;
    let n = N_TRACKED.load(Ordering::Acquire);
    if idx >= n { return; }
    let wi = idx / BITS_PER_WORD;
    let bi = idx % BITS_PER_WORD;
    let mut bm = BITMAP.lock();
    bm[wi] &= !(1 << bi);
}
