//! P0.5 — Memory-order annotations & violation detector (INV-S).
//!
//! Pure types + helpers; the annotations propagate to runtime via
//! the `expect_ordering` predicate which records a violation if the
//! observed ordering disagrees with the declared one.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering as Ord};

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeclaredOrdering {
    Relaxed = 0,
    Acquire = 1,
    Release = 2,
    AcqRel  = 3,
    SeqCst  = 4,
}

impl DeclaredOrdering {
    #[inline]
    pub fn satisfies(self, observed: Ord) -> bool {
        match (self, observed) {
            (Self::Relaxed, _)                              => true,
            (Self::Acquire, Ord::Acquire | Ord::AcqRel | Ord::SeqCst) => true,
            (Self::Release, Ord::Release | Ord::AcqRel | Ord::SeqCst) => true,
            (Self::AcqRel,  Ord::AcqRel  | Ord::SeqCst)     => true,
            (Self::SeqCst,  Ord::SeqCst)                    => true,
            _ => false,
        }
    }
}

static VIOLATIONS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn expect(declared: DeclaredOrdering, observed: Ord) -> bool {
    if declared.satisfies(observed) { true }
    else { VIOLATIONS.fetch_add(1, Ord::Relaxed); false }
}

#[inline]
pub fn violations() -> u64 { VIOLATIONS.load(Ord::Acquire) }

pub fn reset() { VIOLATIONS.store(0, Ord::Release); }
