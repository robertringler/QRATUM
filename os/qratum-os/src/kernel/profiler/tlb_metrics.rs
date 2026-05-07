//! P0.4 — TLB metrics proxy.
//!
//! Real TLB miss counts require PMU. The kernel-portable proxy is
//! "address-space switches" and "page-table walks" attributed by the
//! VMM layer. P0.4 has a single address space, so this module records:
//!
//!   - `as_switches`   — count of CR3 reloads (always 0 in P0.4)
//!   - `pt_walks`      — count of page-fault-driven walks
//!
//! Both are surfaced for the report; the locality proxy is the
//! primary actionable metric in P0.4.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static AS_SWITCHES: AtomicU64 = AtomicU64::new(0);
static PT_WALKS:    AtomicU64 = AtomicU64::new(0);

#[inline] pub fn note_as_switch() { AS_SWITCHES.fetch_add(1, Ordering::Relaxed); }
#[inline] pub fn note_pt_walk()   { PT_WALKS.fetch_add(1, Ordering::Relaxed); }

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TlbSummary {
    pub as_switches: u64,
    pub pt_walks:    u64,
}

pub fn summary() -> TlbSummary {
    TlbSummary {
        as_switches: AS_SWITCHES.load(Ordering::Relaxed),
        pt_walks:    PT_WALKS.load(Ordering::Relaxed),
    }
}

pub fn reset() {
    AS_SWITCHES.store(0, Ordering::Relaxed);
    PT_WALKS.store(0, Ordering::Relaxed);
}
