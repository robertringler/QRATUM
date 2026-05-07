//! P0.6 — Migration epoch numbering. Every migration must carry a
//! monotone epoch; replay reorders by `(epoch, seq)` deterministically.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

static EPOCH: AtomicU64 = AtomicU64::new(1);

#[inline]
pub fn current() -> u64 { EPOCH.load(Ordering::Acquire) }

pub fn advance() -> u64 { EPOCH.fetch_add(1, Ordering::AcqRel) + 1 }

pub fn reset() { EPOCH.store(1, Ordering::Release); }
