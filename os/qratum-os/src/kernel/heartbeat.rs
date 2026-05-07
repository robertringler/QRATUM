//! Heartbeat task — second runnable scheduled by the kernel.
//!
//! This validates acceptance criterion #3 ("Scheduler can switch between
//! at least 2 mock processes"). The task increments a counter and yields;
//! `qsh status` shows the heartbeat count growing while qsh is interactive.

use core::sync::atomic::{AtomicU64, Ordering};

use super::sched;

static BEATS: AtomicU64 = AtomicU64::new(0);

pub fn beats() -> u64 { BEATS.load(Ordering::Relaxed) }

pub extern "C" fn entry() -> ! {
    loop {
        BEATS.fetch_add(1, Ordering::Relaxed);
        // Yield every iteration — gives qsh CPU time and exercises the
        // scheduler. With a 100 Hz timer and tick-driven flag, switches
        // happen on the next yield after each tick.
        sched::yield_now();
    }
}
