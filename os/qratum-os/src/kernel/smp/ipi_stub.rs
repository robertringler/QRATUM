//! Inter-processor interrupt stub (P0.3 single-core).
//!
//! `Ipi::send(target, msg)` enqueues into the target core's inbox.
//! `drain_inbox()` returns pending messages. P0.3 backs both with a
//! single-core SPSC ring; P0.3.1 swaps the send path for an xAPIC ICR
//! write.

use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;

const INBOX_CAP: usize = 32;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Ipi {
    Halt,
    SchedRebalance(u32),       // task id
    AuditFlush,
    InvariantViolation(u32),   // invariant code
}

struct Inbox {
    buf: [Option<Ipi>; INBOX_CAP],
    pub(crate) head: usize,
    pub(crate) tail: usize,
}

impl Inbox {
    const fn new() -> Self { Self { buf: [None; INBOX_CAP], head: 0, tail: 0 } }
    fn push(&mut self, m: Ipi) -> bool {
        let next = (self.tail + 1) % INBOX_CAP;
        if next == self.head { return false; }
        self.buf[self.tail] = Some(m);
        self.tail = next;
        true
    }
    fn pop(&mut self) -> Option<Ipi> {
        if self.head == self.tail { return None; }
        let m = self.buf[self.head].take();
        self.head = (self.head + 1) % INBOX_CAP;
        m
    }
}

static INBOXES: [Mutex<Inbox>; super::MAX_CPUS] = [
    Mutex::new(Inbox::new()), Mutex::new(Inbox::new()),
    Mutex::new(Inbox::new()), Mutex::new(Inbox::new()),
    Mutex::new(Inbox::new()), Mutex::new(Inbox::new()),
    Mutex::new(Inbox::new()), Mutex::new(Inbox::new()),
];

static SEND_COUNT: AtomicUsize = AtomicUsize::new(0);
static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn init() { /* nothing in single-core */ }

/// Send an IPI to `target`. Returns false if the inbox was full
/// (caller decides whether to retry; this is **not** silent failure —
/// the kernel will record an invariant violation).
pub fn send(target: u8, m: Ipi) -> bool {
    SEND_COUNT.fetch_add(1, Ordering::Relaxed);
    let idx = (target as usize) % super::MAX_CPUS;
    let ok = INBOXES[idx].lock().push(m);
    if !ok { DROP_COUNT.fetch_add(1, Ordering::Relaxed); }
    ok
}

/// Drain all pending IPIs for the current CPU. Returns count drained.
pub fn drain_inbox<F: FnMut(Ipi)>(mut handler: F) -> usize {
    let cpu = super::current_cpu();
    let mut inbox = INBOXES[cpu as usize].lock();
    let mut n = 0;
    while let Some(m) = inbox.pop() { handler(m); n += 1; }
    n
}

pub fn dropped() -> usize { DROP_COUNT.load(Ordering::Relaxed) }
pub fn sent() -> usize { SEND_COUNT.load(Ordering::Relaxed) }

/// P0.4 — drain every CPU's inbox in fixed order (0..MAX_CPUS).
/// Used by `barriers::quiesce_for_replay`. Drops messages.
pub fn drain_all() -> usize {
    let mut n = 0;
    for cpu in 0..super::MAX_CPUS {
        let mut inbox = INBOXES[cpu].lock();
        while inbox.pop().is_some() { n += 1; }
    }
    n
}

/// Current depth of a per-CPU inbox. Snapshot only — value can change
/// before the caller observes it; used by `runtime_model` for the
/// lattice projection.
pub fn inbox_len(cpu: u8) -> u32 {
    if (cpu as usize) >= super::MAX_CPUS { return 0; }
    let inbox = INBOXES[cpu as usize].lock();
    let head = inbox.head;
    let tail = inbox.tail;
    if tail >= head {
        (tail - head) as u32
    } else {
        (INBOX_CAP - head + tail) as u32
    }
}
