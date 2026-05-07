//! P0.6 — APIC latency profile consistency across reruns.

use qratum_arbiter::hardware_replay::{HardwareEvent, ReplaySession};

#[test]
fn ipi_only_stream_is_deterministic() {
    let mut s = ReplaySession::new();
    for i in 0u64..200 {
        s.push(HardwareEvent { seq: i, cpu: (i % 4) as u8, kind: 1,
                               vector: 0xF1, payload: i * 100 });
    }
    let h1 = s.stream_hash();
    let h2 = s.replay().stream_hash();
    assert_eq!(h1, h2);
}
