//! INV-V / replay equivalence — capture, replay, hashes match.

use qratum_arbiter::hardware_replay::{HardwareEvent, ReplaySession, replay_equivalent};

fn fixture() -> ReplaySession {
    let mut s = ReplaySession::new();
    for i in 0..256u64 {
        s.push(HardwareEvent {
            seq: i, cpu: (i % 8) as u8, kind: (i % 5) as u8,
            vector: (32 + (i % 16)) as u8, payload: i.wrapping_mul(0x9E37_79B9),
        });
    }
    s
}

#[test]
fn replay_is_bit_identical() {
    let live = fixture();
    let replayed = live.replay();
    assert_eq!(live.stream_hash(), replayed.stream_hash());
    assert!(replay_equivalent(&live, &replayed));
}

#[test]
fn distinct_sessions_distinct_hashes() {
    let mut a = fixture();
    let b = fixture();
    a.push(HardwareEvent { seq: 999, cpu: 0, kind: 0, vector: 0, payload: 0 });
    assert_ne!(a.stream_hash(), b.stream_hash());
}

#[test]
fn determinism_across_runs() {
    assert_eq!(fixture().stream_hash(), fixture().stream_hash());
}
