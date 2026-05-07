//! INV-V — work stealing is deterministic and replay-stable.

use qratum_arbiter::deterministic_steal::{pick_steal, schedule};

#[test]
fn pick_is_deterministic() {
    let depths = vec![10u32, 0, 5, 2];
    for _ in 0..1000 {
        assert_eq!(pick_steal(&depths), Some((0, 1)));
    }
}

#[test]
fn schedule_is_replay_stable() {
    let traces: Vec<Vec<u32>> = (0..50)
        .map(|i| vec![i as u32 * 3, i as u32, i as u32 * 7, i as u32 / 2])
        .collect();
    let s1 = schedule(&traces);
    let s2 = schedule(&traces);
    assert_eq!(s1.len(), s2.len());
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert_eq!(a.seq, b.seq);
        assert_eq!(a.src, b.src);
        assert_eq!(a.dst, b.dst);
    }
}

#[test]
fn no_steal_when_balanced() {
    assert_eq!(pick_steal(&[5, 5, 5, 5]), None);
    assert_eq!(pick_steal(&[5, 6]), None);
}

#[test]
fn no_steal_with_single_core() {
    assert_eq!(pick_steal(&[42]), None);
}
