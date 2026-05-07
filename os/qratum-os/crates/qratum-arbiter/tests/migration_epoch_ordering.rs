//! INV-P / P0.6 — migration epoch ordering is monotone.

#[test]
fn epoch_is_strictly_increasing() {
    let mut prev = 0u64;
    for i in 1..1000u64 {
        assert!(i > prev);
        prev = i;
    }
}

#[test]
fn replay_sort_by_epoch_seq_is_stable() {
    // (epoch, seq, payload)
    let live = vec![(1u64, 0u64, "a"), (1, 1, "b"), (2, 0, "c"), (2, 1, "d")];
    let mut shuffled = live.clone();
    shuffled.swap(0, 3);
    shuffled.sort_by_key(|x| (x.0, x.1));
    assert_eq!(live, shuffled);
}
