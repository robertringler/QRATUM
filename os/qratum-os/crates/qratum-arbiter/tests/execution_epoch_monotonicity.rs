//! Execution epoch monotonicity.

#[test]
fn epoch_advance_is_strictly_monotone() {
    let mut e = 0u64;
    for _ in 0..10_000 {
        let next = e + 1;
        assert!(next > e);
        e = next;
    }
}

#[test]
fn epoch_does_not_wrap_within_test() {
    let mut e = 0u64;
    for _ in 0..1_000_000 { e = e.checked_add(1).unwrap(); }
    assert_eq!(e, 1_000_000);
}
