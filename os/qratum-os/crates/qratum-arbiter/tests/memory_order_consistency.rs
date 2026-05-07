//! P0.5 — INV-S. Memory operation ordering must be at least as strong
//! as declared.

use std::sync::atomic::Ordering;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Declared { Relaxed, Acquire, Release, AcqRel, SeqCst }

fn satisfies(d: Declared, o: Ordering) -> bool {
    use Ordering::*;
    match (d, o) {
        (Declared::Relaxed, _) => true,
        (Declared::Acquire, Acquire | AcqRel | SeqCst) => true,
        (Declared::Release, Release | AcqRel | SeqCst) => true,
        (Declared::AcqRel,  AcqRel  | SeqCst)          => true,
        (Declared::SeqCst,  SeqCst)                    => true,
        _ => false,
    }
}

#[test]
fn release_satisfies_seqcst() {
    assert!(satisfies(Declared::Release, Ordering::SeqCst));
}
#[test]
fn relaxed_does_not_satisfy_acquire_demand() {
    assert!(!satisfies(Declared::Acquire, Ordering::Relaxed));
}
#[test]
fn seqcst_only_satisfies_seqcst() {
    assert!(satisfies(Declared::SeqCst, Ordering::SeqCst));
    assert!(!satisfies(Declared::SeqCst, Ordering::AcqRel));
}
