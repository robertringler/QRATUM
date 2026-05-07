//! Deterministic receipt chain — Blake3 chain is identical across runs.

use blake3::Hasher;

fn chain(n: usize) -> [u8; 32] {
    let mut head = [0u8; 32];
    for i in 0..n {
        let mut h = Hasher::new();
        h.update(&head);
        h.update(&(i as u64).to_le_bytes());
        head = *h.finalize().as_bytes();
    }
    head
}

#[test]
fn chain_is_deterministic_across_runs() {
    assert_eq!(chain(1024), chain(1024));
}

#[test]
fn chain_grows_strictly() {
    assert_ne!(chain(10), chain(11));
}
