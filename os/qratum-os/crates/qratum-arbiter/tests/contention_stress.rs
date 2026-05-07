//! P0.5 — Contention stress. Repeatedly enqueue and migrate; verify
//! INV-K is never violated and composed hash stays deterministic.

use qratum_arbiter::smp_model::SmpModel;

#[test]
fn high_contention_no_dual_occupancy() {
    let mut m = SmpModel::new(8);
    for tid in 0u32..64 { m.create(tid, tid as u64); }
    let mut x = 0xDEAD_BEEFu64;
    for _ in 0..2048 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let tid = (x % 64) as u32;
        let src = ((x >> 32) & 7) as u8;
        let dst = ((x >> 40) & 7) as u8;
        let _ = m.migrate(tid, src, dst);
        assert!(m.no_dual_occupancy(), "INV-K must hold under stress");
    }
}

#[test]
fn high_contention_hash_deterministic() {
    fn run() -> u64 {
        let mut m = SmpModel::new(8);
        for tid in 0u32..64 { m.create(tid, tid as u64); }
        let mut x = 0xBADC_0FFEu64;
        for _ in 0..1024 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let tid = (x % 64) as u32;
            let src = ((x >> 32) & 7) as u8;
            let dst = ((x >> 40) & 7) as u8;
            let _ = m.migrate(tid, src, dst);
        }
        m.composed_state_hash()
    }
    assert_eq!(run(), run());
}
