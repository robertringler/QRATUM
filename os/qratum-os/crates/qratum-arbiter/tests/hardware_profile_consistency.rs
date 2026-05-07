//! P0.5 — Hardware profile must remain deterministic when serialized.

use qratum_arbiter::hardware_profile::HardwareProfile;

#[test]
fn profile_round_trips() {
    let p = HardwareProfile::default();
    let s = serde_json::to_string(&p).unwrap();
    let p2: HardwareProfile = serde_json::from_str(&s).unwrap();
    assert_eq!(p, p2);
}
