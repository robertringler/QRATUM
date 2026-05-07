//! P0.5 — xAPIC / x2APIC selection shim. Today selects xAPIC only;
//! x2APIC enabled in P0.6 once MSR programming lands.

#![allow(dead_code)]

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ApicMode { XApic, X2Apic }

#[inline]
pub fn detect() -> ApicMode { ApicMode::XApic }
