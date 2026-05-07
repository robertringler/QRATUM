//! P0.3.2 — Proof gate. The single source of truth for every
//! published performance / correctness claim.
//!
//! Any string in the QRATUM repo containing the regex
//! `(10x|10×|reduction|speedup|optimization|improvement)` MUST resolve
//! through `claim_registry::CLAIMS` to:
//!
//!   * a metric source (which trace point produces the number),
//!   * a computation function (named in the host crate), and
//!   * a CI gate (test name).
//!
//! The companion validator (`tests/claim_registry_validation_tests.rs`
//! in the host crate) walks the README + spec markdown and fails the
//! build on any unregistered claim.

#![allow(dead_code)]

pub mod claim_registry;
pub mod claim_validator;

pub use claim_registry::{Claim, CLAIMS};
pub use claim_validator::{validate_claims, ValidationReport};
