//! Aethernet - Accountable, Reversible Overlay Network Substrate
//!
//! Core library for QRATUM sovereign intelligence platform.
//!
//! # Features
//!
//! - **TXO (Transaction Object)**: CBOR-primary encoding with dual-control signatures
//! - **RTF (Reversible Transaction Framework)**: Zone-aware execution with rollback
//! - **Biokey**: Ephemeral key derivation from SNP loci with ZK proofs
//! - **Merkle Ledger**: Append-only, snapshot-based rollback capability
//! - **Compliance**: HIPAA and GDPR modules
//!
//! # Example
//!
//! ```rust,ignore
//! use aethernet::txo::TXO;
//! use aethernet::rtf::api::{RTFContext, Zone};
//! use aethernet::ledger::MerkleLedger;
//!
//! // Create ledger
//! let ledger = MerkleLedger::new([0u8; 32]);
//! let mut ctx = RTFContext::new(Zone::Z1, ledger);
//!
//! // Create and execute TXO
//! let mut txo = TXO::new(/* ... */);
//! ctx.execute_txo(&mut txo)?;
//! ctx.commit_txo(&mut txo)?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

extern crate alloc;

/// Transaction Object (TXO) module
pub mod txo;

/// Reversible Transaction Framework (RTF) module
pub mod rtf;

/// Biokey derivation and ZKP verification module
pub mod biokey;

/// Merkle ledger module
pub mod ledger;

/// HIPAA compliance module
#[cfg(feature = "std")]
pub mod hipaa;

/// GDPR compliance module
#[cfg(feature = "std")]
pub mod gdpr;

// Re-export commonly used types
pub use txo::{TXO, IdentityType, OperationClass, PayloadType, SignatureType};
pub use rtf::api::{RTFContext, Zone, RTFError};
pub use ledger::MerkleLedger;
pub use biokey::derivation::EphemeralBiokey;

/// Aethernet version
pub const VERSION: &str = "1.0.0";

/// Aethernet build information
/// Note: RUSTC_VERSION removed due to build environment constraints
/// To track rust version at runtime, use rustc --version in deployment
pub const BUILD_INFO: &str = concat!(
    "Aethernet v",
    env!("CARGO_PKG_VERSION"),
);
