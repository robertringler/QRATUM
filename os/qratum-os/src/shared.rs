//! P0.0 shared types — kernel-canonical envelopes (Agent 8).
//!
//! Mirrors the binary contracts in QC_CORE_OS_BOOT_KERNEL_V1.md §2.3 and §4.2.
//! Kept intentionally minimal: structures here are stable ABI surface that
//! P0.1+ phases will add to (never remove from).

#![allow(dead_code)]

/// `BootInfo.magic` — ASCII "QRATUMBO" little-endian.
pub const QRATUM_BOOT_MAGIC: u64 = 0x4F42_4D55_5441_5251;

/// `IntentEnvelope.magic` — ASCII "INT1".
pub const INTENT_MAGIC: u32 = 0x3154_4E49;
pub const INTENT_VERSION: u16 = 1;

/// Hand-off contract from bootloader stage → kernel stage.
///
/// P0.0 keeps the structure tiny but layout-stable so that P0.1 can add
/// memory-map and framebuffer pointers without breaking the ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BootInfo {
    pub magic: u64,
    pub rng_seed: [u8; 32],
    /// Total usable RAM in bytes (from UEFI memory map).
    pub usable_ram_bytes: u64,
    /// Number of CONVENTIONAL_MEMORY descriptors at hand-off.
    pub usable_regions: u32,
    /// Largest contiguous CONVENTIONAL_MEMORY region (bytes).
    pub largest_region_bytes: u64,
    /// Boot wall-clock timestamp (seconds since epoch, 0 if unknown).
    pub boot_unix_ts: u64,
    /// Kernel command line, NUL-terminated.
    pub command_line: [u8; 256],
    /// P0.7 — ACPI Root System Description Pointer physical address.
    /// `0` means UEFI ConfigurationTable did not advertise an ACPI table
    /// (firmware is too old, or the platform lacks ACPI). `hw::init` skips
    /// the entire APIC bring-up chain and falls back to BSP-only PIC mode
    /// when this is zero.
    pub rsdp_phys: u64,
}

impl BootInfo {
    pub const fn empty() -> Self {
        Self {
            magic: QRATUM_BOOT_MAGIC,
            rng_seed: [0; 32],
            usable_ram_bytes: 0,
            usable_regions: 0,
            largest_region_bytes: 0,
            boot_unix_ts: 0,
            command_line: [0; 256],
            rsdp_phys: 0,
        }
    }
}

// -----------------------------------------------------------------------------
// Intent ABI (P0.0 simplified — no params blob yet; full §4.2 lands in P0.2)
// -----------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Authority {
    Autonomy = 1,
    External = 2,
    Console  = 3,
    System   = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    Actor  = 1,
    World  = 2,
    System = 3,
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Priority {
    Low      = 1,
    Normal   = 2,
    High     = 3,
    Override = 4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IntentEnvelope {
    pub magic:     u32,
    pub version:   u16,
    pub flags:     u16,
    pub id:        [u8; 16],
    pub ts:        u64,
    pub authority: u8,
    pub scope:     u8,
    pub priority:  u8,
    pub _pad0:     u8,
    pub name_len:  u16,
    pub name:      [u8; 64],
}

impl IntentEnvelope {
    pub fn name_str(&self) -> &str {
        let n = (self.name_len as usize).min(self.name.len());
        core::str::from_utf8(&self.name[..n]).unwrap_or("<bad-utf8>")
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Accepted = 0,
    Denied   = 1,
}

/// Stable kernel reason codes (§6.4 of boot-kernel spec).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    AcceptFree         = 0,
    AcceptRefresh      = 1,
    AcceptAuthority    = 2,
    AcceptSystemOvr    = 3,
    DenyConsoleDomin   = 4,
    DenyAutonomyBlock  = 5,
    DenyTimestamp      = 6,
    DenyOverrideThrash = 7,
    DenyQueueFull      = 8,
    DenySchema         = 9,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ResultEnvelope {
    pub id:               [u8; 16],
    pub verdict:          u8,
    pub reason:           u8,
    pub holder_authority: u8,
    pub _pad:             u8,
    pub error_code:       i32,
    pub completed_ts:     u64,
}
