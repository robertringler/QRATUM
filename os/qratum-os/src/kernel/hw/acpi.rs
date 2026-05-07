//! P0.7 — Minimal ACPI walker. Locates RSDP → XSDT/RSDT → MADT/HPET.
//!
//! Only fields the kernel needs are extracted:
//!
//! * Local-APIC base address (MADT header)
//! * Per-CPU LAPIC entries (`(processor_uid, apic_id, flags)`)
//! * IO-APIC entries (address + GSI base)
//! * HPET base address (HPET table)

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

const MAX_LAPICS:   usize = 32;
const MAX_IOAPICS:  usize = 8;

#[derive(Copy, Clone, Debug, Default)]
pub struct LapicEntry {
    pub processor_uid: u8,
    pub apic_id:       u8,
    pub enabled:       bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct IoApicEntry {
    pub id:            u8,
    pub address:       u32,
    pub gsi_base:      u32,
}

static LAPIC_BASE: AtomicU64 = AtomicU64::new(0);
static HPET_BASE:  AtomicU64 = AtomicU64::new(0);
static N_LAPIC:    AtomicU8  = AtomicU8::new(0);
static N_IOAPIC:   AtomicU8  = AtomicU8::new(0);

static mut LAPICS:  [LapicEntry; MAX_LAPICS]  = [LapicEntry { processor_uid: 0, apic_id: 0, enabled: false }; MAX_LAPICS];
static mut IOAPICS: [IoApicEntry; MAX_IOAPICS] = [IoApicEntry { id: 0, address: 0, gsi_base: 0 }; MAX_IOAPICS];

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AcpiError { BadRsdp, BadChecksum, NoMadt }

/// Parse the ACPI tree starting from a known RSDP physical address.
///
/// # Safety
/// `rsdp_addr` must be a valid physical address in the kernel's
/// identity-map. UEFI delivers it via `ConfigurationTable`.
pub unsafe fn parse(rsdp_addr: u64) -> Result<(), AcpiError> {
    let p = rsdp_addr as *const u8;
    if &*core::ptr::slice_from_raw_parts(p, 8) != b"RSD PTR " {
        return Err(AcpiError::BadRsdp);
    }
    // RSDP v2 has XSDT at offset 24 (u64). v1 has RSDT at offset 16 (u32).
    let revision = *p.add(15);
    let sdt_addr: u64 = if revision >= 2 {
        let mut b = [0u8; 8];
        for i in 0..8 { b[i] = *p.add(24 + i); }
        u64::from_le_bytes(b)
    } else {
        let mut b = [0u8; 4];
        for i in 0..4 { b[i] = *p.add(16 + i); }
        u32::from_le_bytes(b) as u64
    };
    walk_root(sdt_addr, revision >= 2)
}

unsafe fn walk_root(addr: u64, is_xsdt: bool) -> Result<(), AcpiError> {
    let hdr = addr as *const u8;
    let length = u32::from_le_bytes([*hdr.add(4), *hdr.add(5), *hdr.add(6), *hdr.add(7)]) as usize;
    let entries_off = 36;
    let stride = if is_xsdt { 8 } else { 4 };
    let n = (length - entries_off) / stride;

    let mut found_madt = false;
    for i in 0..n {
        let p = hdr.add(entries_off + i * stride);
        let entry_addr: u64 = if is_xsdt {
            let mut b = [0u8; 8]; for k in 0..8 { b[k] = *p.add(k); }
            u64::from_le_bytes(b)
        } else {
            let mut b = [0u8; 4]; for k in 0..4 { b[k] = *p.add(k); }
            u32::from_le_bytes(b) as u64
        };
        let sig = core::slice::from_raw_parts(entry_addr as *const u8, 4);
        match sig {
            b"APIC" => { parse_madt(entry_addr); found_madt = true; }
            b"HPET" => { parse_hpet(entry_addr); }
            _ => {}
        }
    }
    if !found_madt { return Err(AcpiError::NoMadt); }
    Ok(())
}

unsafe fn parse_madt(addr: u64) {
    let p = addr as *const u8;
    let length = u32::from_le_bytes([*p.add(4), *p.add(5), *p.add(6), *p.add(7)]) as usize;
    let lapic_addr = u32::from_le_bytes([*p.add(36), *p.add(37), *p.add(38), *p.add(39)]) as u64;
    LAPIC_BASE.store(lapic_addr, Ordering::Release);

    let mut off = 44usize;
    while off + 2 <= length {
        let kind = *p.add(off);
        let len  = *p.add(off + 1) as usize;
        if len == 0 { break; }
        match kind {
            0 => { // Processor Local APIC
                let uid    = *p.add(off + 2);
                let apicid = *p.add(off + 3);
                let flags  = u32::from_le_bytes([*p.add(off + 4), *p.add(off + 5), *p.add(off + 6), *p.add(off + 7)]);
                let idx = N_LAPIC.load(Ordering::Acquire) as usize;
                if idx < MAX_LAPICS {
                    LAPICS[idx] = LapicEntry { processor_uid: uid, apic_id: apicid, enabled: (flags & 1) != 0 };
                    N_LAPIC.store((idx + 1) as u8, Ordering::Release);
                }
            }
            1 => { // I/O APIC
                let id   = *p.add(off + 2);
                let addr = u32::from_le_bytes([*p.add(off + 4), *p.add(off + 5), *p.add(off + 6), *p.add(off + 7)]);
                let gsi  = u32::from_le_bytes([*p.add(off + 8), *p.add(off + 9), *p.add(off + 10), *p.add(off + 11)]);
                let idx = N_IOAPIC.load(Ordering::Acquire) as usize;
                if idx < MAX_IOAPICS {
                    IOAPICS[idx] = IoApicEntry { id, address: addr, gsi_base: gsi };
                    N_IOAPIC.store((idx + 1) as u8, Ordering::Release);
                }
            }
            9 => { // Processor Local x2APIC
                let apicid = u32::from_le_bytes([*p.add(off + 4), *p.add(off + 5), *p.add(off + 6), *p.add(off + 7)]);
                let uid    = u32::from_le_bytes([*p.add(off + 12), *p.add(off + 13), *p.add(off + 14), *p.add(off + 15)]);
                let flags  = u32::from_le_bytes([*p.add(off + 8), *p.add(off + 9), *p.add(off + 10), *p.add(off + 11)]);
                let idx = N_LAPIC.load(Ordering::Acquire) as usize;
                if idx < MAX_LAPICS {
                    LAPICS[idx] = LapicEntry {
                        processor_uid: uid as u8,
                        apic_id: apicid as u8,
                        enabled: (flags & 1) != 0,
                    };
                    N_LAPIC.store((idx + 1) as u8, Ordering::Release);
                }
            }
            _ => {}
        }
        off += len;
    }
}

unsafe fn parse_hpet(addr: u64) {
    let p = addr as *const u8;
    // HPET table: address structure starts at offset 44 (Generic Address Structure).
    // GAS layout: [0]=AddrSpaceID [1]=BitWidth [2]=BitOff [3]=AccessSize [4..12]=Address.
    let mut b = [0u8; 8];
    for i in 0..8 { b[i] = *p.add(44 + 4 + i); }
    HPET_BASE.store(u64::from_le_bytes(b), Ordering::Release);
}

#[inline]
pub fn lapic_base() -> Option<u64> {
    let v = LAPIC_BASE.load(Ordering::Acquire);
    if v == 0 { None } else { Some(v) }
}

#[inline]
pub fn hpet_address() -> Option<u64> {
    let v = HPET_BASE.load(Ordering::Acquire);
    if v == 0 { None } else { Some(v) }
}

pub fn lapics() -> &'static [LapicEntry] {
    let n = N_LAPIC.load(Ordering::Acquire) as usize;
    unsafe { &LAPICS[..n] }
}

pub fn ioapics() -> &'static [IoApicEntry] {
    let n = N_IOAPIC.load(Ordering::Acquire) as usize;
    unsafe { &IOAPICS[..n] }
}

/// P0.7 Phase 2 — minimal APIC topology view consumed by `init_sipi`.
/// `bsp_id` is identified by reading the LAPIC ID of the running CPU
/// before any AP comes up; here we expose only the data ACPI gave us.
#[derive(Copy, Clone, Debug)]
pub struct ApicTopology {
    pub lapic_base: u64,
    /// All LAPIC IDs reported by the MADT (BSP + APs).
    pub all_ids:    [u8; MAX_LAPICS],
    pub n_total:    u8,
}

pub fn topology() -> Option<ApicTopology> {
    let base = lapic_base()?;
    let entries = lapics();
    if entries.is_empty() { return None; }
    let mut ids = [0u8; MAX_LAPICS];
    let mut n: u8 = 0;
    for e in entries {
        if e.enabled && (n as usize) < MAX_LAPICS {
            ids[n as usize] = e.apic_id;
            n += 1;
        }
    }
    Some(ApicTopology { lapic_base: base, all_ids: ids, n_total: n })
}
