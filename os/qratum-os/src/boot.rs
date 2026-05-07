//! Stage-1 bootloader logic (Agent 1).
//!
//! Runs while UEFI Boot Services are still alive. Acquires memory map,
//! optional RNG seed, and walks the descriptors to compute usable RAM
//! statistics. Produces a [`BootInfo`] hand-off struct.

use core::fmt::Write;
use uefi::mem::memory_map::MemoryMap;
use uefi::prelude::*;
use uefi::table::boot::MemoryType;

use crate::shared::{BootInfo, QRATUM_BOOT_MAGIC};

/// Stage-1 driver. Returns a populated [`BootInfo`] on success.
pub fn stage1(st: &mut SystemTable<Boot>) -> Result<BootInfo, Status> {
    let _ = writeln!(st.stdout(), "[BOOT] stage1: acquiring memory map");

    let mut bi = BootInfo::empty();
    bi.magic = QRATUM_BOOT_MAGIC;

    // 1) Memory map walk.
    let bs = st.boot_services();
    let mmap = bs
        .memory_map(MemoryType::LOADER_DATA)
        .map_err(|_| Status::OUT_OF_RESOURCES)?;

    let mut total: u64 = 0;
    let mut largest: u64 = 0;
    let mut regions: u32 = 0;
    for d in mmap.entries() {
        if d.ty == MemoryType::CONVENTIONAL {
            let bytes = d.page_count.saturating_mul(4096);
            total = total.saturating_add(bytes);
            if bytes > largest { largest = bytes; }
            regions += 1;
        }
    }
    bi.usable_ram_bytes      = total;
    bi.usable_regions        = regions;
    bi.largest_region_bytes  = largest;

    // 2) RNG seed (best-effort; not required for P0.0).
    if let Ok(rng) = bs.get_handle_for_protocol::<uefi::proto::rng::Rng>() {
        if let Ok(mut rng) = bs.open_protocol_exclusive::<uefi::proto::rng::Rng>(rng) {
            let _ = rng.get_rng(None, &mut bi.rng_seed);
        }
    }

    // 3) Default command line.
    let cl = b"hz=1000 console=ttyS0";
    bi.command_line[..cl.len()].copy_from_slice(cl);

    // 4) P0.7 — locate ACPI RSDP via UEFI ConfigurationTable.
    //
    // Walks the entries the firmware installed and prefers the ACPI 2.0+
    // pointer (XSDT-bearing) over the legacy 1.0 pointer (RSDT-only).
    // The RSDP's physical address is what `hw::acpi::parse` consumes
    // post-ExitBootServices.
    bi.rsdp_phys = locate_rsdp(st);
    let _ = writeln!(
        st.stdout(),
        "[BOOT] rsdp_phys = 0x{:016X} ({})",
        bi.rsdp_phys,
        if bi.rsdp_phys == 0 { "ACPI ABSENT" } else { "ACPI DETECTED" }
    );

    Ok(bi)
}

/// Scan the UEFI configuration table for an ACPI RSDP pointer.
/// Returns the **physical** address of the RSDP, or 0 if not found.
fn locate_rsdp(st: &SystemTable<Boot>) -> u64 {
    use uefi::table::cfg::{ACPI2_GUID, ACPI_GUID};
    let mut acpi1: u64 = 0;
    let mut acpi2: u64 = 0;
    for entry in st.config_table() {
        if entry.guid == ACPI2_GUID {
            acpi2 = entry.address as u64;
        } else if entry.guid == ACPI_GUID {
            acpi1 = entry.address as u64;
        }
    }
    if acpi2 != 0 { acpi2 } else { acpi1 }
}
