# P0.7 — Bare-Metal Validation Procedure

> **You** must run this. The agent built and verified the model on host but
> **cannot** image USB media, control physical hardware, or capture a serial
> console on real silicon.

## What this validates

The P0.7 exit criterion: *"Kernel boots on physical machine with multi-core
scheduling active."*

This procedure currently validates the **scaffolded** hardware layer
(`hw::init()` invoked from `kmain`). The long-mode AP trampoline bridge
(deferred to P0.7-followup) is required before "multi-core scheduling
active" can be asserted. Until then, the success criterion below is
*"BSP reaches `BootReady`; APs accept SIPI without triple-faulting"*.

---

## Required hardware

* Test machine with x86_64 UEFI firmware, ≥ 2 cores, USB boot enabled in
  firmware setup. **Avoid** machines that disable legacy A20/real-mode at
  reset (some recent server boards) — the AP trampoline currently lives in
  real mode.
* USB drive ≥ 256 MiB (will be wiped).
* USB-to-TTL serial adapter (FTDI / CH340), 3.3 V logic, terminated to the
  target's COM1 header **or** a PCI/PCIe serial card. If neither is
  available, fall back to firmware "BIOS over EFI" framebuffer capture.
* A second machine running `screen` / PuTTY at **115 200 8N1** (the rate
  used by `kernel/serial.rs`).

## Build the bootable image

```powershell
cd c:\Users\rober\QRATUM-1\os\qratum-os
cargo +nightly-2025-04-01 build -p qratum-os `
  "-Zbuild-std=core,compiler_builtins,alloc" `
  "-Zbuild-std-features=compiler-builtins-mem"
```

Output: `target/x86_64-unknown-uefi/debug/qratum.efi` (~125 KiB).

## Image to USB

### Windows (Rufus)

1. Format USB as **FAT32**, MBR, single partition.
2. Create directory `\EFI\BOOT\` on the USB.
3. Copy `qratum.efi` → `\EFI\BOOT\BOOTX64.EFI` on the USB.
4. Eject cleanly.

### Linux

```bash
sudo umount /dev/sdX1 || true
sudo mkfs.vfat -F32 /dev/sdX1
sudo mount /dev/sdX1 /mnt
sudo mkdir -p /mnt/EFI/BOOT
sudo cp target/x86_64-unknown-uefi/debug/qratum.efi /mnt/EFI/BOOT/BOOTX64.EFI
sudo umount /mnt
```

## Configure the target

1. Enter firmware setup (DEL / F2 / F12 — vendor-specific).
2. Disable Secure Boot (the binary is unsigned).
3. Disable CSM / legacy boot.
4. Confirm "USB UEFI" is in the boot order **above** the internal disk.
5. Connect the serial cable to COM1 (or the PCI serial card).

## Capture procedure

On the capture machine:

```bash
# Linux
sudo screen /dev/ttyUSB0 115200

# Windows (PuTTY)
# Connection type: Serial. Line: COM3 (your adapter). Speed: 115200.
# Logging: Printable output → boot_capture.log
```

## Boot the target

1. Insert USB; power on.
2. If the firmware boot menu appears, select the USB UEFI entry.

## Expected serial output (current scaffolded slice)

```
[QRATUM] stage1: ExitBootServices ok
[QRATUM] kmain: BootBegin seq=1
[QRATUM] kmain: gdt+idt installed
[QRATUM] kmain: pic remapped, masked
[QRATUM] kmain: BootReady seq=N (audit chain valid)
[QRATUM] qsh> _
```

(no `hw::init` lines yet — gating until this baseline boot is confirmed
on your silicon).

## Success criteria

| # | Check | Pass condition |
|---|-------|----------------|
| 1 | Boot reaches `BootReady` | seq ≥ 4, audit chain prints OK |
| 2 | No spontaneous reset | machine stays in `qsh>` prompt for ≥ 60 s |
| 3 | Audit ring intact | `qsh> audit tail` prints frames in order |

## Failure triage

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Triple fault before `BootBegin` | Stack/CR3 mismatch post-ExitBootServices | Capture last 16 bytes of serial; report. Do **not** rebuild — open a P0.7-bug ticket. |
| Hangs after `pic remapped` | IRQ storm (legacy device firing into masked PIC) | Confirm BIOS legacy USB mass-storage emulation is OFF. |
| Garbled serial | Wrong baud / wrong COM port | Re-check 115200 8N1; try alternate COM header. |
| Boot loops | Firmware rejects the EFI binary | Check Secure Boot is OFF; binary is at `\EFI\BOOT\BOOTX64.EFI` exact path. |

## Reporting

After capture, save:

* `boot_capture.log` (full serial transcript)
* Photo of any framebuffer message if serial unavailable
* Target machine model + CPU + firmware version

Drop them under `reports/baremetal/<machine_id>/` and open a follow-up
ticket. **The agent will not advance to P0.8 until at least one
successful bare-metal report is filed.**
