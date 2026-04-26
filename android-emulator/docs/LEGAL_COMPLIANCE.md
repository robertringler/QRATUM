# Legal Compliance Notes

## Overview

This document describes the legal compliance measures taken in the development of the QRATUM Android Emulator.

## Licensing

All code in this project is licensed under the **Apache License 2.0**.

### Permitted Uses
- Commercial use
- Modification
- Distribution
- Patent use
- Private use

### Requirements
- License and copyright notice must be included
- State changes must be documented
- Original attribution must be preserved

## Original Work Declaration

The following components are **original implementations** written from scratch:

### CPU Emulation (emu-cpu)
- ARMv8-A instruction decoder: Based on public ARM Architecture Reference Manual
- Interpreter/executor: Original implementation
- Register file: Based on documented ARM register model

### Memory Subsystem (emu-memory)
- Page table implementation: Original design
- MMU emulation: Original implementation
- Snapshot system: Original implementation

### Android Runtime (emu-android-runtime)
- Binder IPC: Clean-room implementation based on documented interfaces
- Bionic compatibility: Based on public AOSP documentation
- Syscall handler: Based on public Linux syscall documentation

### Graphics (emu-graphics)
- OpenGL ES state tracking: Original implementation
- Compositor: Original SurfaceFlinger-like implementation
- Framebuffer management: Original implementation

### Audio (emu-audio)
- OpenSL ES emulation: Based on public Khronos specifications
- Audio mixer: Original implementation
- Stream management: Original implementation

### Filesystem (emu-filesystem)
- VFS layer: Original implementation
- EXT4 emulation: Based on public filesystem specifications
- Mount management: Original implementation

### PC Compatibility (emu-pc-compat)
- Win32 syscall shim: Research implementation based on documented APIs
- DirectX abstraction: Research implementation based on public specifications
- Timing normalization: Original implementation

## What This Project Does NOT Include

### Proprietary Software
- ❌ No EA games or assets
- ❌ No The Sims 4 binaries or assets
- ❌ No proprietary Android apps
- ❌ No copyrighted game content

### Reverse Engineering
- ❌ No reverse-engineered proprietary code
- ❌ No decompiled binaries
- ❌ No extracted firmware

### DRM/Protection Bypass
- ❌ No DRM circumvention code
- ❌ No license validation bypass
- ❌ No copy protection defeat

## User Responsibilities

Users of this emulator must:

1. **Provide Legal Software**: Users must supply their own legally obtained Android system images and applications
2. **Comply with EULAs**: Users must respect the end-user license agreements of software they run
3. **Respect Copyrights**: Users must not use this emulator to infringe copyrights
4. **Follow Local Laws**: Users must comply with applicable laws in their jurisdiction

## Third-Party Dependencies

All dependencies are permissively licensed:

| Dependency | License | Purpose |
|------------|---------|---------|
| log | MIT/Apache-2.0 | Logging |
| thiserror | MIT/Apache-2.0 | Error handling |
| anyhow | MIT/Apache-2.0 | Error handling |
| serde | MIT/Apache-2.0 | Serialization |
| parking_lot | MIT/Apache-2.0 | Synchronization |
| clap | MIT/Apache-2.0 | CLI parsing |
| tokio | MIT | Async runtime |
| bitflags | MIT/Apache-2.0 | Bitfield utilities |
| byteorder | MIT/Unlicense | Byte ordering |
| memmap2 | MIT/Apache-2.0 | Memory mapping |

## Research Exception

The PC Compatibility Layer (emu-pc-compat) is provided for **research purposes only**:

- It implements documented public APIs
- It does not include any proprietary code
- It is intended for compatibility research, not piracy
- Users must provide their own legally obtained software

## Contact

For legal inquiries, contact: legal@qratum.ai

## Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE AUTHORS ARE NOT RESPONSIBLE FOR ANY MISUSE OF THIS SOFTWARE. USERS ASSUME ALL RESPONSIBILITY FOR COMPLIANCE WITH APPLICABLE LAWS AND SOFTWARE LICENSES.
