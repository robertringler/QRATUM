# QRATUM Android Emulator

A fully functional, legally compliant Android emulator and PC game compatibility research framework.

## Overview

This project provides:
- **Core Emulator Kernel**: ARMv8 ↔ x86-64 CPU emulation with dynamic binary translation
- **Android Runtime Layer**: Clean-room ART implementation with Bionic libc compatibility
- **GPU & Graphics**: Vulkan translation layer with OpenGL ES backend
- **Audio & Input**: OpenSL ES emulation with touch/keyboard/gamepad support
- **Filesystem & I/O**: Virtual EXT4 filesystem with sandboxed permissions
- **PC Game Compatibility Research Layer**: Win32 syscall translation and DirectX abstraction

## Legal Notice

⚠️ **IMPORTANT**: This implementation is original work written from scratch:
- No proprietary code is included
- No copyrighted game assets are bundled
- No DRM bypass or reverse-engineered code
- All components are licensed under Apache 2.0

Users must provide their own legally obtained software to use with this emulator.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         emu-cli                              │
│                    (Command Line Interface)                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                         emu-core                             │
│              (Emulator Orchestration & Lifecycle)            │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────┬───────────┼───────────┬─────────────┐
    │             │           │           │             │
┌───┴───┐   ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐ ┌─────┴─────┐
│emu-cpu│   │emu-android│ │emu-   │ │  emu-     │ │  emu-     │
│       │   │ -runtime  │ │graphics│ │  audio    │ │  input    │
└───────┘   └───────────┘ └───────┘ └───────────┘ └───────────┘
    │             │
┌───┴─────────────┴───┐                      ┌─────────────────┐
│     emu-memory      │                      │ emu-filesystem  │
│   (MMU, Page Tables)│                      │  (VFS, EXT4)    │
└─────────────────────┘                      └─────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     emu-pc-compat                            │
│          (PC Game Compatibility Research Layer)              │
└─────────────────────────────────────────────────────────────┘
```

## Building

### Prerequisites

- Rust 1.75+ (with `cargo`)
- Linux or macOS (Windows support planned)
- Optional: Vulkan SDK for hardware acceleration

### Build Commands

```bash
# Build all crates
cd android-emulator
cargo build --release

# Run tests
cargo test

# Build documentation
cargo doc --open

# Run the emulator
cargo run --release --bin emu -- run --headless
```

## Usage

### Basic Commands

```bash
# Show help
emu --help

# List available device profiles
emu list-profiles

# Run emulator with phone profile
emu run --profile phone

# Run in headless mode
emu run --headless

# Run with custom settings
emu run --profile tablet --ram 8192 --cores 8

# Create a snapshot
emu snapshot create my-snapshot

# Restore from snapshot
emu snapshot restore my-snapshot

# Run execution trace
emu trace --output trace.log --count 10000
```

### Device Profiles

| Profile | Resolution | DPI | RAM |
|---------|-----------|-----|-----|
| phone | 1080x1920 | 420 | 4 GB |
| phone_hd | 720x1280 | 320 | 4 GB |
| tablet | 1920x1200 | 240 | 8 GB |
| tablet_hd | 2560x1600 | 320 | 8 GB |
| tv | 1920x1080 | 320 | 2 GB |

## Crate Overview

| Crate | Description |
|-------|-------------|
| `emu-core` | Main emulator orchestration and lifecycle |
| `emu-cpu` | ARMv8 (AArch64) CPU emulation and execution |
| `emu-memory` | Virtual memory, MMU, page tables, snapshots |
| `emu-android-runtime` | ART, Bionic libc, Binder IPC emulation |
| `emu-graphics` | OpenGL ES to Vulkan translation, compositor |
| `emu-audio` | OpenSL ES emulation, audio mixing |
| `emu-input` | Touch, keyboard, gamepad input handling |
| `emu-filesystem` | Virtual EXT4 filesystem, mount management |
| `emu-pc-compat` | Win32 syscall shim, DirectX translation |
| `emu-cli` | Command-line interface |

## Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific crate tests
cargo test -p emu-cpu

# Run benchmarks
cargo bench
```

## Performance

The emulator uses several optimization techniques:
- Dynamic binary translation for hot code paths
- Copy-on-write memory snapshots
- Shader caching for graphics translation
- Lock-free data structures where possible

## Contributing

Contributions are welcome! Please ensure:
1. All code is original or properly licensed (Apache 2.0, MIT, BSD)
2. No proprietary code or reverse-engineered material
3. Tests are included for new functionality
4. Documentation is updated

## License

Licensed under Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Disclaimer

This software is provided for educational and research purposes. The developers are not responsible for any misuse. Users must comply with all applicable laws and software licenses when using this emulator.
