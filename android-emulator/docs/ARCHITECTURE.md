# Architecture Overview

This document describes the architecture of the QRATUM Android Emulator.

## System Design

### Core Principles

1. **Modularity**: Each subsystem is a separate crate with well-defined interfaces
2. **Safety**: Rust's memory safety guarantees prevent common vulnerabilities
3. **Legal Compliance**: All code is original or permissively licensed
4. **Determinism**: Optional deterministic execution for debugging and testing

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                   (CLI / GUI / Android Studio)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Emulator Core (emu-core)                    │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Config    │  │  Lifecycle  │  │  Snapshot   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Virtual Hardware                            │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │     CPU     │  │    Memory   │  │   Display   │             │
│  │  (AArch64)  │  │    (MMU)    │  │   (Vulkan)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Audio    │  │    Input    │  │  Filesystem │             │
│  │  (OpenSL)   │  │  (Touch/KB) │  │   (EXT4)    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Android Runtime                             │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │     ART     │  │    Bionic   │  │    Binder   │             │
│  │   Runtime   │  │     libc    │  │     IPC     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## CPU Emulation (emu-cpu)

### Instruction Execution Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Fetch   │───▶│  Decode  │───▶│ Execute  │───▶│ Writeback│
│          │    │          │    │          │    │          │
│ Read 4B  │    │ Parse    │    │ ALU/FPU  │    │ Update   │
│ from PC  │    │ opcode   │    │ Memory   │    │ Registers│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Supported Instructions

The decoder supports the following ARMv8-A instruction categories:

- Data Processing (Immediate & Register)
- Branches (Conditional & Unconditional)
- Loads and Stores
- System Instructions
- SIMD/FP (planned)

### Exception Handling

Exceptions are handled through the CpuError enum:

- Synchronous exceptions (SVC, HVC, SMC)
- Breakpoints
- Memory faults
- Undefined instructions

## Memory Subsystem (emu-memory)

### Address Translation

```
Virtual Address (64-bit)
         │
         ▼
┌─────────────────┐
│   Page Table    │
│   (Software)    │
└────────┬────────┘
         │
         ▼
Physical Address → Host Memory
```

### Page Table Structure

- Two-level page table (L1 + L2)
- 4KB pages (default)
- 2MB huge pages (optional)
- Copy-on-write support for snapshots

### Memory Regions

| Type | Description |
|------|-------------|
| RAM | Main system memory |
| ROM | Read-only firmware |
| MMIO | Memory-mapped I/O devices |
| Stack | Thread stack memory |
| Heap | Dynamic allocation |

## Graphics Subsystem (emu-graphics)

### Rendering Pipeline

```
┌─────────────────┐
│ Android App     │
│ (OpenGL ES)     │
└────────┬────────┘
         │
┌────────▼────────┐
│ GLES State      │
│ Tracking        │
└────────┬────────┘
         │
┌────────▼────────┐
│ Command         │
│ Translation     │
└────────┬────────┘
         │
┌────────▼────────┐
│ Vulkan          │
│ Backend         │
└────────┬────────┘
         │
┌────────▼────────┐
│ Host GPU        │
└─────────────────┘
```

### Compositor (SurfaceFlinger-like)

- Multiple layer support
- Alpha blending
- Z-ordering
- Frame pacing

## Android Runtime (emu-android-runtime)

### Component Overview

1. **ART Runtime**: DEX bytecode execution (stub)
2. **Bionic libc**: C library compatibility
3. **Binder IPC**: Inter-process communication
4. **Syscall Handler**: Linux syscall translation

### Syscall Support

The following syscall categories are supported:

- File operations (open, read, write, close)
- Memory management (mmap, mprotect, brk)
- Process control (exit, clone)
- Time and clock functions
- Random number generation

## Boot Sequence

```
1. Emulator Creation
   └── Initialize configuration
   
2. Initialization
   ├── Setup memory regions
   ├── Initialize filesystem
   ├── Setup mount points
   └── Initialize runtime

3. Boot
   ├── Set initial PC
   ├── Set stack pointer
   └── Enter running state

4. Execution Loop
   ├── Fetch instruction
   ├── Decode
   ├── Execute
   ├── Handle syscalls
   └── Update display
```

## PC Compatibility Layer (emu-pc-compat)

### Purpose

This layer is for **research purposes only**. It provides:

- Win32 syscall translation study
- DirectX to Vulkan abstraction research
- Timing normalization for compatibility testing

### Components

1. **Win32 Syscall Shim**: Translates NT syscalls to host equivalents
2. **DirectX Translator**: DXBC to SPIR-V shader conversion
3. **Timing Normalizer**: Consistent timing for deterministic execution

## Future Enhancements

- [ ] JIT compilation for hot code paths
- [ ] Full SIMD/NEON support
- [ ] Hardware acceleration (KVM/Hypervisor)
- [ ] Network stack emulation
- [ ] Multi-core CPU emulation
- [ ] GUI frontend (Qt or Flutter)

## Performance Considerations

### Optimization Strategies

1. **Instruction Caching**: Decoded instructions are cached
2. **Memory Mapping**: Direct host memory access when safe
3. **Shader Caching**: Translated shaders are cached
4. **Lock-Free Structures**: Minimal contention in hot paths

### Benchmarking

Run benchmarks with:

```bash
cargo bench
```

Key metrics:

- Instructions per second
- Memory throughput
- Graphics frame rate
- Syscall latency
