"""AION Backends Module.

Implements code generation backends for multiple targets:
- LLVM: CPU and JIT
- CUDA/PTX/SPIR-V: GPU
- FPGA: Verilog via HLS
- WASM: WebAssembly
- JVM/CLR: Managed runtimes

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .llvm_backend import LLVMBackend, LLVMCodeGen
from .wasm_backend import WASMBackend, WASMCodeGen

__all__ = [
    "LLVMBackend",
    "LLVMCodeGen",
    "WASMBackend",
    "WASMCodeGen",
]
