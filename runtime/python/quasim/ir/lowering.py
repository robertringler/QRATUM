"""Lowering passes for converting IR to target backends."""
from __future__ import annotations

from enum import Enum
from typing import List

from .ir_builder import IRNode, IRType


class Backend(Enum):
    """Supported compilation backends."""
    CUDA = "cuda"
    HIP = "hip"
    TRITON = "triton"
    CPU = "cpu"
    JAX = "jax"
    PYTORCH = "pytorch"


def lower_to_backend(nodes: List[IRNode], backend: Backend) -> str:
    """Lower IR nodes to target backend code."""
    if backend == Backend.CUDA:
        return _lower_to_cuda(nodes)
    elif backend == Backend.HIP:
        return _lower_to_hip(nodes)
    elif backend == Backend.TRITON:
        return _lower_to_triton(nodes)
    elif backend == Backend.CPU:
        return _lower_to_cpu(nodes)
    elif backend == Backend.JAX:
        return _lower_to_jax(nodes)
    elif backend == Backend.PYTORCH:
        return _lower_to_pytorch(nodes)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _lower_to_cuda(nodes: List[IRNode]) -> str:
    """Generate CUDA kernel code from IR nodes."""
    lines = [
        "#include <cuda_runtime.h>",
        "#include <cuda_fp16.h>",
        "",
        "__global__ void quasim_kernel(float* output, const float* input, int n) {",
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx < n) {",
    ]
    
    for node in nodes:
        if node.op == "add":
            lines.append(f"        output[idx] = input[idx] + input[idx];  // {node.op}")
        elif node.op == "mul":
            lines.append(f"        output[idx] = input[idx] * input[idx];  // {node.op}")
        elif node.op == "relu":
            lines.append(f"        output[idx] = max(0.0f, input[idx]);  // {node.op}")
            
    lines.extend([
        "    }",
        "}",
    ])
    return "\n".join(lines)


def _lower_to_hip(nodes: List[IRNode]) -> str:
    """Generate HIP kernel code from IR nodes."""
    lines = [
        "#include <hip/hip_runtime.h>",
        "",
        "__global__ void quasim_kernel_hip(float* output, const float* input, int n) {",
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx < n) {",
    ]
    
    for node in nodes:
        if node.op == "add":
            lines.append(f"        output[idx] = input[idx] + input[idx];  // {node.op}")
        elif node.op == "mul":
            lines.append(f"        output[idx] = input[idx] * input[idx];  // {node.op}")
            
    lines.extend([
        "    }",
        "}",
    ])
    return "\n".join(lines)


def _lower_to_triton(nodes: List[IRNode]) -> str:
    """Generate Triton kernel code from IR nodes."""
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        "def quasim_kernel_triton(output_ptr, input_ptr, n, BLOCK_SIZE: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offsets < n",
        "    x = tl.load(input_ptr + offsets, mask=mask)",
    ]
    
    for node in nodes:
        if node.op == "add":
            lines.append(f"    x = x + x  # {node.op}")
        elif node.op == "mul":
            lines.append(f"    x = x * x  # {node.op}")
        elif node.op == "relu":
            lines.append(f"    x = tl.maximum(0.0, x)  # {node.op}")
            
    lines.append("    tl.store(output_ptr + offsets, x, mask=mask)")
    return "\n".join(lines)


def _lower_to_cpu(nodes: List[IRNode]) -> str:
    """Generate optimized CPU code from IR nodes."""
    lines = [
        "#include <algorithm>",
        "#include <cmath>",
        "",
        "void quasim_kernel_cpu(float* output, const float* input, int n) {",
        "    #pragma omp parallel for",
        "    for (int i = 0; i < n; i++) {",
        "        float val = input[i];",
    ]
    
    for node in nodes:
        if node.op == "add":
            lines.append(f"        val = val + val;  // {node.op}")
        elif node.op == "mul":
            lines.append(f"        val = val * val;  // {node.op}")
        elif node.op == "relu":
            lines.append(f"        val = std::max(0.0f, val);  // {node.op}")
            
    lines.extend([
        "        output[i] = val;",
        "    }",
        "}",
    ])
    return "\n".join(lines)


def _lower_to_jax(nodes: List[IRNode]) -> str:
    """Generate JAX code from IR nodes."""
    lines = [
        "import jax",
        "import jax.numpy as jnp",
        "",
        "def quasim_kernel_jax(x):",
    ]
    
    for node in nodes:
        if node.op == "add":
            lines.append(f"    x = x + x  # {node.op}")
        elif node.op == "mul":
            lines.append(f"    x = x * x  # {node.op}")
        elif node.op == "relu":
            lines.append(f"    x = jax.nn.relu(x)  # {node.op}")
            
    lines.append("    return x")
    return "\n".join(lines)


def _lower_to_pytorch(nodes: List[IRNode]) -> str:
    """Generate PyTorch code from IR nodes."""
    lines = [
        "import torch",
        "import torch.nn.functional as F",
        "",
        "def quasim_kernel_pytorch(x):",
    ]
    
    for node in nodes:
        if node.op == "add":
            lines.append(f"    x = x + x  # {node.op}")
        elif node.op == "mul":
            lines.append(f"    x = x * x  # {node.op}")
        elif node.op == "relu":
            lines.append(f"    x = F.relu(x)  # {node.op}")
            
    lines.append("    return x")
    return "\n".join(lines)
