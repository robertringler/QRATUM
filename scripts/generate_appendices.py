#!/usr/bin/env python3
"""Generate comprehensive appendices for technical documentation.

This module generates detailed appendices including:
- YAML benchmark specifications
- CUDA pseudocode
- Statistical derivations
- Reproducibility proofs
- Multi-format reporting examples

Author: QuASIM Engineering Team
Date: 2025-12-14
Version: 1.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_yaml_benchmark_spec(output_dir: Path) -> Path:
    """Generate YAML benchmark specification appendix.

    Args:
        output_dir: Output directory for appendix

    Returns:
        Path to generated file
    """

    output_path = output_dir / "appendix_a_benchmark_specs.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Appendix A: Benchmark Specifications\n\n")
        f.write("## BM_001: Large-Strain Rubber Block Compression\n\n")
        f.write("### YAML Specification\n\n")
        f.write("```yaml\n")
        f.write("benchmark:\n")
        f.write("  id: BM_001\n")
        f.write("  name: Large-Strain Rubber Block Compression\n")
        f.write("  version: 1.0.0\n")
        f.write("  \n")
        f.write("  problem:\n")
        f.write("    description: Nonlinear hyperelastic compression test\n")
        f.write("    domain: Structural mechanics\n")
        f.write("    physics: Large-strain elasticity\n")
        f.write("    \n")
        f.write("  geometry:\n")
        f.write("    type: cuboid\n")
        f.write("    dimensions: [100, 100, 100]  # mm\n")
        f.write("    mesh_size: 5.0  # mm\n")
        f.write("    element_type: SOLID186\n")
        f.write("    \n")
        f.write("  material:\n")
        f.write("    model: mooney_rivlin\n")
        f.write("    parameters:\n")
        f.write("      c10: 0.8  # MPa\n")
        f.write("      c01: 0.2  # MPa\n")
        f.write("      bulk_modulus: 1000.0  # MPa\n")
        f.write("      density: 1100.0  # kg/m^3\n")
        f.write("    \n")
        f.write("  loading:\n")
        f.write("    type: displacement_controlled\n")
        f.write("    magnitude: -50.0  # mm (50% compression)\n")
        f.write("    direction: [0, 0, -1]\n")
        f.write("    steps: 10\n")
        f.write("    \n")
        f.write("  boundary_conditions:\n")
        f.write("    - type: fixed\n")
        f.write("      location: bottom_face\n")
        f.write("      dof: [ux, uy, uz]\n")
        f.write("    - type: displacement\n")
        f.write("      location: top_face\n")
        f.write("      value: -50.0  # mm\n")
        f.write("      \n")
        f.write("  solver:\n")
        f.write("    type: nonlinear\n")
        f.write("    algorithm: newton_raphson\n")
        f.write("    convergence:\n")
        f.write("      displacement: 1.0e-6\n")
        f.write("      force: 1.0e-3\n")
        f.write("      max_iterations: 100\n")
        f.write("      \n")
        f.write("  acceptance_criteria:\n")
        f.write("    speedup_min: 3.0\n")
        f.write("    displacement_error_max: 0.02  # 2%\n")
        f.write("    stress_error_max: 0.05  # 5%\n")
        f.write("    energy_error_max: 1.0e-6\n")
        f.write("    coefficient_variation_max: 0.02  # 2%\n")
        f.write("    \n")
        f.write("  execution:\n")
        f.write("    runs_per_solver: 5\n")
        f.write("    cooldown_seconds: 5\n")
        f.write("    seed: 42\n")
        f.write("    device: gpu\n")
        f.write("```\n\n")

    logger.info(f"Generated benchmark spec: {output_path}")
    return output_path


def generate_cuda_pseudocode(output_dir: Path) -> Path:
    """Generate CUDA kernel pseudocode appendix.

    Args:
        output_dir: Output directory for appendix

    Returns:
        Path to generated file
    """

    output_path = output_dir / "appendix_b_cuda_pseudocode.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Appendix B: CUDA Kernel Pseudocode\n\n")
        f.write("## Tensor Contraction Kernel\n\n")
        f.write("### High-Level Algorithm\n\n")
        f.write("```python\n")
        f.write("def tensor_contraction_kernel(\n")
        f.write("    tensor_a: Tensor,\n")
        f.write("    tensor_b: Tensor,\n")
        f.write("    contraction_indices: List[Tuple[int, int]],\n")
        f.write("    device: str = 'gpu'\n")
        f.write(") -> Tensor:\n")
        f.write('    """\n')
        f.write("    GPU-accelerated tensor contraction.\n")
        f.write("    \n")
        f.write("    Args:\n")
        f.write("        tensor_a: First input tensor\n")
        f.write("        tensor_b: Second input tensor\n")
        f.write("        contraction_indices: Indices to contract\n")
        f.write("        device: Compute device ('cpu' or 'gpu')\n")
        f.write("    \n")
        f.write("    Returns:\n")
        f.write("        Contracted output tensor\n")
        f.write('    """\n')
        f.write("    # 1. Validate input tensors\n")
        f.write("    assert tensor_a.ndim >= 2\n")
        f.write("    assert tensor_b.ndim >= 2\n")
        f.write("    \n")
        f.write("    # 2. Determine optimal contraction path\n")
        f.write("    contraction_path = optimize_contraction_path(\n")
        f.write("        tensor_a.shape,\n")
        f.write("        tensor_b.shape,\n")
        f.write("        contraction_indices\n")
        f.write("    )\n")
        f.write("    \n")
        f.write("    # 3. Allocate GPU memory\n")
        f.write("    if device == 'gpu':\n")
        f.write("        tensor_a_gpu = cuda.to_device(tensor_a)\n")
        f.write("        tensor_b_gpu = cuda.to_device(tensor_b)\n")
        f.write("        output_gpu = cuda.device_array(output_shape)\n")
        f.write("    \n")
        f.write("    # 4. Launch CUDA kernel\n")
        f.write("    threads_per_block = (16, 16)\n")
        f.write("    blocks_per_grid = compute_grid_size(output_shape, threads_per_block)\n")
        f.write("    \n")
        f.write("    tensor_contract_cuda[blocks_per_grid, threads_per_block](\n")
        f.write("        tensor_a_gpu,\n")
        f.write("        tensor_b_gpu,\n")
        f.write("        output_gpu,\n")
        f.write("        contraction_indices\n")
        f.write("    )\n")
        f.write("    \n")
        f.write("    # 5. Copy result back to host\n")
        f.write("    output = output_gpu.copy_to_host()\n")
        f.write("    \n")
        f.write("    return output\n")
        f.write("```\n\n")

        f.write("### CUDA Kernel Implementation\n\n")
        f.write("```cuda\n")
        f.write("__global__ void tensor_contract_cuda(\n")
        f.write("    const float* tensor_a,\n")
        f.write("    const float* tensor_b,\n")
        f.write("    float* output,\n")
        f.write("    const int* indices,\n")
        f.write("    const int* shapes,\n")
        f.write("    const int ndim_a,\n")
        f.write("    const int ndim_b\n")
        f.write(") {\n")
        f.write("    // Thread index\n")
        f.write("    int tx = blockIdx.x * blockDim.x + threadIdx.x;\n")
        f.write("    int ty = blockIdx.y * blockDim.y + threadIdx.y;\n")
        f.write("    \n")
        f.write("    // Shared memory for tile-based computation\n")
        f.write("    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];\n")
        f.write("    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];\n")
        f.write("    \n")
        f.write("    float sum = 0.0f;\n")
        f.write("    \n")
        f.write("    // Loop over tiles\n")
        f.write("    for (int tile = 0; tile < num_tiles; ++tile) {\n")
        f.write("        // Load tile from global memory\n")
        f.write("        tile_a[threadIdx.y][threadIdx.x] = tensor_a[...];\n")
        f.write("        tile_b[threadIdx.y][threadIdx.x] = tensor_b[...];\n")
        f.write("        __syncthreads();\n")
        f.write("        \n")
        f.write("        // Compute partial contraction\n")
        f.write("        for (int k = 0; k < TILE_SIZE; ++k) {\n")
        f.write("            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];\n")
        f.write("        }\n")
        f.write("        __syncthreads();\n")
        f.write("    }\n")
        f.write("    \n")
        f.write("    // Write result to global memory\n")
        f.write("    if (tx < output_width && ty < output_height) {\n")
        f.write("        output[ty * output_width + tx] = sum;\n")
        f.write("    }\n")
        f.write("}\n")
        f.write("```\n\n")

    logger.info(f"Generated CUDA pseudocode: {output_path}")
    return output_path


def generate_statistical_derivations(output_dir: Path) -> Path:
    """Generate statistical methods derivations appendix.

    Args:
        output_dir: Output directory for appendix

    Returns:
        Path to generated file
    """

    output_path = output_dir / "appendix_c_statistical_methods.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Appendix C: Statistical Methods and Derivations\n\n")

        f.write("## Bootstrap Confidence Intervals\n\n")
        f.write("### Method\n\n")
        f.write("Bootstrap resampling provides non-parametric confidence intervals:\n\n")
        f.write("1. Original sample: $X = \\{x_1, x_2, ..., x_n\\}$\n")
        f.write("2. Generate $B$ bootstrap samples by sampling with replacement\n")
        f.write("3. Compute statistic $\\hat{\\theta}^*_b$ for each bootstrap sample\n")
        f.write(
            "4. Confidence interval: $(\\hat{\\theta}_{\\alpha/2}, \\hat{\\theta}_{1-\\alpha/2})$\n\n"
        )

        f.write("### Implementation\n\n")
        f.write("```python\n")
        f.write(
            "def bootstrap_ci(data: np.ndarray, num_samples: int = 1000, alpha: float = 0.05):\n"
        )
        f.write('    """Compute bootstrap confidence interval."""\n')
        f.write("    bootstrap_means = []\n")
        f.write("    \n")
        f.write("    for _ in range(num_samples):\n")
        f.write("        # Resample with replacement\n")
        f.write("        sample = np.random.choice(data, size=len(data), replace=True)\n")
        f.write("        bootstrap_means.append(np.mean(sample))\n")
        f.write("    \n")
        f.write("    # Compute percentiles\n")
        f.write("    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)\n")
        f.write("    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))\n")
        f.write("    \n")
        f.write("    return ci_lower, ci_upper\n")
        f.write("```\n\n")

        f.write("## Modified Z-Score for Outlier Detection\n\n")
        f.write("### Method\n\n")
        f.write(
            "Modified Z-score uses median absolute deviation (MAD) for robust outlier detection:\n\n"
        )
        f.write("$$M_i = \\frac{0.6745(x_i - \\tilde{x})}{MAD}$$\n\n")
        f.write("where:\n")
        f.write("- $\\tilde{x}$ is the median\n")
        f.write("- $MAD = median(|x_i - \\tilde{x}|)$\n")
        f.write("- Threshold: $|M_i| > 3.5$ indicates outlier\n\n")

        f.write("### Implementation\n\n")
        f.write("```python\n")
        f.write("def modified_z_score(data: np.ndarray, threshold: float = 3.5):\n")
        f.write('    """Identify outliers using modified Z-score."""\n')
        f.write("    median = np.median(data)\n")
        f.write("    mad = np.median(np.abs(data - median))\n")
        f.write("    \n")
        f.write("    # Compute modified Z-scores\n")
        f.write("    modified_z = 0.6745 * (data - median) / mad\n")
        f.write("    \n")
        f.write("    # Identify outliers\n")
        f.write("    outliers = np.abs(modified_z) > threshold\n")
        f.write("    \n")
        f.write("    return outliers, modified_z\n")
        f.write("```\n\n")

        f.write("## Coefficient of Variation\n\n")
        f.write("### Definition\n\n")
        f.write("$$CV = \\frac{\\sigma}{\\mu}$$\n\n")
        f.write("where $\\sigma$ is standard deviation and $\\mu$ is mean.\n\n")
        f.write("### Interpretation\n\n")
        f.write("- CV < 0.02 (2%): Excellent reproducibility\n")
        f.write("- CV < 0.05 (5%): Good reproducibility\n")
        f.write("- CV > 0.10 (10%): Poor reproducibility\n\n")

    logger.info(f"Generated statistical methods: {output_path}")
    return output_path


def generate_reproducibility_proof(output_dir: Path) -> Path:
    """Generate reproducibility proof appendix.

    Args:
        output_dir: Output directory for appendix

    Returns:
        Path to generated file
    """

    output_path = output_dir / "appendix_d_reproducibility_proof.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Appendix D: Reproducibility Verification Protocol\n\n")

        f.write("## SHA-256 State Verification\n\n")
        f.write("### Method\n\n")
        f.write("Cryptographic hashing ensures deterministic execution:\n\n")
        f.write("```python\n")
        f.write("import hashlib\n")
        f.write("import numpy as np\n")
        f.write("\n")
        f.write("def compute_state_hash(state_vector: np.ndarray) -> str:\n")
        f.write('    """Compute SHA-256 hash of state vector."""\n')
        f.write("    # Convert to bytes (preserves bit-exact representation)\n")
        f.write("    state_bytes = state_vector.tobytes()\n")
        f.write("    \n")
        f.write("    # Compute SHA-256 hash\n")
        f.write("    hash_obj = hashlib.sha256(state_bytes)\n")
        f.write("    hash_hex = hash_obj.hexdigest()\n")
        f.write("    \n")
        f.write("    return hash_hex\n")
        f.write("```\n\n")

        f.write("### Verification Protocol\n\n")
        f.write("1. **Initialization:** Set fixed RNG seed (e.g., 42)\n")
        f.write("2. **Execution:** Run solver with deterministic settings\n")
        f.write("3. **Hash Computation:** Compute SHA-256 of result vector\n")
        f.write("4. **Comparison:** Compare hash against reference\n")
        f.write("5. **Validation:** Assert bit-exact match\n\n")

        f.write("### Example Verification\n\n")
        f.write("```python\n")
        f.write("# Run 1\n")
        f.write("np.random.seed(42)\n")
        f.write("result_1 = run_solver()\n")
        f.write("hash_1 = compute_state_hash(result_1)\n")
        f.write("\n")
        f.write("# Run 2 (identical seed)\n")
        f.write("np.random.seed(42)\n")
        f.write("result_2 = run_solver()\n")
        f.write("hash_2 = compute_state_hash(result_2)\n")
        f.write("\n")
        f.write("# Verify reproducibility\n")
        f.write("assert hash_1 == hash_2, 'Non-deterministic execution detected!'\n")
        f.write("print(f'✓ Reproducibility verified: {hash_1}')\n")
        f.write("```\n\n")

        f.write("### Temporal Drift Tolerance\n\n")
        f.write("For production systems, we require:\n")
        f.write("- **Bit-exact reproducibility** for identical hardware\n")
        f.write("- **<1μs drift tolerance** across different runs\n")
        f.write("- **Platform consistency** between CPU and GPU execution\n\n")

    logger.info(f"Generated reproducibility proof: {output_path}")
    return output_path


def generate_reporting_examples(output_dir: Path) -> Path:
    """Generate multi-format reporting examples appendix.

    Args:
        output_dir: Output directory for appendix

    Returns:
        Path to generated file
    """

    output_path = output_dir / "appendix_e_reporting_formats.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Appendix E: Multi-Format Reporting Examples\n\n")

        f.write("## CSV Format\n\n")
        f.write("```csv\n")
        f.write("run_id,solver,execution_time,displacement,stress,energy,hash\n")
        f.write("1,ansys,120.45,0.0,50.234,1.2345e-5,abc123...\n")
        f.write("1,quasim,35.23,0.015,50.189,1.2346e-5,def456...\n")
        f.write("2,ansys,121.12,0.0,50.241,1.2344e-5,abc123...\n")
        f.write("2,quasim,35.67,0.014,50.195,1.2347e-5,def456...\n")
        f.write("```\n\n")

        f.write("## JSON Format\n\n")
        f.write("```json\n")
        f.write("{\n")
        f.write('  "benchmark": "BM_001",\n')
        f.write('  "version": "1.0.0",\n')
        f.write('  "timestamp": "2025-12-14T07:00:00Z",\n')
        f.write('  "status": "PASS",\n')
        f.write('  "statistical_metrics": {\n')
        f.write('    "speedup": 3.42,\n')
        f.write('    "speedup_ci_lower": 3.28,\n')
        f.write('    "speedup_ci_upper": 3.56,\n')
        f.write('    "displacement_error": 0.015,\n')
        f.write('    "stress_error": 0.023,\n')
        f.write('    "energy_error": 1.2e-7,\n')
        f.write('    "coefficient_of_variation": 0.012\n')
        f.write("  },\n")
        f.write('  "reproducibility": {\n')
        f.write('    "deterministic": true,\n')
        f.write('    "hash_match": "100%",\n')
        f.write('    "temporal_drift": "0.8μs"\n')
        f.write("  }\n")
        f.write("}\n")
        f.write("```\n\n")

        f.write("## HTML Report Structure\n\n")
        f.write("```html\n")
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("  <title>BM_001 Benchmark Report</title>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("  <h1>BM_001: Large-Strain Rubber Block Compression</h1>\n")
        f.write("  \n")
        f.write("  <h2>Executive Summary</h2>\n")
        f.write("  <p>Status: <strong>PASS</strong></p>\n")
        f.write("  <p>Speedup: <strong>3.42x</strong> (95% CI: [3.28, 3.56])</p>\n")
        f.write("  \n")
        f.write("  <h2>Performance Metrics</h2>\n")
        f.write("  <table>\n")
        f.write("    <tr><td>Ansys Time</td><td>120.5s</td></tr>\n")
        f.write("    <tr><td>QuASIM Time</td><td>35.2s</td></tr>\n")
        f.write("    <tr><td>Speedup</td><td>3.42x</td></tr>\n")
        f.write("  </table>\n")
        f.write("  \n")
        f.write("  <h2>Visualizations</h2>\n")
        f.write("  <img src='performance_comparison.png' />\n")
        f.write("</body>\n")
        f.write("</html>\n")
        f.write("```\n\n")

        f.write("## PDF Generation\n\n")
        f.write("PDF reports use ReportLab library:\n\n")
        f.write("```python\n")
        f.write("from reportlab.lib.pagesizes import letter\n")
        f.write("from reportlab.pdfgen import canvas\n")
        f.write("\n")
        f.write("def generate_pdf_report(results, output_path):\n")
        f.write('    """Generate PDF benchmark report."""\n')
        f.write("    c = canvas.Canvas(output_path, pagesize=letter)\n")
        f.write("    \n")
        f.write("    # Title\n")
        f.write("    c.setFont('Helvetica-Bold', 24)\n")
        f.write("    c.drawString(100, 750, 'BM_001 Benchmark Report')\n")
        f.write("    \n")
        f.write("    # Metrics\n")
        f.write("    c.setFont('Helvetica', 12)\n")
        f.write("    c.drawString(100, 700, f'Speedup: {results.speedup:.2f}x')\n")
        f.write("    c.drawString(100, 680, f'Displacement Error: {results.disp_error:.2%}')\n")
        f.write("    \n")
        f.write("    c.save()\n")
        f.write("```\n\n")

    logger.info(f"Generated reporting examples: {output_path}")
    return output_path


def generate_all_appendices(output_dir: Path) -> list[Path]:
    """Generate all appendices.

    Args:
        output_dir: Output directory for appendices

    Returns:
        List of generated file paths
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    appendices = [
        generate_yaml_benchmark_spec(output_dir),
        generate_cuda_pseudocode(output_dir),
        generate_statistical_derivations(output_dir),
        generate_reproducibility_proof(output_dir),
        generate_reporting_examples(output_dir),
    ]

    logger.info(f"Generated {len(appendices)} appendices")
    return appendices
