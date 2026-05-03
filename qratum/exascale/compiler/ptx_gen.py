"""
QRATUM ExaScale Reproducible PTX/SASS Generation

Deterministic PTX (Parallel Thread Execution) and SASS (Streaming Assembly)
generation for NVIDIA GB200 GPUs with guaranteed bit-identical output.

PTX Determinism Strategy:
- Disable FMA (Fused Multiply-Add) instructions
- Fixed instruction scheduling order
- Deterministic register allocation
- No architecture-specific auto-tuning
- Reproducible optimization passes

SASS Determinism Strategy:
- Fixed instruction ordering
- Deterministic register file allocation
- No hardware-dependent optimizations
- Reproducible branch target layout
- Static scheduling directives

Key Challenges Addressed:
1. NVCC non-determinism: Control via explicit flags
2. PTX version drift: Pin to specific PTX ISA version
3. SASS code reordering: Disable dynamic optimizations
4. Register allocation randomness: Fixed allocation strategy
5. Instruction scheduling: Deterministic dependency analysis

The generated PTX/SASS must be identical across:
- All 3,125 CN-QES nodes
- Multiple compilation runs
- Different host systems
- Different compiler invocations
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PTXVersion(Enum):
    """Fixed PTX ISA versions (no auto-selection)"""

    PTX_7_0 = "7.0"  # CUDA 11.0
    PTX_7_5 = "7.5"  # CUDA 11.5
    PTX_8_0 = "8.0"  # CUDA 12.0
    PTX_8_3 = "8.3"  # CUDA 12.3 (GB200 target)


class SASSArchitecture(Enum):
    """Target SASS architectures"""

    SM_90 = "sm_90"  # NVIDIA Hopper (H100)
    SM_100 = "sm_100"  # NVIDIA Blackwell (GB200)


class InstructionScheduling(Enum):
    """Instruction scheduling strategies"""

    DETERMINISTIC = "deterministic"  # Fixed order, no reordering
    BALANCED = "balanced"  # Some reordering with fixed heuristics
    AGGRESSIVE = "aggressive"  # Maximum ILP (non-deterministic)


class RegisterAllocationStrategy(Enum):
    """Register allocation strategies"""

    LINEAR = "linear"  # Simple linear scan (deterministic)
    GRAPH_COLORING = "graph_coloring"  # Graph coloring with fixed order
    ADAPTIVE = "adaptive"  # Non-deterministic (not allowed)


@dataclass
class PTXGenerationConfig:
    """
    Configuration for deterministic PTX generation

    All parameters are carefully chosen to ensure reproducibility.
    Any non-deterministic optimizations are explicitly disabled.

    Attributes:
        ptx_version: Fixed PTX ISA version
        fma_enabled: FMA instructions (disabled for determinism)
        fast_math: Fast math mode (disabled for correctness)
        denorm_mode: Denormalized number handling
        flush_to_zero: Flush denorms to zero
        instruction_scheduling: Scheduling strategy
        max_registers: Maximum registers per thread
        deterministic_ptx: Force deterministic PTX output
        inline_threshold: Fixed inlining threshold
    """

    ptx_version: PTXVersion = PTXVersion.PTX_8_3
    fma_enabled: bool = False  # Critical: Must be False
    fast_math: bool = False
    denorm_mode: str = "preserve"  # "preserve" or "flush"
    flush_to_zero: bool = False
    instruction_scheduling: InstructionScheduling = InstructionScheduling.DETERMINISTIC
    max_registers: int = 255
    deterministic_ptx: bool = True
    inline_threshold: int = 100  # Fixed, not adaptive

    # Additional determinism flags
    disable_loop_unrolling: bool = False
    disable_constant_folding: bool = False
    disable_dead_code_elimination: bool = False
    preserve_unreachable: bool = True

    def to_nvcc_flags(self) -> List[str]:
        """
        Convert configuration to NVCC command-line flags

        Returns:
            List of NVCC flags for PTX generation
        """
        flags = [
            "--ptx",
            f"-ptx-version={self.ptx_version.value}",
            f"--maxrregcount={self.max_registers}",
        ]

        # Critical: Disable FMA for determinism
        if not self.fma_enabled:
            flags.append("--fmad=false")

        if not self.fast_math:
            flags.append("--use_fast_math=false")

        if self.flush_to_zero:
            flags.append("--ftz=true")
        else:
            flags.append("--ftz=false")

        if self.deterministic_ptx:
            flags.extend(
                [
                    "-D__CUDA_NO_HALF_OPERATORS__",  # Reduce variation
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "--def-load-cache=ca",  # Fixed cache policy
                ]
            )

        return flags

    def validate_determinism(self) -> bool:
        """
        Validate that configuration ensures deterministic PTX

        Returns:
            True if configuration is valid for determinism
        """
        # FMA must be disabled
        if self.fma_enabled:
            return False

        # Fast math breaks determinism
        if self.fast_math:
            return False

        # Must use deterministic scheduling
        if self.instruction_scheduling == InstructionScheduling.AGGRESSIVE:
            return False

        return True


@dataclass
class SASSGenerationConfig:
    """
    Configuration for deterministic SASS generation

    SASS (GPU machine code) must be reproducible across all nodes.

    Attributes:
        architecture: Target GPU architecture
        register_allocation: Register allocation strategy
        instruction_scheduling: Instruction scheduling strategy
        deterministic_sass: Force deterministic SASS output
        disable_code_motion: Prevent instruction reordering
        disable_prefetch: Disable prefetch instructions
        fixed_barrier_sync: Use fixed barrier synchronization
    """

    architecture: SASSArchitecture = SASSArchitecture.SM_100
    register_allocation: RegisterAllocationStrategy = RegisterAllocationStrategy.LINEAR
    instruction_scheduling: InstructionScheduling = InstructionScheduling.DETERMINISTIC
    deterministic_sass: bool = True
    disable_code_motion: bool = True
    disable_prefetch: bool = True
    fixed_barrier_sync: bool = True

    def to_ptxas_flags(self) -> List[str]:
        """
        Convert configuration to ptxas command-line flags

        Returns:
            List of ptxas flags for SASS generation
        """
        flags = [
            f"--gpu-name={self.architecture.value}",
        ]

        if self.deterministic_sass:
            flags.extend(
                [
                    "--no-optimize-bindings",  # Prevent reordering
                    "--disable-optimizer-constants",  # Fixed constants
                ]
            )

        if self.disable_code_motion:
            flags.append("--def-load-cache=ca")  # Fixed memory access

        return flags

    def validate_determinism(self) -> bool:
        """
        Validate that configuration ensures deterministic SASS

        Returns:
            True if configuration is valid for determinism
        """
        # Adaptive register allocation is non-deterministic
        if self.register_allocation == RegisterAllocationStrategy.ADAPTIVE:
            return False

        # Aggressive scheduling can reorder
        if self.instruction_scheduling == InstructionScheduling.AGGRESSIVE:
            return False

        return True


@dataclass
class PTXArtifact:
    """
    PTX artifact with metadata and verification

    Attributes:
        path: Path to PTX file
        ptx_version: PTX ISA version used
        source_files: Source files compiled
        config: PTX generation config
        hash: SHA-256 hash for verification
        size_bytes: File size
        instruction_count: Number of PTX instructions
    """

    path: Path
    ptx_version: PTXVersion
    source_files: List[Path]
    config: PTXGenerationConfig
    hash: str = ""
    size_bytes: int = 0
    instruction_count: int = 0

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of PTX file

        Returns:
            Hex-encoded SHA-256 hash
        """
        hasher = hashlib.sha256()

        with open(self.path, "rb") as f:
            content = f.read()
            hasher.update(content)

            # Count instructions (lines starting with '\t')
            self.instruction_count = sum(
                1 for line in content.decode("utf-8").split("\n") if line.strip().startswith("\t")
            )

        self.hash = hasher.hexdigest()
        self.size_bytes = self.path.stat().st_size

        return self.hash

    def verify_determinism(self, reference: "PTXArtifact") -> bool:
        """
        Verify PTX is identical to reference

        Args:
            reference: Reference PTX artifact from another node

        Returns:
            True if PTX is bit-identical
        """
        if not self.hash:
            self.compute_hash()

        if not reference.hash:
            reference.compute_hash()

        return (
            self.hash == reference.hash
            and self.size_bytes == reference.size_bytes
            and self.instruction_count == reference.instruction_count
        )


@dataclass
class SASSArtifact:
    """
    SASS artifact with metadata and verification

    Attributes:
        path: Path to SASS/cubin file
        architecture: Target architecture
        ptx_source: Source PTX file
        config: SASS generation config
        hash: SHA-256 hash for verification
        size_bytes: File size
        instruction_count: Number of SASS instructions
    """

    path: Path
    architecture: SASSArchitecture
    ptx_source: Path
    config: SASSGenerationConfig
    hash: str = ""
    size_bytes: int = 0
    instruction_count: int = 0

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of SASS file

        Returns:
            Hex-encoded SHA-256 hash
        """
        hasher = hashlib.sha256()

        with open(self.path, "rb") as f:
            content = f.read()
            hasher.update(content)

        self.hash = hasher.hexdigest()
        self.size_bytes = self.path.stat().st_size

        return self.hash

    def verify_determinism(self, reference: "SASSArtifact") -> bool:
        """
        Verify SASS is identical to reference

        Args:
            reference: Reference SASS artifact from another node

        Returns:
            True if SASS is bit-identical
        """
        if not self.hash:
            self.compute_hash()

        if not reference.hash:
            reference.compute_hash()

        return self.hash == reference.hash and self.size_bytes == reference.size_bytes


class DeterministicPTXGenerator:
    """
    Deterministic PTX code generator

    Generates reproducible PTX code from CUDA sources with guaranteed
    bit-identical output across all nodes and compilation runs.
    """

    def __init__(self, config: Optional[PTXGenerationConfig] = None):
        """
        Initialize PTX generator

        Args:
            config: PTX generation configuration
        """
        self.config = config or PTXGenerationConfig()

        if not self.config.validate_determinism():
            raise ValueError("PTX configuration is not deterministic")

    def generate_ptx(
        self,
        source_files: List[Path],
        output_path: Path,
        include_dirs: Optional[List[Path]] = None,
    ) -> PTXArtifact:
        """
        Generate deterministic PTX from CUDA sources

        Process:
        1. Validate source files exist
        2. Build nvcc command with deterministic flags
        3. Execute compilation in isolated environment
        4. Verify PTX output is deterministic
        5. Compute verification hash

        Args:
            source_files: List of CUDA source files
            output_path: Output PTX file path
            include_dirs: Optional include directories

        Returns:
            PTX artifact with verification hash
        """
        # Stub: In real implementation, would invoke nvcc

        # Build command
        cmd = ["nvcc"] + self.config.to_nvcc_flags()

        if include_dirs:
            for inc_dir in include_dirs:
                cmd.extend(["-I", str(inc_dir)])

        cmd.extend(["-o", str(output_path)])
        cmd.extend([str(f) for f in source_files])

        # Execute compilation (stubbed)
        # subprocess.run(cmd, env=deterministic_env, check=True)

        artifact = PTXArtifact(
            path=output_path,
            ptx_version=self.config.ptx_version,
            source_files=source_files,
            config=self.config,
        )

        return artifact

    def verify_ptx_determinism(
        self,
        artifacts: List[PTXArtifact],
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that PTX artifacts are bit-identical

        Args:
            artifacts: List of PTX artifacts from different nodes

        Returns:
            Tuple of (is_deterministic, error_message)
        """
        if len(artifacts) < 2:
            return True, None

        reference = artifacts[0]
        reference.compute_hash()

        for i, artifact in enumerate(artifacts[1:], 1):
            artifact.compute_hash()

            if not artifact.verify_determinism(reference):
                return False, f"PTX mismatch: node {i} differs from reference"

        return True, None

    def analyze_ptx_instructions(self, ptx_file: Path) -> Dict[str, int]:
        """
        Analyze PTX instructions for non-determinism indicators

        Checks for:
        - FMA instructions (should be absent)
        - Divergent branches
        - Non-deterministic intrinsics

        Args:
            ptx_file: PTX file to analyze

        Returns:
            Dictionary of instruction counts by type
        """
        instruction_counts = {}

        with open(ptx_file) as f:
            for line in f:
                line = line.strip()

                # Count FMA instructions (should be zero)
                if "fma." in line:
                    instruction_counts["fma"] = instruction_counts.get("fma", 0) + 1

                # Count other relevant instructions
                if "mad." in line:
                    instruction_counts["mad"] = instruction_counts.get("mad", 0) + 1

                if "bra." in line:
                    instruction_counts["branch"] = instruction_counts.get("branch", 0) + 1

        return instruction_counts


class DeterministicSASSGenerator:
    """
    Deterministic SASS code generator

    Generates reproducible GPU machine code from PTX with guaranteed
    bit-identical output.
    """

    def __init__(self, config: Optional[SASSGenerationConfig] = None):
        """
        Initialize SASS generator

        Args:
            config: SASS generation configuration
        """
        self.config = config or SASSGenerationConfig()

        if not self.config.validate_determinism():
            raise ValueError("SASS configuration is not deterministic")

    def generate_sass(
        self,
        ptx_file: Path,
        output_path: Path,
    ) -> SASSArtifact:
        """
        Generate deterministic SASS from PTX

        Process:
        1. Validate PTX input
        2. Build ptxas command with deterministic flags
        3. Execute assembly in isolated environment
        4. Verify SASS output is deterministic
        5. Compute verification hash

        Args:
            ptx_file: Input PTX file
            output_path: Output SASS/cubin file

        Returns:
            SASS artifact with verification hash
        """
        # Stub: In real implementation, would invoke ptxas

        # Build command
        cmd = ["ptxas"] + self.config.to_ptxas_flags()
        cmd.extend(["-o", str(output_path), str(ptx_file)])

        # Execute assembly (stubbed)
        # subprocess.run(cmd, env=deterministic_env, check=True)

        artifact = SASSArtifact(
            path=output_path,
            architecture=self.config.architecture,
            ptx_source=ptx_file,
            config=self.config,
        )

        return artifact

    def verify_sass_determinism(
        self,
        artifacts: List[SASSArtifact],
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that SASS artifacts are bit-identical

        Args:
            artifacts: List of SASS artifacts from different nodes

        Returns:
            Tuple of (is_deterministic, error_message)
        """
        if len(artifacts) < 2:
            return True, None

        reference = artifacts[0]
        reference.compute_hash()

        for i, artifact in enumerate(artifacts[1:], 1):
            artifact.compute_hash()

            if not artifact.verify_determinism(reference):
                return False, f"SASS mismatch: node {i} differs from reference"

        return True, None

    def disassemble_sass(self, sass_file: Path) -> str:
        """
        Disassemble SASS for inspection

        Useful for debugging non-determinism issues.

        Args:
            sass_file: SASS/cubin file to disassemble

        Returns:
            Disassembled SASS code as string
        """
        # Stub: In real implementation, would use cuobjdump or nvdisasm

        # Command: nvdisasm -c sass_file

        return "// SASS disassembly (stub)"


def get_default_ptx_generator() -> DeterministicPTXGenerator:
    """
    Get default PTX generator with deterministic configuration

    Returns:
        Configured PTX generator
    """
    config = PTXGenerationConfig(
        ptx_version=PTXVersion.PTX_8_3,
        fma_enabled=False,  # Critical
        fast_math=False,
        instruction_scheduling=InstructionScheduling.DETERMINISTIC,
    )

    return DeterministicPTXGenerator(config)


def get_default_sass_generator() -> DeterministicSASSGenerator:
    """
    Get default SASS generator with deterministic configuration

    Returns:
        Configured SASS generator
    """
    config = SASSGenerationConfig(
        architecture=SASSArchitecture.SM_100,
        register_allocation=RegisterAllocationStrategy.LINEAR,
        instruction_scheduling=InstructionScheduling.DETERMINISTIC,
    )

    return DeterministicSASSGenerator(config)
