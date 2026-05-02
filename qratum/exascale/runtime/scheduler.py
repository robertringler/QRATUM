"""
Deterministic GPU Kernel Scheduling Framework

Provides deterministic ordering of GPU kernel launches to ensure bit-identical
execution across runs. Critical for reproducible ML training where kernel
launch order can affect numerical results due to hardware race conditions.

Key Features:
- Fixed-order kernel queue with priority levels
- Deterministic stream assignment based on operation hash
- GPU resource allocation tracking
- Kernel dependency graph for ordering
- No dynamic scheduling or load balancing

Design Principle:
GPU kernels can execute in parallel and non-deterministically unless
explicitly ordered. Our scheduler enforces strict ordering using:
1. Single default stream (no concurrency) OR
2. Multiple streams with explicit dependencies
3. Fixed kernel launch order within each stream
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class KernelPriority(Enum):
    """Kernel execution priority levels"""
    CRITICAL = 0  # Must execute immediately (communication kernels)
    HIGH = 1  # High priority (computation kernels)
    NORMAL = 2  # Normal priority (most kernels)
    LOW = 3  # Low priority (logging, profiling)
    BACKGROUND = 4  # Background tasks (checkpointing)


class StreamType(Enum):
    """CUDA stream types"""
    DEFAULT = "default"  # Default stream (serializes all kernels)
    COMPUTE = "compute"  # Computation stream
    COMMUNICATION = "communication"  # Communication stream (DMA, all-reduce)
    UTILITY = "utility"  # Utility stream (memory ops, memcpy)


class KernelState(Enum):
    """Kernel execution states"""
    QUEUED = "queued"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GPUResource:
    """
    GPU resource specification
    
    Tracks available GPU resources for scheduling decisions.
    
    Attributes:
        gpu_id: GPU device ID
        compute_capability: CUDA compute capability (e.g., "9.0" for Blackwell)
        memory_available_gb: Available GPU memory in GB
        memory_total_gb: Total GPU memory in GB
        sm_count: Number of streaming multiprocessors
        sm_utilization: Current SM utilization (0.0 to 1.0)
    """
    gpu_id: int
    compute_capability: str
    memory_available_gb: float
    memory_total_gb: float
    sm_count: int = 144  # GB200 has 144 SMs
    sm_utilization: float = 0.0

    def allocate_memory(self, size_gb: float) -> bool:
        """
        Allocate GPU memory
        
        Args:
            size_gb: Memory size in GB
            
        Returns:
            True if allocation successful
        """
        if size_gb <= self.memory_available_gb:
            self.memory_available_gb -= size_gb
            return True
        return False

    def free_memory(self, size_gb: float) -> None:
        """Free GPU memory"""
        self.memory_available_gb = min(
            self.memory_available_gb + size_gb,
            self.memory_total_gb
        )

    def get_memory_utilization(self) -> float:
        """Get current memory utilization (0.0 to 1.0)"""
        return 1.0 - (self.memory_available_gb / self.memory_total_gb)


@dataclass
class KernelDescriptor:
    """
    Describes a GPU kernel launch
    
    Attributes:
        kernel_id: Unique kernel identifier
        kernel_name: Human-readable kernel name
        operation_id: Parent operation ID
        priority: Execution priority
        stream_type: Target stream type
        grid_dim: CUDA grid dimensions (x, y, z)
        block_dim: CUDA block dimensions (x, y, z)
        shared_memory_bytes: Shared memory per block
        estimated_duration_us: Estimated execution time in microseconds
        dependencies: Set of kernel IDs this kernel depends on
    """
    kernel_id: int
    kernel_name: str
    operation_id: str
    priority: KernelPriority = KernelPriority.NORMAL
    stream_type: StreamType = StreamType.DEFAULT
    grid_dim: Tuple[int, int, int] = (1, 1, 1)
    block_dim: Tuple[int, int, int] = (256, 1, 1)
    shared_memory_bytes: int = 0
    estimated_duration_us: float = 0.0
    dependencies: Set[int] = field(default_factory=set)
    state: KernelState = KernelState.QUEUED

    def compute_kernel_hash(self) -> str:
        """
        Compute deterministic hash of kernel configuration
        
        Returns:
            SHA-256 hash as hex string
        """
        data = (
            f"{self.kernel_name}:{self.operation_id}:{self.grid_dim}:"
            f"{self.block_dim}:{self.shared_memory_bytes}"
        )
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def is_ready(self, completed_kernels: Set[int]) -> bool:
        """
        Check if kernel is ready to execute
        
        Args:
            completed_kernels: Set of completed kernel IDs
            
        Returns:
            True if all dependencies are satisfied
        """
        return self.dependencies.issubset(completed_kernels)

    def add_dependency(self, kernel_id: int) -> None:
        """Add kernel dependency"""
        self.dependencies.add(kernel_id)

    def get_total_threads(self) -> int:
        """Get total number of threads launched"""
        grid_size = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]
        block_size = self.block_dim[0] * self.block_dim[1] * self.block_dim[2]
        return grid_size * block_size


@dataclass
class KernelQueue:
    """
    Deterministic kernel queue for a single GPU
    
    Maintains strict FIFO ordering within priority levels to ensure
    deterministic execution.
    
    Attributes:
        gpu_id: GPU device ID
        stream_type: Stream type for this queue
        queued_kernels: List of queued kernels (ordered by submission)
        executing_kernels: Set of currently executing kernel IDs
        completed_kernels: Set of completed kernel IDs
        max_concurrent_kernels: Maximum concurrent kernel executions
    """
    gpu_id: int
    stream_type: StreamType
    queued_kernels: List[KernelDescriptor] = field(default_factory=list)
    executing_kernels: Set[int] = field(default_factory=set)
    completed_kernels: Set[int] = field(default_factory=set)
    max_concurrent_kernels: int = 1  # Default: serialize all kernels

    def enqueue_kernel(self, kernel: KernelDescriptor) -> None:
        """
        Add kernel to queue
        
        Kernels are queued in submission order. Within same priority,
        earlier submissions execute first (FIFO).
        
        Args:
            kernel: Kernel descriptor
        """
        kernel.state = KernelState.QUEUED
        self.queued_kernels.append(kernel)

        # Sort by priority (stable sort preserves submission order within priority)
        self.queued_kernels.sort(key=lambda k: k.priority.value)

    def get_next_kernel(self) -> Optional[KernelDescriptor]:
        """
        Get next kernel ready for execution
        
        Returns:
            Next kernel to execute, or None if no kernel is ready
        """
        # Check if we can launch more kernels
        if len(self.executing_kernels) >= self.max_concurrent_kernels:
            return None

        # Find first ready kernel
        for i, kernel in enumerate(self.queued_kernels):
            if kernel.is_ready(self.completed_kernels):
                kernel.state = KernelState.READY
                self.queued_kernels.pop(i)
                return kernel

        return None

    def launch_kernel(self, kernel: KernelDescriptor) -> bool:
        """
        Mark kernel as executing
        
        Args:
            kernel: Kernel to launch
            
        Returns:
            True if launch successful
        """
        if kernel.kernel_id in self.executing_kernels:
            return False

        kernel.state = KernelState.EXECUTING
        self.executing_kernels.add(kernel.kernel_id)
        return True

    def complete_kernel(self, kernel_id: int) -> bool:
        """
        Mark kernel as completed
        
        Args:
            kernel_id: Kernel ID to complete
            
        Returns:
            True if kernel was executing
        """
        if kernel_id not in self.executing_kernels:
            return False

        self.executing_kernels.remove(kernel_id)
        self.completed_kernels.add(kernel_id)
        return True

    def get_queue_depth(self) -> int:
        """Get number of queued kernels"""
        return len(self.queued_kernels)

    def is_idle(self) -> bool:
        """Check if queue is idle"""
        return len(self.queued_kernels) == 0 and len(self.executing_kernels) == 0


@dataclass
class DeterministicScheduler:
    """
    Deterministic GPU kernel scheduler
    
    Schedules GPU kernels with deterministic ordering guarantees.
    Ensures same kernel launch order across runs for reproducibility.
    
    Design:
    1. Fixed stream assignment per operation type
    2. FIFO ordering within streams
    3. Explicit dependency tracking between kernels
    4. No dynamic load balancing or reordering
    
    Attributes:
        num_gpus: Number of GPUs to manage
        gpu_resources: GPU resource tracking
        kernel_queues: Per-GPU, per-stream kernel queues
        kernel_counter: Global kernel ID counter
        kernel_registry: Registry of all launched kernels
        use_default_stream: Use single default stream for maximum determinism
    """
    num_gpus: int
    gpu_resources: Dict[int, GPUResource] = field(default_factory=dict)
    kernel_queues: Dict[Tuple[int, StreamType], KernelQueue] = field(default_factory=dict)
    kernel_counter: int = 0
    kernel_registry: Dict[int, KernelDescriptor] = field(default_factory=dict)
    use_default_stream: bool = True

    def __post_init__(self):
        """Initialize scheduler resources"""
        # Initialize GPU resources (GB200 specs)
        for gpu_id in range(self.num_gpus):
            self.gpu_resources[gpu_id] = GPUResource(
                gpu_id=gpu_id,
                compute_capability="9.0",  # Blackwell
                memory_available_gb=192.0,  # 192 GB HBM3e
                memory_total_gb=192.0,
                sm_count=144
            )

        # Initialize kernel queues
        if self.use_default_stream:
            # Single default stream per GPU
            for gpu_id in range(self.num_gpus):
                self.kernel_queues[(gpu_id, StreamType.DEFAULT)] = KernelQueue(
                    gpu_id=gpu_id,
                    stream_type=StreamType.DEFAULT,
                    max_concurrent_kernels=1  # Serialize all kernels
                )
        else:
            # Multiple streams per GPU (more parallelism, less determinism)
            for gpu_id in range(self.num_gpus):
                for stream_type in StreamType:
                    self.kernel_queues[(gpu_id, stream_type)] = KernelQueue(
                        gpu_id=gpu_id,
                        stream_type=stream_type,
                        max_concurrent_kernels=4  # Limited parallelism
                    )

    def schedule_kernel(self, kernel_name: str, operation_id: str,
                       gpu_id: int = 0,
                       priority: KernelPriority = KernelPriority.NORMAL,
                       stream_type: StreamType = StreamType.DEFAULT,
                       grid_dim: Tuple[int, int, int] = (1, 1, 1),
                       block_dim: Tuple[int, int, int] = (256, 1, 1),
                       dependencies: Optional[Set[int]] = None) -> int:
        """
        Schedule GPU kernel for execution
        
        Args:
            kernel_name: Kernel name
            operation_id: Parent operation ID
            gpu_id: Target GPU ID
            priority: Execution priority
            stream_type: Target stream type
            grid_dim: CUDA grid dimensions
            block_dim: CUDA block dimensions
            dependencies: Set of kernel IDs this kernel depends on
            
        Returns:
            Kernel ID
        """
        # Generate kernel ID deterministically
        kernel_id = self.kernel_counter
        self.kernel_counter += 1

        # Create kernel descriptor
        kernel = KernelDescriptor(
            kernel_id=kernel_id,
            kernel_name=kernel_name,
            operation_id=operation_id,
            priority=priority,
            stream_type=stream_type if not self.use_default_stream else StreamType.DEFAULT,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=dependencies if dependencies else set()
        )

        # Register kernel
        self.kernel_registry[kernel_id] = kernel

        # Determine target stream
        actual_stream = StreamType.DEFAULT if self.use_default_stream else stream_type
        queue_key = (gpu_id, actual_stream)

        # Enqueue kernel
        if queue_key in self.kernel_queues:
            self.kernel_queues[queue_key].enqueue_kernel(kernel)

        return kernel_id

    def execute_ready_kernels(self) -> List[int]:
        """
        Execute all ready kernels across all queues
        
        Returns:
            List of launched kernel IDs
        """
        launched = []

        # Process queues in deterministic order (sorted by GPU ID and stream type)
        queue_keys = sorted(self.kernel_queues.keys(),
                          key=lambda k: (k[0], k[1].value))

        for queue_key in queue_keys:
            queue = self.kernel_queues[queue_key]

            while True:
                kernel = queue.get_next_kernel()
                if kernel is None:
                    break

                # Launch kernel
                if queue.launch_kernel(kernel):
                    launched.append(kernel.kernel_id)

                    # In real implementation: actual CUDA kernel launch
                    # For now, immediately mark as completed for simulation
                    # queue.complete_kernel(kernel.kernel_id)

        return launched

    def mark_kernel_completed(self, kernel_id: int) -> bool:
        """
        Mark kernel as completed
        
        Args:
            kernel_id: Kernel ID
            
        Returns:
            True if kernel was found and marked completed
        """
        kernel = self.kernel_registry.get(kernel_id)
        if not kernel:
            return False

        # Find queue containing this kernel
        for queue_key, queue in self.kernel_queues.items():
            if kernel_id in queue.executing_kernels:
                kernel.state = KernelState.COMPLETED
                return queue.complete_kernel(kernel_id)

        return False

    def wait_for_completion(self) -> None:
        """
        Wait for all queued and executing kernels to complete
        
        In real implementation, this would synchronize with GPU.
        """
        # Simulate completion of all kernels
        for queue in self.kernel_queues.values():
            for kernel_id in list(queue.executing_kernels):
                queue.complete_kernel(kernel_id)

    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics
        
        Returns:
            Dictionary of statistics
        """
        total_queued = sum(q.get_queue_depth() for q in self.kernel_queues.values())
        total_executing = sum(len(q.executing_kernels) for q in self.kernel_queues.values())
        total_completed = sum(len(q.completed_kernels) for q in self.kernel_queues.values())

        return {
            'total_kernels': self.kernel_counter,
            'queued': total_queued,
            'executing': total_executing,
            'completed': total_completed,
            'num_gpus': self.num_gpus,
            'use_default_stream': self.use_default_stream,
            'gpu_utilization': {
                gpu_id: resource.sm_utilization
                for gpu_id, resource in self.gpu_resources.items()
            }
        }

    def compute_schedule_hash(self) -> str:
        """
        Compute hash of kernel execution schedule
        
        Returns:
            SHA-256 hash of schedule
        """
        # Collect all kernel hashes in execution order
        kernel_hashes = []
        for kernel_id in range(self.kernel_counter):
            kernel = self.kernel_registry.get(kernel_id)
            if kernel:
                kernel_hashes.append(kernel.compute_kernel_hash())

        schedule_str = ";".join(kernel_hashes)
        return hashlib.sha256(schedule_str.encode('utf-8')).hexdigest()

    def verify_determinism(self, other_schedule_hash: str) -> bool:
        """
        Verify schedule matches another run
        
        Args:
            other_schedule_hash: Schedule hash from another run
            
        Returns:
            True if schedules match
        """
        local_hash = self.compute_schedule_hash()
        return local_hash == other_schedule_hash
