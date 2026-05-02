"""
Quantum-Classical Hybrid Workload Scheduler

Orchestrates workload distribution between PTAQ-1000 quantum processors
and GB200 GPU clusters, optimizing for:
- Quantum advantage detection
- Classical preprocessing/postprocessing
- Workload decomposition
- Resource utilization
- End-to-end latency minimization

Key Concepts:
- Hybrid algorithms: Quantum subroutines within classical loops
- Variational algorithms: VQE, QAOA, quantum machine learning
- Classical simulation fallback for small circuits
- Adaptive scheduling based on quantum availability

Architecture:
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Classical  │────▶│   Hybrid     │────▶│   Quantum    │
│    GPUs     │     │  Scheduler   │     │  PTAQ-1000   │
│  (GB200)    │◀────│              │◀────│              │
└─────────────┘     └──────────────┘     └──────────────┘
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from qratum.exascale.quantum.ptaq import PTAQ1000, QuantumCircuit, QuantumJob


class WorkloadType(Enum):
    """Hybrid workload types"""
    PURE_CLASSICAL = "pure_classical"  # GPU-only execution
    PURE_QUANTUM = "pure_quantum"  # Quantum-only execution
    HYBRID_VQE = "hybrid_vqe"  # Variational Quantum Eigensolver
    HYBRID_QAOA = "hybrid_qaoa"  # Quantum Approximate Optimization
    HYBRID_QML = "hybrid_qml"  # Quantum Machine Learning
    HYBRID_SAMPLING = "hybrid_sampling"  # Quantum sampling with classical post-processing


class SchedulingPolicy(Enum):
    """Scheduling policies"""
    GREEDY = "greedy"  # Execute immediately on available resource
    COST_OPTIMAL = "cost_optimal"  # Minimize total cost
    LATENCY_OPTIMAL = "latency_optimal"  # Minimize end-to-end latency
    QUANTUM_FIRST = "quantum_first"  # Prefer quantum when available
    CLASSICAL_FIRST = "classical_first"  # Prefer classical simulation
    ADVANTAGE_BASED = "advantage_based"  # Schedule based on quantum advantage


class TaskState(Enum):
    """Task execution states"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    EXECUTING_CLASSICAL = "executing_classical"
    EXECUTING_QUANTUM = "executing_quantum"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ClassicalTask:
    """
    Classical preprocessing/postprocessing task
    
    Attributes:
        task_id: Unique task identifier
        workload_id: Parent workload identifier
        operation: Description of classical operation
        input_size_mb: Input data size in MB
        estimated_flops: Estimated floating-point operations
        estimated_time_ms: Estimated execution time in milliseconds
        gpu_id: Assigned GPU ID (if scheduled)
        requires_quantum_result: Whether task depends on quantum result
    """
    task_id: str
    workload_id: str
    operation: str
    input_size_mb: float
    estimated_flops: float
    estimated_time_ms: float
    gpu_id: Optional[int] = None
    requires_quantum_result: bool = False
    state: TaskState = TaskState.PENDING

    def compute_task_hash(self) -> str:
        """Compute deterministic hash of task"""
        data = f"{self.task_id}:{self.workload_id}:{self.operation}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()


@dataclass
class QuantumTask:
    """
    Quantum computation task
    
    Attributes:
        task_id: Unique task identifier
        workload_id: Parent workload identifier
        circuit: Quantum circuit to execute
        num_shots: Number of circuit repetitions
        estimated_advantage: Estimated quantum advantage factor
        requires_classical_input: Whether task depends on classical preprocessing
        module_ids: Assigned quantum module IDs (if scheduled)
    """
    task_id: str
    workload_id: str
    circuit: QuantumCircuit
    num_shots: int
    estimated_advantage: float
    requires_classical_input: bool = False
    module_ids: Optional[List[int]] = None
    state: TaskState = TaskState.PENDING

    def compute_task_hash(self) -> str:
        """Compute deterministic hash of task"""
        circuit_hash = self.circuit.compute_circuit_hash()
        data = f"{self.task_id}:{self.workload_id}:{circuit_hash}:{self.num_shots}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()


@dataclass
class HybridWorkload:
    """
    Quantum-classical hybrid workload
    
    Represents a complete hybrid computation with classical and quantum
    components, dependencies, and execution constraints.
    
    Attributes:
        workload_id: Unique workload identifier
        workload_type: Type of hybrid workload
        classical_tasks: List of classical tasks
        quantum_tasks: List of quantum tasks
        task_dependencies: Task dependency graph (task_id -> [dependency_ids])
        priority: Workload priority (0 = highest)
        deadline_ms: Deadline in milliseconds (optional)
        status: Current workload status
    """
    workload_id: str
    workload_type: WorkloadType
    classical_tasks: List[ClassicalTask] = field(default_factory=list)
    quantum_tasks: List[QuantumTask] = field(default_factory=list)
    task_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    priority: int = 1
    deadline_ms: Optional[float] = None
    status: str = "queued"

    def add_classical_task(self, task: ClassicalTask) -> None:
        """Add classical task to workload"""
        self.classical_tasks.append(task)
        if task.task_id not in self.task_dependencies:
            self.task_dependencies[task.task_id] = set()

    def add_quantum_task(self, task: QuantumTask) -> None:
        """Add quantum task to workload"""
        self.quantum_tasks.append(task)
        if task.task_id not in self.task_dependencies:
            self.task_dependencies[task.task_id] = set()

    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        """
        Add task dependency
        
        Args:
            task_id: Task that depends on dependency
            dependency_id: Task that must complete first
        """
        if task_id not in self.task_dependencies:
            self.task_dependencies[task_id] = set()
        self.task_dependencies[task_id].add(dependency_id)

    def get_ready_tasks(self, completed_tasks: Set[str]) -> Tuple[List[ClassicalTask], List[QuantumTask]]:
        """
        Get tasks ready for execution
        
        Args:
            completed_tasks: Set of completed task IDs
            
        Returns:
            Tuple of (ready_classical_tasks, ready_quantum_tasks)
        """
        ready_classical = []
        ready_quantum = []

        # Check classical tasks
        for task in self.classical_tasks:
            if task.state == TaskState.PENDING:
                deps = self.task_dependencies.get(task.task_id, set())
                if deps.issubset(completed_tasks):
                    ready_classical.append(task)

        # Check quantum tasks
        for task in self.quantum_tasks:
            if task.state == TaskState.PENDING:
                deps = self.task_dependencies.get(task.task_id, set())
                if deps.issubset(completed_tasks):
                    ready_quantum.append(task)

        return ready_classical, ready_quantum

    def estimate_total_time_ms(self) -> float:
        """
        Estimate total workload execution time
        
        Returns:
            Estimated time in milliseconds
        """
        # Simplified estimate: sum of all task times (ignoring parallelism)
        classical_time = sum(t.estimated_time_ms for t in self.classical_tasks)
        quantum_time = sum(
            t.circuit.get_gate_count() * 50e-6 * t.num_shots  # 50ns per gate
            for t in self.quantum_tasks
        )
        return classical_time + quantum_time

    def get_total_tasks(self) -> int:
        """Get total number of tasks"""
        return len(self.classical_tasks) + len(self.quantum_tasks)


@dataclass
class ResourceAllocation:
    """
    Resource allocation decision
    
    Attributes:
        task_id: Task identifier
        resource_type: "gpu" or "quantum"
        resource_ids: List of allocated resource IDs
        estimated_completion_ms: Estimated completion time
        actual_completion_ms: Actual completion time (when completed)
    """
    task_id: str
    resource_type: str
    resource_ids: List[int]
    estimated_completion_ms: float
    actual_completion_ms: Optional[float] = None


@dataclass
class HybridScheduler:
    """
    Quantum-Classical Hybrid Workload Scheduler
    
    Orchestrates execution across GPU clusters and quantum processors,
    optimizing for quantum advantage while maintaining deterministic behavior.
    
    Design Principles:
    1. Decompose hybrid workloads into classical and quantum tasks
    2. Schedule tasks based on dependencies and resource availability
    3. Route quantum-advantageous work to PTAQ-1000
    4. Execute classical preprocessing/postprocessing on GPUs
    5. Maintain deterministic execution order
    
    Attributes:
        num_gpus: Number of available GPUs
        quantum_processor: PTAQ-1000 quantum processor
        scheduling_policy: Scheduling policy to use
        workload_queue: Queue of pending workloads
        active_workloads: Currently executing workloads
        completed_workloads: Completed workloads
        completed_tasks: Set of completed task IDs
        gpu_allocations: Current GPU allocations
        quantum_allocations: Current quantum allocations
    """
    num_gpus: int
    quantum_processor: PTAQ1000
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.ADVANTAGE_BASED
    workload_queue: List[HybridWorkload] = field(default_factory=list)
    active_workloads: Dict[str, HybridWorkload] = field(default_factory=dict)
    completed_workloads: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    gpu_allocations: Dict[int, Optional[str]] = field(default_factory=dict)
    quantum_allocations: Dict[int, Optional[str]] = field(default_factory=dict)
    allocation_history: List[ResourceAllocation] = field(default_factory=list)

    def __post_init__(self):
        """Initialize scheduler resources"""
        # Initialize GPU allocations (None = idle)
        for gpu_id in range(self.num_gpus):
            self.gpu_allocations[gpu_id] = None

        # Initialize quantum module allocations
        for module in self.quantum_processor.modules:
            self.quantum_allocations[module.module_id] = None

    def submit_workload(self, workload: HybridWorkload) -> str:
        """
        Submit hybrid workload for execution
        
        Args:
            workload: Hybrid workload to submit
            
        Returns:
            Workload ID
        """
        workload.status = "queued"
        self.workload_queue.append(workload)

        # Sort by priority (stable sort preserves submission order)
        self.workload_queue.sort(key=lambda w: w.priority)

        return workload.workload_id

    def should_use_quantum(self, task: QuantumTask) -> bool:
        """
        Decide whether to use quantum or classical simulation
        
        Args:
            task: Quantum task to evaluate
            
        Returns:
            True if quantum execution is advantageous
        """
        if self.scheduling_policy == SchedulingPolicy.QUANTUM_FIRST:
            return True
        elif self.scheduling_policy == SchedulingPolicy.CLASSICAL_FIRST:
            return False
        elif self.scheduling_policy == SchedulingPolicy.ADVANTAGE_BASED:
            # Use quantum if advantage > 1000x
            return task.estimated_advantage > 1000.0
        else:
            # Default: use quantum for circuits that benefit
            circuit_size = task.circuit.num_qubits
            return circuit_size >= 50  # Classical simulation infeasible above 50 qubits

    def allocate_gpu(self, task: ClassicalTask) -> Optional[int]:
        """
        Allocate GPU for classical task
        
        Args:
            task: Classical task to allocate
            
        Returns:
            GPU ID if allocated, None otherwise
        """
        # Find idle GPU
        for gpu_id, allocation in self.gpu_allocations.items():
            if allocation is None:
                self.gpu_allocations[gpu_id] = task.task_id
                task.gpu_id = gpu_id
                return gpu_id

        return None

    def allocate_quantum_modules(self, task: QuantumTask,
                                 num_modules: int = 1) -> Optional[List[int]]:
        """
        Allocate quantum modules for quantum task
        
        Args:
            task: Quantum task to allocate
            num_modules: Number of modules required
            
        Returns:
            List of module IDs if allocated, None otherwise
        """
        # Find idle modules
        idle_modules = [
            module_id for module_id, allocation in self.quantum_allocations.items()
            if allocation is None
        ]

        if len(idle_modules) >= num_modules:
            allocated = idle_modules[:num_modules]
            for module_id in allocated:
                self.quantum_allocations[module_id] = task.task_id
            task.module_ids = allocated
            return allocated

        return None

    def schedule_step(self) -> Dict[str, Any]:
        """
        Execute one scheduling step
        
        Returns:
            Dictionary with scheduling statistics
        """
        scheduled_classical = 0
        scheduled_quantum = 0

        # Process active workloads
        for workload_id, workload in list(self.active_workloads.items()):
            # Get ready tasks
            ready_classical, ready_quantum = workload.get_ready_tasks(self.completed_tasks)

            # Schedule classical tasks
            for task in ready_classical:
                gpu_id = self.allocate_gpu(task)
                if gpu_id is not None:
                    task.state = TaskState.EXECUTING_CLASSICAL
                    scheduled_classical += 1

                    # Record allocation
                    allocation = ResourceAllocation(
                        task_id=task.task_id,
                        resource_type="gpu",
                        resource_ids=[gpu_id],
                        estimated_completion_ms=task.estimated_time_ms,
                    )
                    self.allocation_history.append(allocation)

            # Schedule quantum tasks
            for task in ready_quantum:
                if self.should_use_quantum(task):
                    # Use quantum processor
                    module_ids = self.allocate_quantum_modules(task)
                    if module_ids is not None:
                        task.state = TaskState.EXECUTING_QUANTUM
                        scheduled_quantum += 1

                        # Submit to PTAQ-1000
                        job = QuantumJob(
                            job_id=task.task_id,
                            circuit=task.circuit,
                            target_modules=module_ids,
                            num_shots=task.num_shots,
                        )
                        self.quantum_processor.submit_job(job)

                        # Record allocation
                        allocation = ResourceAllocation(
                            task_id=task.task_id,
                            resource_type="quantum",
                            resource_ids=module_ids,
                            estimated_completion_ms=task.circuit.get_gate_count() * 50e-6 * task.num_shots,
                        )
                        self.allocation_history.append(allocation)
                else:
                    # Use classical simulation on GPU
                    gpu_id = self.allocate_gpu(
                        ClassicalTask(
                            task_id=task.task_id,
                            workload_id=task.workload_id,
                            operation="quantum_simulation",
                            input_size_mb=0.0,
                            estimated_flops=2 ** task.circuit.num_qubits,
                            estimated_time_ms=100.0,
                        )
                    )
                    if gpu_id is not None:
                        task.state = TaskState.EXECUTING_CLASSICAL
                        scheduled_classical += 1

        # Try to activate queued workloads
        if self.workload_queue:
            workload = self.workload_queue.pop(0)
            workload.status = "active"
            self.active_workloads[workload.workload_id] = workload

        return {
            'scheduled_classical': scheduled_classical,
            'scheduled_quantum': scheduled_quantum,
            'active_workloads': len(self.active_workloads),
            'queued_workloads': len(self.workload_queue),
        }

    def complete_task(self, task_id: str) -> bool:
        """
        Mark task as completed
        
        Args:
            task_id: Task ID to complete
            
        Returns:
            True if task was active
        """
        # Release GPU allocation
        for gpu_id, allocation in self.gpu_allocations.items():
            if allocation == task_id:
                self.gpu_allocations[gpu_id] = None
                break

        # Release quantum allocation
        for module_id, allocation in self.quantum_allocations.items():
            if allocation == task_id:
                self.quantum_allocations[module_id] = None

        # Mark task as completed
        self.completed_tasks.add(task_id)

        # Check if workload is complete
        for workload_id, workload in list(self.active_workloads.items()):
            all_tasks = (
                [t.task_id for t in workload.classical_tasks] +
                [t.task_id for t in workload.quantum_tasks]
            )
            if all(tid in self.completed_tasks for tid in all_tasks):
                # Workload complete
                workload.status = "completed"
                del self.active_workloads[workload_id]
                self.completed_workloads.add(workload_id)

        return True

    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics
        
        Returns:
            Dictionary of statistics
        """
        # GPU utilization
        idle_gpus = sum(1 for a in self.gpu_allocations.values() if a is None)
        gpu_utilization = 1.0 - (idle_gpus / self.num_gpus) if self.num_gpus > 0 else 0.0

        # Quantum utilization
        idle_modules = sum(1 for a in self.quantum_allocations.values() if a is None)
        total_modules = len(self.quantum_allocations)
        quantum_utilization = 1.0 - (idle_modules / total_modules) if total_modules > 0 else 0.0

        # Task counts
        total_tasks = sum(w.get_total_tasks() for w in self.active_workloads.values())
        total_tasks += sum(w.get_total_tasks() for w in self.workload_queue)

        return {
            'scheduling_policy': self.scheduling_policy.value,
            'num_gpus': self.num_gpus,
            'num_quantum_modules': total_modules,
            'gpu_utilization': gpu_utilization,
            'quantum_utilization': quantum_utilization,
            'workloads_queued': len(self.workload_queue),
            'workloads_active': len(self.active_workloads),
            'workloads_completed': len(self.completed_workloads),
            'tasks_completed': len(self.completed_tasks),
            'total_pending_tasks': total_tasks,
            'allocations': len(self.allocation_history),
        }


class HybridSchedulerInterface:
    """
    High-level interface to hybrid scheduler
    
    Provides convenient methods for submitting and managing hybrid workloads.
    """

    def __init__(self, num_gpus: int = 16,
                 processor_id: str = "PTAQ-1000-0",
                 policy: SchedulingPolicy = SchedulingPolicy.ADVANTAGE_BASED):
        """
        Initialize hybrid scheduler
        
        Args:
            num_gpus: Number of GPUs available
            processor_id: Quantum processor ID
            policy: Scheduling policy
        """
        from qratum.exascale.quantum.ptaq import PTAQ1000

        self.quantum_processor = PTAQ1000(processor_id=processor_id)
        self.scheduler = HybridScheduler(
            num_gpus=num_gpus,
            quantum_processor=self.quantum_processor,
            scheduling_policy=policy,
        )

    def create_workload(self, workload_id: str,
                       workload_type: WorkloadType) -> HybridWorkload:
        """
        Create new hybrid workload
        
        Args:
            workload_id: Workload identifier
            workload_type: Type of workload
            
        Returns:
            HybridWorkload instance
        """
        return HybridWorkload(workload_id=workload_id, workload_type=workload_type)

    def submit(self, workload: HybridWorkload) -> str:
        """
        Submit workload for execution
        
        Args:
            workload: Hybrid workload
            
        Returns:
            Workload ID
        """
        return self.scheduler.submit_workload(workload)

    def run_scheduling_loop(self, num_steps: int = 100) -> Dict[str, Any]:
        """
        Run scheduling loop for specified steps
        
        Args:
            num_steps: Number of scheduling steps
            
        Returns:
            Final statistics
        """
        for _ in range(num_steps):
            self.scheduler.schedule_step()

            # Simulate task completion (in real system: wait for hardware)
            time.sleep(0.001)

        return self.scheduler.get_scheduler_statistics()

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return self.scheduler.get_scheduler_statistics()
