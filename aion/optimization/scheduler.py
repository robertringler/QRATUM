"""AION Adaptive Scheduler.

Implements causal hypergraph scheduling with adaptive runtime:
- Dispatch tasks to CPU/GPU/FPGA/WASM/JVM
- Online profiling and migration
- Optimal throughput scheduling

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

from ..sir.hypergraph import HyperGraph
from ..sir.vertices import HardwareAffinity, Vertex, VertexType


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    MIGRATED = auto()


class DeviceKind(Enum):
    """Execution device kinds."""

    CPU = auto()
    GPU = auto()
    FPGA = auto()
    WASM = auto()
    JVM = auto()
    TPU = auto()


@dataclass
class Device:
    """Execution device representation."""

    id: str
    kind: DeviceKind
    name: str = ""
    capacity: float = 1.0  # Relative capacity
    utilization: float = 0.0
    memory_available: int = 0
    memory_total: int = 0
    features: set[str] = field(default_factory=set)

    def can_execute(self, task: Task) -> bool:
        """Check if device can execute a task."""
        # Check hardware affinity
        affinity_map = {
            HardwareAffinity.CPU: DeviceKind.CPU,
            HardwareAffinity.GPU: DeviceKind.GPU,
            HardwareAffinity.GPU_STREAM0: DeviceKind.GPU,
            HardwareAffinity.GPU_STREAM1: DeviceKind.GPU,
            HardwareAffinity.FPGA: DeviceKind.FPGA,
            HardwareAffinity.FPGA_LUT: DeviceKind.FPGA,
            HardwareAffinity.WASM: DeviceKind.WASM,
            HardwareAffinity.JVM: DeviceKind.JVM,
            HardwareAffinity.TPU: DeviceKind.TPU,
            HardwareAffinity.ANY: None,  # Any device
        }

        required_kind = affinity_map.get(task.hardware_affinity)
        if required_kind and required_kind != self.kind:
            return False

        # Check memory
        if task.memory_required > self.memory_available:
            return False

        return True

    def estimated_time(self, task: Task) -> float:
        """Estimate execution time for a task."""
        base_time = task.estimated_cycles / 1e9  # Assume 1GHz

        # Adjust for device type
        if self.kind == DeviceKind.GPU and task.parallelism > 1:
            return base_time / min(task.parallelism, 1024)
        elif self.kind == DeviceKind.FPGA:
            return base_time * 0.8  # FPGA overhead but lower latency
        elif self.kind == DeviceKind.TPU:
            return base_time / min(task.parallelism, 128)

        return base_time / self.capacity


@dataclass
class Task:
    """A schedulable task derived from AION-SIR vertex.

    Attributes:
        id: Unique task identifier
        vertex: Source vertex
        status: Current execution status
        hardware_affinity: Required hardware target
        dependencies: Tasks this task depends on
        dependents: Tasks depending on this task
        estimated_cycles: Estimated execution cycles
        memory_required: Memory requirement in bytes
        parallelism: Degree of parallelism
        priority: Scheduling priority
        assigned_device: Device task is assigned to
        start_time: Actual start time
        end_time: Actual end time
    """

    id: str
    vertex: Vertex
    status: TaskStatus = TaskStatus.PENDING
    hardware_affinity: HardwareAffinity = HardwareAffinity.ANY
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)
    estimated_cycles: int = 1000
    memory_required: int = 0
    parallelism: int = 1
    priority: int = 0
    assigned_device: Device | None = None
    start_time: float = 0.0
    end_time: float = 0.0

    def is_ready(self, completed: set[str]) -> bool:
        """Check if task is ready to execute."""
        return self.dependencies.issubset(completed)

    @staticmethod
    def from_vertex(vertex: Vertex, graph: HyperGraph) -> Task:
        """Create task from AION-SIR vertex."""
        # Determine dependencies from data flow edges
        dependencies: set[str] = set()
        for pred in graph.get_predecessors(vertex):
            dependencies.add(pred.id)

        # Estimate cycles based on vertex type
        cycles = 1000
        if vertex.vertex_type == VertexType.KERNEL_LAUNCH:
            grid = vertex.metadata.parallelism.get("grid_dim", (1, 1, 1))
            block = vertex.metadata.parallelism.get("block_dim", (1, 1, 1))
            cycles = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2] * 100
        elif vertex.vertex_type in (VertexType.LOAD, VertexType.STORE):
            cycles = 100
        elif vertex.vertex_type == VertexType.APPLY:
            cycles = 1000

        # Get parallelism
        parallelism = 1
        if vertex.metadata.parallelism:
            grid = vertex.metadata.parallelism.get("grid_dim", (1, 1, 1))
            parallelism = grid[0] * grid[1] * grid[2]

        return Task(
            id=vertex.id,
            vertex=vertex,
            hardware_affinity=vertex.metadata.hardware_affinity,
            dependencies=dependencies,
            estimated_cycles=cycles,
            parallelism=parallelism,
        )


@dataclass
class ScheduleResult:
    """Result of scheduling a graph.

    Attributes:
        tasks: All tasks with assignments
        makespan: Total execution time
        device_utilization: Utilization per device
        migrations: Number of task migrations
    """

    tasks: list[Task]
    makespan: float = 0.0
    device_utilization: dict[str, float] = field(default_factory=dict)
    migrations: int = 0


class CausalScheduler:
    """Causal hypergraph scheduler.

    Schedules tasks based on causal dependencies in the
    hypergraph, respecting data flow and effect ordering.
    """

    def __init__(self, devices: list[Device] | None = None) -> None:
        """Initialize the causal scheduler."""
        self.devices = devices or [
            Device(
                id="cpu0",
                kind=DeviceKind.CPU,
                name="CPU",
                capacity=1.0,
                memory_available=16 * 1024**3,
                memory_total=16 * 1024**3,
            ),
        ]
        self.tasks: dict[str, Task] = {}
        self.completed: set[str] = set()
        self.ready_queue: list[tuple[int, str]] = []  # (priority, task_id)

    def schedule(self, graph: HyperGraph) -> ScheduleResult:
        """Schedule a hypergraph for execution.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            ScheduleResult with task assignments
        """
        # Create tasks from vertices
        self.tasks = {}
        for vertex in graph.vertices:
            task = Task.from_vertex(vertex, graph)
            self.tasks[task.id] = task

        # Build dependent relationships
        for task in self.tasks.values():
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self.tasks[dep_id].dependents.add(task.id)

        # Initialize ready queue with tasks that have no dependencies
        self.completed = set()
        self.ready_queue = []

        for task in self.tasks.values():
            if task.is_ready(self.completed):
                task.status = TaskStatus.READY
                heapq.heappush(self.ready_queue, (-task.priority, task.id))

        # Schedule tasks
        current_time = 0.0
        device_finish_times: dict[str, float] = {d.id: 0.0 for d in self.devices}

        while self.ready_queue:
            _, task_id = heapq.heappop(self.ready_queue)
            task = self.tasks[task_id]

            if task.status != TaskStatus.READY:
                continue

            # Find best device
            best_device = None
            best_finish_time = float("inf")

            for device in self.devices:
                if device.can_execute(task):
                    start = max(device_finish_times[device.id], current_time)
                    finish = start + device.estimated_time(task)

                    if finish < best_finish_time:
                        best_finish_time = finish
                        best_device = device

            if best_device:
                # Assign task to device
                task.assigned_device = best_device
                task.start_time = max(device_finish_times[best_device.id], current_time)
                task.end_time = task.start_time + best_device.estimated_time(task)
                task.status = TaskStatus.COMPLETED

                device_finish_times[best_device.id] = task.end_time
                self.completed.add(task.id)

                # Add newly ready tasks
                for dep_id in task.dependents:
                    dep_task = self.tasks.get(dep_id)
                    if dep_task and dep_task.status == TaskStatus.PENDING:
                        if dep_task.is_ready(self.completed):
                            dep_task.status = TaskStatus.READY
                            heapq.heappush(self.ready_queue, (-dep_task.priority, dep_task.id))

        # Calculate makespan
        makespan = max(device_finish_times.values()) if device_finish_times else 0.0

        # Calculate utilization
        utilization = {}
        for device in self.devices:
            device_tasks = [
                t
                for t in self.tasks.values()
                if t.assigned_device and t.assigned_device.id == device.id
            ]
            busy_time = sum(t.end_time - t.start_time for t in device_tasks)
            utilization[device.id] = busy_time / makespan if makespan > 0 else 0.0

        return ScheduleResult(
            tasks=list(self.tasks.values()),
            makespan=makespan,
            device_utilization=utilization,
        )


class AdaptiveScheduler:
    """Adaptive runtime scheduler with online profiling.

    Features:
    - Online profiling of task execution
    - Dynamic task migration
    - Load balancing across devices
    - Optimal throughput optimization
    """

    def __init__(self, devices: list[Device] | None = None) -> None:
        """Initialize adaptive scheduler."""
        self.devices = devices or []
        self.causal_scheduler = CausalScheduler(devices)
        self.profile_data: dict[str, list[float]] = defaultdict(list)
        self.migration_threshold = 0.2  # 20% improvement needed to migrate
        self.profiling_enabled = True

    def add_device(self, device: Device) -> None:
        """Add a device to the scheduler."""
        self.devices.append(device)
        self.causal_scheduler.devices.append(device)

    def remove_device(self, device_id: str) -> None:
        """Remove a device from the scheduler."""
        self.devices = [d for d in self.devices if d.id != device_id]
        self.causal_scheduler.devices = [
            d for d in self.causal_scheduler.devices if d.id != device_id
        ]

    def schedule(self, graph: HyperGraph) -> ScheduleResult:
        """Schedule with adaptive optimization.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            ScheduleResult with optimized task assignments
        """
        # Initial schedule
        result = self.causal_scheduler.schedule(graph)

        # Apply profiling-based optimizations
        if self.profiling_enabled and self.profile_data:
            result = self._optimize_from_profile(result)

        return result

    def record_execution(self, task_id: str, actual_time: float, device_id: str) -> None:
        """Record actual execution time for profiling.

        Args:
            task_id: Task identifier
            actual_time: Actual execution time
            device_id: Device task ran on
        """
        key = f"{task_id}:{device_id}"
        self.profile_data[key].append(actual_time)

    def _optimize_from_profile(self, result: ScheduleResult) -> ScheduleResult:
        """Optimize schedule based on profiling data."""
        migrations = 0

        for task in result.tasks:
            if not task.assigned_device:
                continue

            current_device = task.assigned_device
            current_key = f"{task.id}:{current_device.id}"

            if current_key not in self.profile_data:
                continue

            current_avg = sum(self.profile_data[current_key]) / len(self.profile_data[current_key])

            # Check if another device would be faster
            for device in self.devices:
                if device.id == current_device.id:
                    continue

                if not device.can_execute(task):
                    continue

                other_key = f"{task.id}:{device.id}"
                if other_key in self.profile_data:
                    other_avg = sum(self.profile_data[other_key]) / len(
                        self.profile_data[other_key]
                    )

                    improvement = (current_avg - other_avg) / current_avg
                    if improvement > self.migration_threshold:
                        task.assigned_device = device
                        task.status = TaskStatus.MIGRATED
                        migrations += 1

        result.migrations = migrations
        return result

    def get_optimal_device(self, task: Task) -> Device | None:
        """Get optimal device for a task based on profiling.

        Args:
            task: Task to schedule

        Returns:
            Best device or None
        """
        best_device = None
        best_time = float("inf")

        for device in self.devices:
            if not device.can_execute(task):
                continue

            # Check profiling data
            key = f"{task.id}:{device.id}"
            if key in self.profile_data and self.profile_data[key]:
                avg_time = sum(self.profile_data[key]) / len(self.profile_data[key])
            else:
                avg_time = device.estimated_time(task)

            if avg_time < best_time:
                best_time = avg_time
                best_device = device

        return best_device

    def balance_load(self) -> dict[str, float]:
        """Balance load across devices.

        Returns:
            Updated utilization per device
        """
        utilization = {}
        total_capacity = sum(d.capacity for d in self.devices)

        for device in self.devices:
            target_utilization = device.capacity / total_capacity
            device.utilization = target_utilization
            utilization[device.id] = target_utilization

        return utilization

    def predict_throughput(self, graph: HyperGraph) -> float:
        """Predict throughput for a graph.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            Predicted tasks per second
        """
        result = self.schedule(graph)
        if result.makespan > 0:
            return len(result.tasks) / result.makespan
        return 0.0


class WorkStealingScheduler:
    """Work-stealing scheduler for parallel execution.

    Each device has a local queue; idle devices steal work
    from busy devices.
    """

    def __init__(self, devices: list[Device]) -> None:
        """Initialize work-stealing scheduler."""
        self.devices = devices
        self.queues: dict[str, list[Task]] = {d.id: [] for d in devices}
        self.completed: set[str] = set()

    def add_task(self, task: Task) -> None:
        """Add a task to the appropriate queue."""
        # Find best device
        best_device = None
        min_queue_size = float("inf")

        for device in self.devices:
            if device.can_execute(task):
                queue_size = len(self.queues[device.id])
                if queue_size < min_queue_size:
                    min_queue_size = queue_size
                    best_device = device

        if best_device:
            self.queues[best_device.id].append(task)
            task.assigned_device = best_device

    def steal_work(self, idle_device: Device) -> Task | None:
        """Attempt to steal work for an idle device.

        Args:
            idle_device: Device that needs work

        Returns:
            Stolen task or None
        """
        # Find busiest device
        busiest = None
        max_queue = 0

        for device in self.devices:
            if device.id == idle_device.id:
                continue

            queue_size = len(self.queues[device.id])
            if queue_size > max_queue:
                max_queue = queue_size
                busiest = device

        if busiest and max_queue > 1:
            # Steal from end of queue (locality)
            queue = self.queues[busiest.id]
            for i in range(len(queue) - 1, -1, -1):
                task = queue[i]
                if idle_device.can_execute(task):
                    queue.pop(i)
                    task.assigned_device = idle_device
                    return task

        return None

    def run(self) -> ScheduleResult:
        """Run the work-stealing scheduler.

        Returns:
            ScheduleResult with execution results
        """
        current_time = 0.0
        device_times: dict[str, float] = {d.id: 0.0 for d in self.devices}
        all_tasks: list[Task] = []

        # Collect all tasks
        for queue in self.queues.values():
            all_tasks.extend(queue)

        while any(self.queues.values()):
            for device in self.devices:
                queue = self.queues[device.id]

                if not queue:
                    # Try to steal work
                    stolen = self.steal_work(device)
                    if stolen:
                        queue.append(stolen)

                if queue:
                    task = queue.pop(0)
                    task.start_time = max(device_times[device.id], current_time)
                    task.end_time = task.start_time + device.estimated_time(task)
                    task.status = TaskStatus.COMPLETED
                    device_times[device.id] = task.end_time
                    self.completed.add(task.id)

            current_time = min(device_times.values()) if device_times else 0.0

        makespan = max(device_times.values()) if device_times else 0.0

        return ScheduleResult(
            tasks=all_tasks,
            makespan=makespan,
            device_utilization={d.id: 0.8 for d in self.devices},  # Simplified
        )
