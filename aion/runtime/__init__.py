"""AION Runtime Module.

Adaptive runtime for executing AION-SIR with:
- Cross-device execution
- Online profiling
- Dynamic optimization

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

from aion.sir.vertices import Vertex, VertexType, HardwareAffinity
from aion.sir.hypergraph import HyperGraph
from aion.optimization.scheduler import AdaptiveScheduler, Device, DeviceKind, Task, ScheduleResult
from aion.proof.verifier import ProofVerifier, ProofTerm


@dataclass
class ExecutionContext:
    """Context for program execution."""
    variables: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, bytes] = field(default_factory=dict)
    stack: list[Any] = field(default_factory=list)
    heap_ptr: int = 0
    profiling: bool = True
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of program execution."""
    success: bool
    value: Any = None
    error: str | None = None
    execution_time: float = 0.0
    memory_used: int = 0
    device_used: str = "cpu"


class AIONRuntime:
    """AION adaptive runtime.
    
    Executes AION-SIR graphs with:
    - Automatic device selection
    - Online profiling
    - Dynamic optimization
    """
    
    def __init__(self) -> None:
        """Initialize the runtime."""
        self.scheduler = AdaptiveScheduler()
        self.verifier = ProofVerifier()
        self.context = ExecutionContext()
        self.proofs: list[ProofTerm] = []
        
        # Add default CPU device
        self.scheduler.add_device(Device(
            id="cpu0",
            kind=DeviceKind.CPU,
            name="Default CPU",
            capacity=1.0,
            memory_available=8 * 1024**3,
            memory_total=8 * 1024**3,
        ))
    
    def add_device(self, device: Device) -> None:
        """Add an execution device."""
        self.scheduler.add_device(device)
    
    def load_proofs(self, proofs: list[ProofTerm]) -> bool:
        """Load and verify proofs.
        
        Args:
            proofs: Proof terms to verify
            
        Returns:
            True if all proofs valid
        """
        self.proofs = proofs
        return all(self.verifier.verify(p) for p in proofs)
    
    def execute(self, graph: HyperGraph) -> ExecutionResult:
        """Execute an AION-SIR graph.
        
        Args:
            graph: AION-SIR hypergraph
            
        Returns:
            ExecutionResult with output
        """
        start_time = time.time()
        
        try:
            # Schedule the graph
            schedule = self.scheduler.schedule(graph)
            
            # Execute tasks in order
            for task in schedule.tasks:
                self._execute_task(task)
            
            # Get result
            if graph.exits:
                result_vertex = graph.exits[0]
                result_value = self.context.variables.get(result_vertex.id)
            else:
                result_value = None
            
            execution_time = time.time() - start_time
            
            # Record profiling
            if self.context.profiling:
                for task in schedule.tasks:
                    if task.assigned_device:
                        self.scheduler.record_execution(
                            task.id,
                            task.end_time - task.start_time,
                            task.assigned_device.id,
                        )
            
            return ExecutionResult(
                success=True,
                value=result_value,
                execution_time=execution_time,
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task."""
        vertex = task.vertex
        
        if vertex.vertex_type == VertexType.CONST:
            self.context.variables[vertex.id] = vertex.value
            return vertex.value
        
        elif vertex.vertex_type == VertexType.ALLOC:
            size = vertex.attributes.get("size", 8)
            addr = self.context.heap_ptr
            self.context.heap_ptr += size
            self.context.variables[vertex.id] = addr
            return addr
        
        elif vertex.vertex_type == VertexType.LOAD:
            # Get address from predecessor
            addr = 0
            for dep_id in task.dependencies:
                if dep_id in self.context.variables:
                    addr = self.context.variables[dep_id]
                    break
            value = self.context.memory.get(str(addr), b'\x00' * 8)
            int_value = int.from_bytes(value[:8], 'little')
            self.context.variables[vertex.id] = int_value
            return int_value
        
        elif vertex.vertex_type == VertexType.STORE:
            # Store value at address
            addr = 0
            value = 0
            deps = list(task.dependencies)
            if len(deps) >= 2:
                value = self.context.variables.get(deps[0], 0)
                addr = self.context.variables.get(deps[1], 0)
            self.context.memory[str(addr)] = int(value).to_bytes(8, 'little')
            return None
        
        elif vertex.vertex_type == VertexType.APPLY:
            func_name = vertex.attributes.get("function", "")
            args = [self.context.variables.get(d) for d in task.dependencies]
            
            # Execute built-in operations
            result = self._execute_builtin(func_name, args)
            self.context.variables[vertex.id] = result
            return result
        
        elif vertex.vertex_type == VertexType.RETURN:
            if task.dependencies:
                dep_id = list(task.dependencies)[0]
                value = self.context.variables.get(dep_id)
                self.context.variables[vertex.id] = value
                return value
            return None
        
        elif vertex.vertex_type == VertexType.PARAMETER:
            # Parameters should be set externally
            return self.context.variables.get(vertex.id, 0)
        
        return None
    
    def _execute_builtin(self, func_name: str, args: list[Any]) -> Any:
        """Execute a built-in function."""
        # Arithmetic operations
        if func_name == "op_+" and len(args) >= 2:
            return (args[0] or 0) + (args[1] or 0)
        elif func_name == "op_-" and len(args) >= 2:
            return (args[0] or 0) - (args[1] or 0)
        elif func_name == "op_*" and len(args) >= 2:
            return (args[0] or 0) * (args[1] or 0)
        elif func_name == "op_/" and len(args) >= 2:
            return (args[0] or 0) // (args[1] or 1)
        
        # Comparison
        elif func_name == "op_<" and len(args) >= 2:
            return 1 if (args[0] or 0) < (args[1] or 0) else 0
        elif func_name == "op_>" and len(args) >= 2:
            return 1 if (args[0] or 0) > (args[1] or 0) else 0
        elif func_name == "op_==" and len(args) >= 2:
            return 1 if args[0] == args[1] else 0
        
        # Default
        return 0
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the execution context."""
        self.context.variables[name] = value
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the execution context."""
        return self.context.variables.get(name)
    
    def reset(self) -> None:
        """Reset the execution context."""
        self.context = ExecutionContext()


class AIONInterpreter:
    """Interpreter for AION source language.
    
    Provides direct interpretation of AION source code
    without compilation.
    """
    
    def __init__(self) -> None:
        """Initialize interpreter."""
        self.runtime = AIONRuntime()
        from aion.language import AIONParser
        self.parser = AIONParser()
    
    def eval(self, source: str) -> Any:
        """Evaluate AION source code.
        
        Args:
            source: AION source code
            
        Returns:
            Evaluation result
        """
        # Parse to AST
        ast = self.parser.parse(source)
        
        # Convert to SIR
        from aion.language import AIONCompiler
        compiler = AIONCompiler()
        graph = compiler.compile_ast(ast)
        
        # Execute
        result = self.runtime.execute(graph)
        
        return result.value if result.success else None
    
    def repl(self) -> None:
        """Start interactive REPL."""
        print("AION Interactive (type 'exit' to quit)")
        
        while True:
            try:
                line = input("aion> ")
                if line.strip() == "exit":
                    break
                
                result = self.eval(line)
                if result is not None:
                    print(f"=> {result}")
                    
            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except Exception as e:
                print(f"Error: {e}")
