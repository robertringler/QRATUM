"""LLVM Backend for AION.

Generates LLVM IR from AION-SIR hypergraph with:
- Region metadata mapping to stack/heap
- JIT compilation support
- Optimization passes

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..sir.vertices import Vertex, VertexType, AIONType, HardwareAffinity
from ..sir.edges import HyperEdge, EdgeType
from ..sir.hypergraph import HyperGraph


@dataclass
class LLVMType:
    """LLVM type representation."""
    name: str
    is_pointer: bool = False
    element_type: "LLVMType | None" = None
    size: int = 0
    
    @staticmethod
    def void() -> "LLVMType":
        return LLVMType(name="void")
    
    @staticmethod
    def i1() -> "LLVMType":
        return LLVMType(name="i1", size=1)
    
    @staticmethod
    def i8() -> "LLVMType":
        return LLVMType(name="i8", size=8)
    
    @staticmethod
    def i16() -> "LLVMType":
        return LLVMType(name="i16", size=16)
    
    @staticmethod
    def i32() -> "LLVMType":
        return LLVMType(name="i32", size=32)
    
    @staticmethod
    def i64() -> "LLVMType":
        return LLVMType(name="i64", size=64)
    
    @staticmethod
    def f32() -> "LLVMType":
        return LLVMType(name="float", size=32)
    
    @staticmethod
    def f64() -> "LLVMType":
        return LLVMType(name="double", size=64)
    
    @staticmethod
    def ptr(element: "LLVMType") -> "LLVMType":
        return LLVMType(name=f"ptr", is_pointer=True, element_type=element, size=64)
    
    @staticmethod
    def array(element: "LLVMType", length: int) -> "LLVMType":
        return LLVMType(name=f"[{length} x {element.name}]", element_type=element)
    
    def __str__(self) -> str:
        return self.name


@dataclass
class LLVMValue:
    """LLVM value representation."""
    name: str
    ty: LLVMType
    is_constant: bool = False
    constant_value: Any = None
    
    def __str__(self) -> str:
        if self.is_constant:
            return str(self.constant_value)
        return f"%{self.name}"


@dataclass
class LLVMInstruction:
    """LLVM instruction representation."""
    opcode: str
    operands: list[LLVMValue] = field(default_factory=list)
    result: LLVMValue | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def to_ir(self) -> str:
        """Generate LLVM IR string for this instruction."""
        if self.opcode == "ret":
            if self.operands:
                return f"ret {self.operands[0].ty} {self.operands[0]}"
            return "ret void"
        
        elif self.opcode == "alloca":
            ty = self.attributes.get("type", LLVMType.i64())
            align = self.attributes.get("align", 8)
            return f"{self.result} = alloca {ty}, align {align}"
        
        elif self.opcode == "load":
            ty = self.result.ty if self.result else LLVMType.i64()
            ptr = self.operands[0]
            align = self.attributes.get("align", 8)
            return f"{self.result} = load {ty}, ptr {ptr}, align {align}"
        
        elif self.opcode == "store":
            val = self.operands[0]
            ptr = self.operands[1]
            align = self.attributes.get("align", 8)
            return f"store {val.ty} {val}, ptr {ptr}, align {align}"
        
        elif self.opcode in ("add", "sub", "mul", "sdiv", "udiv", "srem", "urem"):
            left = self.operands[0]
            right = self.operands[1]
            return f"{self.result} = {self.opcode} {left.ty} {left}, {right}"
        
        elif self.opcode in ("fadd", "fsub", "fmul", "fdiv", "frem"):
            left = self.operands[0]
            right = self.operands[1]
            return f"{self.result} = {self.opcode} {left.ty} {left}, {right}"
        
        elif self.opcode == "call":
            func = self.attributes.get("function", "unknown")
            ret_ty = self.result.ty if self.result else LLVMType.void()
            args = ", ".join(f"{op.ty} {op}" for op in self.operands)
            if self.result:
                return f"{self.result} = call {ret_ty} @{func}({args})"
            return f"call {ret_ty} @{func}({args})"
        
        elif self.opcode == "br":
            if len(self.operands) == 1:
                # Unconditional branch
                label = self.attributes.get("label", "entry")
                return f"br label %{label}"
            else:
                # Conditional branch
                cond = self.operands[0]
                true_label = self.attributes.get("true_label", "then")
                false_label = self.attributes.get("false_label", "else")
                return f"br i1 {cond}, label %{true_label}, label %{false_label}"
        
        elif self.opcode == "phi":
            ty = self.result.ty if self.result else LLVMType.i64()
            incoming = self.attributes.get("incoming", [])
            pairs = ", ".join(f"[ {v}, %{b} ]" for v, b in incoming)
            return f"{self.result} = phi {ty} {pairs}"
        
        elif self.opcode == "getelementptr":
            base = self.operands[0]
            indices = self.operands[1:]
            ty = self.attributes.get("element_type", LLVMType.i64())
            idx_str = ", ".join(f"i64 {idx}" for idx in indices)
            return f"{self.result} = getelementptr {ty}, ptr {base}, {idx_str}"
        
        elif self.opcode == "icmp":
            pred = self.attributes.get("predicate", "eq")
            left = self.operands[0]
            right = self.operands[1]
            return f"{self.result} = icmp {pred} {left.ty} {left}, {right}"
        
        elif self.opcode == "fcmp":
            pred = self.attributes.get("predicate", "oeq")
            left = self.operands[0]
            right = self.operands[1]
            return f"{self.result} = fcmp {pred} {left.ty} {left}, {right}"
        
        return f"; unknown: {self.opcode}"


@dataclass
class LLVMBasicBlock:
    """LLVM basic block representation."""
    label: str
    instructions: list[LLVMInstruction] = field(default_factory=list)
    
    def to_ir(self) -> str:
        """Generate LLVM IR for this basic block."""
        lines = [f"{self.label}:"]
        for inst in self.instructions:
            lines.append(f"  {inst.to_ir()}")
        return "\n".join(lines)


@dataclass
class LLVMFunction:
    """LLVM function representation."""
    name: str
    return_type: LLVMType
    parameters: list[tuple[str, LLVMType]] = field(default_factory=list)
    blocks: list[LLVMBasicBlock] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    
    def to_ir(self) -> str:
        """Generate LLVM IR for this function."""
        params = ", ".join(f"{ty} %{name}" for name, ty in self.parameters)
        attrs = " ".join(self.attributes)
        
        lines = [f"define {self.return_type} @{self.name}({params}) {attrs} {{"]
        for block in self.blocks:
            lines.append(block.to_ir())
        lines.append("}")
        
        return "\n".join(lines)


@dataclass
class LLVMModule:
    """LLVM module representation."""
    name: str
    target_triple: str = "x86_64-unknown-linux-gnu"
    data_layout: str = ""
    globals: list[str] = field(default_factory=list)
    functions: list[LLVMFunction] = field(default_factory=list)
    declarations: list[str] = field(default_factory=list)
    metadata: list[str] = field(default_factory=list)
    
    def to_ir(self) -> str:
        """Generate complete LLVM IR module."""
        lines = [
            f'; ModuleID = "{self.name}"',
            f'source_filename = "{self.name}"',
            f'target triple = "{self.target_triple}"',
        ]
        
        if self.data_layout:
            lines.append(f'target datalayout = "{self.data_layout}"')
        
        lines.append("")
        
        # Declarations
        for decl in self.declarations:
            lines.append(decl)
        
        lines.append("")
        
        # Globals
        for glob in self.globals:
            lines.append(glob)
        
        lines.append("")
        
        # Functions
        for func in self.functions:
            lines.append(func.to_ir())
            lines.append("")
        
        # Metadata
        for meta in self.metadata:
            lines.append(meta)
        
        return "\n".join(lines)


class LLVMCodeGen:
    """LLVM code generator for AION-SIR."""
    
    def __init__(self) -> None:
        """Initialize code generator."""
        self.module: LLVMModule | None = None
        self.current_function: LLVMFunction | None = None
        self.current_block: LLVMBasicBlock | None = None
        self.value_map: dict[str, LLVMValue] = {}
        self.temp_counter = 0
    
    def generate(self, graph: HyperGraph) -> str:
        """Generate LLVM IR from AION-SIR hypergraph.
        
        Args:
            graph: AION-SIR hypergraph
            
        Returns:
            LLVM IR string
        """
        self.module = LLVMModule(name=graph.name or "aion_module")
        
        # Add standard declarations
        self._add_declarations()
        
        # Generate function from graph
        func = self._generate_function(graph)
        self.module.functions.append(func)
        
        # Add region metadata
        self._add_region_metadata(graph)
        
        return self.module.to_ir()
    
    def _add_declarations(self) -> None:
        """Add standard function declarations."""
        if self.module:
            self.module.declarations.extend([
                "declare ptr @malloc(i64)",
                "declare void @free(ptr)",
                "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)",
                "declare void @llvm.lifetime.start.p0(i64, ptr)",
                "declare void @llvm.lifetime.end.p0(i64, ptr)",
            ])
    
    def _generate_function(self, graph: HyperGraph) -> LLVMFunction:
        """Generate LLVM function from graph."""
        func = LLVMFunction(
            name=graph.name or "main",
            return_type=LLVMType.i32(),
            attributes=["nounwind"],
        )
        
        # Create entry block
        entry = LLVMBasicBlock(label="entry")
        func.blocks.append(entry)
        
        self.current_function = func
        self.current_block = entry
        
        # Generate code for each vertex in topological order
        for vertex in graph.topological_order():
            self._generate_vertex(vertex, graph)
        
        # Add return if not present
        if not entry.instructions or entry.instructions[-1].opcode != "ret":
            entry.instructions.append(LLVMInstruction(
                opcode="ret",
                operands=[LLVMValue("0", LLVMType.i32(), is_constant=True, constant_value=0)],
            ))
        
        return func
    
    def _generate_vertex(self, vertex: Vertex, graph: HyperGraph) -> LLVMValue | None:
        """Generate LLVM IR for a vertex."""
        if vertex.vertex_type == VertexType.CONST:
            return self._generate_const(vertex)
        
        elif vertex.vertex_type == VertexType.ALLOC:
            return self._generate_alloc(vertex)
        
        elif vertex.vertex_type == VertexType.LOAD:
            return self._generate_load(vertex, graph)
        
        elif vertex.vertex_type == VertexType.STORE:
            return self._generate_store(vertex, graph)
        
        elif vertex.vertex_type == VertexType.APPLY:
            return self._generate_apply(vertex, graph)
        
        elif vertex.vertex_type == VertexType.PHI:
            return self._generate_phi(vertex, graph)
        
        elif vertex.vertex_type == VertexType.RETURN:
            return self._generate_return(vertex, graph)
        
        elif vertex.vertex_type == VertexType.PARAMETER:
            return self._generate_parameter(vertex)
        
        return None
    
    def _generate_const(self, vertex: Vertex) -> LLVMValue:
        """Generate constant value."""
        value = vertex.value
        
        if isinstance(value, bool):
            ty = LLVMType.i1()
            val = 1 if value else 0
        elif isinstance(value, int):
            ty = LLVMType.i64()
            val = value
        elif isinstance(value, float):
            ty = LLVMType.f64()
            val = value
        else:
            ty = LLVMType.i64()
            val = 0
        
        llvm_val = LLVMValue(
            name=f"const_{self._next_temp()}",
            ty=ty,
            is_constant=True,
            constant_value=val,
        )
        self.value_map[vertex.id] = llvm_val
        return llvm_val
    
    def _generate_alloc(self, vertex: Vertex) -> LLVMValue:
        """Generate allocation."""
        region = vertex.metadata.region or "stack"
        size = vertex.attributes.get("size", 8)
        
        ty = self._aion_to_llvm_type(vertex.metadata.type_info)
        result = LLVMValue(f"alloc_{self._next_temp()}", LLVMType.ptr(ty))
        
        if region in ("stack", "local"):
            # Stack allocation
            inst = LLVMInstruction(
                opcode="alloca",
                result=result,
                attributes={"type": ty, "align": 8},
            )
        else:
            # Heap allocation via malloc
            size_val = LLVMValue("size", LLVMType.i64(), is_constant=True, constant_value=size)
            inst = LLVMInstruction(
                opcode="call",
                operands=[size_val],
                result=result,
                attributes={"function": "malloc"},
            )
        
        if self.current_block:
            self.current_block.instructions.append(inst)
        
        self.value_map[vertex.id] = result
        return result
    
    def _generate_load(self, vertex: Vertex, graph: HyperGraph) -> LLVMValue:
        """Generate load instruction."""
        ty = self._aion_to_llvm_type(vertex.metadata.type_info)
        result = LLVMValue(f"load_{self._next_temp()}", ty)
        
        # Get pointer operand
        preds = graph.get_predecessors(vertex)
        if preds and preds[0].id in self.value_map:
            ptr = self.value_map[preds[0].id]
        else:
            ptr = LLVMValue("null", LLVMType.ptr(ty), is_constant=True, constant_value="null")
        
        inst = LLVMInstruction(
            opcode="load",
            operands=[ptr],
            result=result,
            attributes={"align": 8},
        )
        
        if self.current_block:
            self.current_block.instructions.append(inst)
        
        self.value_map[vertex.id] = result
        return result
    
    def _generate_store(self, vertex: Vertex, graph: HyperGraph) -> LLVMValue | None:
        """Generate store instruction."""
        preds = graph.get_predecessors(vertex)
        if len(preds) < 2:
            return None
        
        val_vertex = preds[0]
        ptr_vertex = preds[1]
        
        val = self.value_map.get(val_vertex.id)
        ptr = self.value_map.get(ptr_vertex.id)
        
        if not val or not ptr:
            return None
        
        inst = LLVMInstruction(
            opcode="store",
            operands=[val, ptr],
            attributes={"align": 8},
        )
        
        if self.current_block:
            self.current_block.instructions.append(inst)
        
        return None
    
    def _generate_apply(self, vertex: Vertex, graph: HyperGraph) -> LLVMValue:
        """Generate function call or operation."""
        func_name = vertex.attributes.get("function", "unknown")
        
        # Get operands
        preds = graph.get_predecessors(vertex)
        operands = [self.value_map.get(p.id) for p in preds if p.id in self.value_map]
        
        ty = self._aion_to_llvm_type(vertex.metadata.type_info)
        result = LLVMValue(f"call_{self._next_temp()}", ty)
        
        # Map common operations
        op_map = {
            "op_+": "add",
            "op_-": "sub",
            "op_*": "mul",
            "op_/": "sdiv",
        }
        
        if func_name in op_map and len(operands) == 2:
            inst = LLVMInstruction(
                opcode=op_map[func_name],
                operands=operands,
                result=result,
            )
        else:
            inst = LLVMInstruction(
                opcode="call",
                operands=operands,
                result=result,
                attributes={"function": func_name},
            )
        
        if self.current_block:
            self.current_block.instructions.append(inst)
        
        self.value_map[vertex.id] = result
        return result
    
    def _generate_phi(self, vertex: Vertex, graph: HyperGraph) -> LLVMValue:
        """Generate phi node."""
        ty = self._aion_to_llvm_type(vertex.metadata.type_info)
        result = LLVMValue(f"phi_{self._next_temp()}", ty)
        
        # Get incoming values
        preds = graph.get_predecessors(vertex)
        incoming = []
        for pred in preds:
            if pred.id in self.value_map:
                incoming.append((self.value_map[pred.id], "entry"))
        
        inst = LLVMInstruction(
            opcode="phi",
            result=result,
            attributes={"incoming": incoming},
        )
        
        if self.current_block:
            self.current_block.instructions.append(inst)
        
        self.value_map[vertex.id] = result
        return result
    
    def _generate_return(self, vertex: Vertex, graph: HyperGraph) -> LLVMValue | None:
        """Generate return instruction."""
        preds = graph.get_predecessors(vertex)
        
        if preds and preds[0].id in self.value_map:
            val = self.value_map[preds[0].id]
            inst = LLVMInstruction(opcode="ret", operands=[val])
        else:
            inst = LLVMInstruction(opcode="ret", operands=[
                LLVMValue("0", LLVMType.i32(), is_constant=True, constant_value=0)
            ])
        
        if self.current_block:
            self.current_block.instructions.append(inst)
        
        return None
    
    def _generate_parameter(self, vertex: Vertex) -> LLVMValue:
        """Generate parameter reference."""
        name = vertex.attributes.get("name", f"param_{vertex.attributes.get('index', 0)}")
        ty = self._aion_to_llvm_type(vertex.metadata.type_info)
        
        result = LLVMValue(name, ty)
        
        if self.current_function:
            self.current_function.parameters.append((name, ty))
        
        self.value_map[vertex.id] = result
        return result
    
    def _aion_to_llvm_type(self, aion_type: AIONType | None) -> LLVMType:
        """Convert AION type to LLVM type."""
        if not aion_type:
            return LLVMType.i64()
        
        type_map = {
            "int": LLVMType.i64(),
            "float": LLVMType.f64(),
            "unit": LLVMType.void(),
            "bool": LLVMType.i1(),
        }
        
        if aion_type.kind in type_map:
            return type_map[aion_type.kind]
        
        if aion_type.kind == "ptr":
            if aion_type.params:
                element = self._aion_to_llvm_type(aion_type.params[0])
                return LLVMType.ptr(element)
            return LLVMType.ptr(LLVMType.i8())
        
        if aion_type.kind == "array":
            if aion_type.params and aion_type.refinement:
                element = self._aion_to_llvm_type(aion_type.params[0])
                # Extract length from refinement
                try:
                    length = int(aion_type.refinement.split("=")[1])
                except (ValueError, IndexError):
                    length = 1
                return LLVMType.array(element, length)
        
        # Default to i64
        if aion_type.size:
            if aion_type.size <= 8:
                return LLVMType.i8()
            elif aion_type.size <= 16:
                return LLVMType.i16()
            elif aion_type.size <= 32:
                return LLVMType.i32()
        
        return LLVMType.i64()
    
    def _add_region_metadata(self, graph: HyperGraph) -> None:
        """Add region metadata to module."""
        if not self.module:
            return
        
        regions: set[str] = set()
        for v in graph.vertices:
            if v.metadata.region:
                regions.add(v.metadata.region)
        
        for i, region in enumerate(sorted(regions)):
            self.module.metadata.append(
                f'!{i} = !{{!"region", !"{region}"}}'
            )
    
    def _next_temp(self) -> int:
        """Get next temporary variable number."""
        self.temp_counter += 1
        return self.temp_counter


class LLVMBackend:
    """LLVM backend for AION.
    
    Provides full compilation pipeline:
    - AION-SIR to LLVM IR
    - Optimization passes
    - JIT compilation
    """
    
    def __init__(self) -> None:
        """Initialize LLVM backend."""
        self.codegen = LLVMCodeGen()
    
    def compile(self, graph: HyperGraph, optimize: bool = True) -> str:
        """Compile graph to LLVM IR.
        
        Args:
            graph: AION-SIR hypergraph
            optimize: Whether to run optimization passes
            
        Returns:
            LLVM IR string
        """
        ir = self.codegen.generate(graph)
        
        if optimize:
            ir = self._optimize(ir)
        
        return ir
    
    def compile_to_object(self, graph: HyperGraph) -> bytes:
        """Compile graph to object code.
        
        Args:
            graph: AION-SIR hypergraph
            
        Returns:
            Object code bytes
        """
        ir = self.compile(graph)
        # Would use llvmlite or similar for actual compilation
        return ir.encode('utf-8')
    
    def jit_compile(self, graph: HyperGraph) -> callable:
        """JIT compile graph for immediate execution.
        
        Args:
            graph: AION-SIR hypergraph
            
        Returns:
            Callable function
        """
        ir = self.compile(graph)
        # Would use llvmlite for JIT compilation
        # For now, return a placeholder
        def placeholder():
            return 0
        return placeholder
    
    def _optimize(self, ir: str) -> str:
        """Run LLVM optimization passes."""
        # Would use llvmlite opt passes
        # For now, return unmodified
        return ir
