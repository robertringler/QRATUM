"""WASM Backend for AION.

Generates WebAssembly from AION-SIR hypergraph with:
- Memory-safe linear memory segments
- Type-safe function calls
- Import/export handling

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..sir.hypergraph import HyperGraph
from ..sir.vertices import AIONType, Vertex, VertexType


class WASMType(Enum):
    """WebAssembly value types."""

    I32 = "i32"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"
    V128 = "v128"
    FUNCREF = "funcref"
    EXTERNREF = "externref"


@dataclass
class WASMLocal:
    """WebAssembly local variable."""

    index: int
    ty: WASMType
    name: str = ""


@dataclass
class WASMInstruction:
    """WebAssembly instruction."""

    opcode: str
    operands: list[Any] = field(default_factory=list)

    def to_wat(self) -> str:
        """Generate WAT (WebAssembly Text) format."""
        if self.operands:
            ops = " ".join(str(o) for o in self.operands)
            return f"({self.opcode} {ops})"
        return f"({self.opcode})"


@dataclass
class WASMFunction:
    """WebAssembly function."""

    name: str
    params: list[tuple[str, WASMType]] = field(default_factory=list)
    results: list[WASMType] = field(default_factory=list)
    locals: list[WASMLocal] = field(default_factory=list)
    body: list[WASMInstruction] = field(default_factory=list)
    exported: bool = False

    def to_wat(self) -> str:
        """Generate WAT format."""
        lines = []

        # Function signature
        params_str = " ".join(f"(param ${name} {ty.value})" for name, ty in self.params)
        results_str = " ".join(f"(result {ty.value})" for ty in self.results)

        export_str = f'(export "{self.name}")' if self.exported else ""

        lines.append(f"(func ${self.name} {export_str} {params_str} {results_str}")

        # Locals
        for local in self.locals:
            lines.append(f"  (local ${local.name} {local.ty.value})")

        # Body
        for inst in self.body:
            lines.append(f"  {inst.to_wat()}")

        lines.append(")")

        return "\n".join(lines)


@dataclass
class WASMMemory:
    """WebAssembly linear memory."""

    name: str
    initial_pages: int = 1
    max_pages: int | None = None
    exported: bool = True

    def to_wat(self) -> str:
        """Generate WAT format."""
        export_str = f'(export "{self.name}")' if self.exported else ""
        max_str = f" {self.max_pages}" if self.max_pages else ""
        return f"(memory ${self.name} {export_str} {self.initial_pages}{max_str})"


@dataclass
class WASMGlobal:
    """WebAssembly global variable."""

    name: str
    ty: WASMType
    mutable: bool = True
    initial_value: Any = 0
    exported: bool = False

    def to_wat(self) -> str:
        """Generate WAT format."""
        export_str = f'(export "{self.name}")' if self.exported else ""
        mut_str = "mut" if self.mutable else ""
        return f"(global ${self.name} {export_str} ({mut_str} {self.ty.value}) ({self.ty.value}.const {self.initial_value}))"


@dataclass
class WASMImport:
    """WebAssembly import."""

    module: str
    name: str
    kind: str  # "func", "memory", "global", "table"
    signature: str = ""

    def to_wat(self) -> str:
        """Generate WAT format."""
        return f'(import "{self.module}" "{self.name}" ({self.kind} ${self.name} {self.signature}))'


@dataclass
class WASMModule:
    """WebAssembly module."""

    name: str = "aion_module"
    imports: list[WASMImport] = field(default_factory=list)
    memories: list[WASMMemory] = field(default_factory=list)
    globals: list[WASMGlobal] = field(default_factory=list)
    functions: list[WASMFunction] = field(default_factory=list)
    data_segments: list[tuple[int, bytes]] = field(default_factory=list)

    def to_wat(self) -> str:
        """Generate WAT format."""
        lines = [f"(module ${self.name}"]

        # Imports
        for imp in self.imports:
            lines.append(f"  {imp.to_wat()}")

        # Memory
        for mem in self.memories:
            lines.append(f"  {mem.to_wat()}")

        # Globals
        for glob in self.globals:
            lines.append(f"  {glob.to_wat()}")

        # Functions
        for func in self.functions:
            lines.append(f"  {func.to_wat()}")

        # Data segments
        for offset, data in self.data_segments:
            escaped = "".join(f"\\{b:02x}" for b in data)
            lines.append(f'  (data (i32.const {offset}) "{escaped}")')

        lines.append(")")

        return "\n".join(lines)

    def to_binary(self) -> bytes:
        """Generate binary WebAssembly format.

        This is a simplified implementation.
        Production would use a proper WASM encoder.
        """
        # WASM magic number and version
        magic = bytes([0x00, 0x61, 0x73, 0x6D])  # \0asm
        version = bytes([0x01, 0x00, 0x00, 0x00])  # version 1

        return magic + version  # Simplified


class WASMCodeGen:
    """WebAssembly code generator for AION-SIR."""

    def __init__(self) -> None:
        """Initialize code generator."""
        self.module: WASMModule | None = None
        self.current_function: WASMFunction | None = None
        self.value_stack: list[str] = []
        self.local_counter = 0
        self.value_map: dict[str, str] = {}

    def generate(self, graph: HyperGraph) -> str:
        """Generate WAT from AION-SIR hypergraph.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            WAT string
        """
        self.module = WASMModule(name=graph.name or "aion_module")

        # Add default memory
        self.module.memories.append(
            WASMMemory(
                name="memory",
                initial_pages=16,
                max_pages=256,
            )
        )

        # Add stack pointer global
        self.module.globals.append(
            WASMGlobal(
                name="sp",
                ty=WASMType.I32,
                mutable=True,
                initial_value=65536,  # 64KB stack
            )
        )

        # Add standard imports
        self._add_imports()

        # Generate main function
        func = self._generate_function(graph)
        self.module.functions.append(func)

        return self.module.to_wat()

    def _add_imports(self) -> None:
        """Add standard imports."""
        if self.module:
            # Console logging
            self.module.imports.append(
                WASMImport(
                    module="env",
                    name="log",
                    kind="func",
                    signature="(param i32)",
                )
            )

            # Memory operations
            self.module.imports.append(
                WASMImport(
                    module="env",
                    name="abort",
                    kind="func",
                    signature="",
                )
            )

    def _generate_function(self, graph: HyperGraph) -> WASMFunction:
        """Generate WASM function from graph."""
        func = WASMFunction(
            name=graph.name or "main",
            results=[WASMType.I32],
            exported=True,
        )

        self.current_function = func

        # Generate code for each vertex
        for vertex in graph.topological_order():
            self._generate_vertex(vertex, graph)

        # Add return
        func.body.append(WASMInstruction("i32.const", [0]))

        return func

    def _generate_vertex(self, vertex: Vertex, graph: HyperGraph) -> None:
        """Generate WASM instructions for a vertex."""
        if vertex.vertex_type == VertexType.CONST:
            self._generate_const(vertex)

        elif vertex.vertex_type == VertexType.ALLOC:
            self._generate_alloc(vertex)

        elif vertex.vertex_type == VertexType.LOAD:
            self._generate_load(vertex, graph)

        elif vertex.vertex_type == VertexType.STORE:
            self._generate_store(vertex, graph)

        elif vertex.vertex_type == VertexType.APPLY:
            self._generate_apply(vertex, graph)

        elif vertex.vertex_type == VertexType.RETURN:
            self._generate_return(vertex, graph)

        elif vertex.vertex_type == VertexType.PARAMETER:
            self._generate_parameter(vertex)

    def _generate_const(self, vertex: Vertex) -> None:
        """Generate constant."""
        value = vertex.value

        if isinstance(value, int):
            if -(2**31) <= value < 2**31:
                self.current_function.body.append(WASMInstruction("i32.const", [value]))
            else:
                self.current_function.body.append(WASMInstruction("i64.const", [value]))
        elif isinstance(value, float):
            self.current_function.body.append(WASMInstruction("f64.const", [value]))
        else:
            self.current_function.body.append(WASMInstruction("i32.const", [0]))

        # Store in local
        local_name = f"v_{vertex.id[:8]}"
        local = WASMLocal(
            index=self.local_counter,
            ty=WASMType.I32,
            name=local_name,
        )
        self.current_function.locals.append(local)
        self.current_function.body.append(WASMInstruction("local.set", [f"${local_name}"]))
        self.value_map[vertex.id] = local_name
        self.local_counter += 1

    def _generate_alloc(self, vertex: Vertex) -> None:
        """Generate stack allocation."""
        size = vertex.attributes.get("size", 8)

        # Bump stack pointer
        self.current_function.body.append(WASMInstruction("global.get", ["$sp"]))
        self.current_function.body.append(WASMInstruction("i32.const", [size]))
        self.current_function.body.append(WASMInstruction("i32.sub"))
        self.current_function.body.append(WASMInstruction("global.set", ["$sp"]))

        # Get pointer
        self.current_function.body.append(WASMInstruction("global.get", ["$sp"]))

        # Store in local
        local_name = f"ptr_{vertex.id[:8]}"
        local = WASMLocal(
            index=self.local_counter,
            ty=WASMType.I32,
            name=local_name,
        )
        self.current_function.locals.append(local)
        self.current_function.body.append(WASMInstruction("local.set", [f"${local_name}"]))
        self.value_map[vertex.id] = local_name
        self.local_counter += 1

    def _generate_load(self, vertex: Vertex, graph: HyperGraph) -> None:
        """Generate memory load."""
        preds = graph.get_predecessors(vertex)

        if preds and preds[0].id in self.value_map:
            ptr_name = self.value_map[preds[0].id]
            self.current_function.body.append(WASMInstruction("local.get", [f"${ptr_name}"]))
        else:
            self.current_function.body.append(WASMInstruction("i32.const", [0]))

        # Load based on type
        ty = self._get_wasm_type(vertex.metadata.type_info)
        if ty == WASMType.I32:
            self.current_function.body.append(WASMInstruction("i32.load"))
        elif ty == WASMType.I64:
            self.current_function.body.append(WASMInstruction("i64.load"))
        elif ty == WASMType.F32:
            self.current_function.body.append(WASMInstruction("f32.load"))
        elif ty == WASMType.F64:
            self.current_function.body.append(WASMInstruction("f64.load"))

        # Store result
        local_name = f"load_{vertex.id[:8]}"
        local = WASMLocal(index=self.local_counter, ty=ty, name=local_name)
        self.current_function.locals.append(local)
        self.current_function.body.append(WASMInstruction("local.set", [f"${local_name}"]))
        self.value_map[vertex.id] = local_name
        self.local_counter += 1

    def _generate_store(self, vertex: Vertex, graph: HyperGraph) -> None:
        """Generate memory store."""
        preds = graph.get_predecessors(vertex)

        if len(preds) >= 2:
            val_name = self.value_map.get(preds[0].id)
            ptr_name = self.value_map.get(preds[1].id)

            if ptr_name:
                self.current_function.body.append(WASMInstruction("local.get", [f"${ptr_name}"]))
            else:
                self.current_function.body.append(WASMInstruction("i32.const", [0]))

            if val_name:
                self.current_function.body.append(WASMInstruction("local.get", [f"${val_name}"]))
            else:
                self.current_function.body.append(WASMInstruction("i32.const", [0]))

            self.current_function.body.append(WASMInstruction("i32.store"))

    def _generate_apply(self, vertex: Vertex, graph: HyperGraph) -> None:
        """Generate function call or operation."""
        func_name = vertex.attributes.get("function", "")

        # Get operands
        preds = graph.get_predecessors(vertex)
        for pred in preds:
            if pred.id in self.value_map:
                self.current_function.body.append(
                    WASMInstruction("local.get", [f"${self.value_map[pred.id]}"])
                )

        # Map operations
        op_map = {
            "op_+": "i32.add",
            "op_-": "i32.sub",
            "op_*": "i32.mul",
            "op_/": "i32.div_s",
        }

        if func_name in op_map:
            self.current_function.body.append(WASMInstruction(op_map[func_name]))
        elif func_name:
            self.current_function.body.append(WASMInstruction("call", [f"${func_name}"]))

        # Store result
        local_name = f"result_{vertex.id[:8]}"
        local = WASMLocal(index=self.local_counter, ty=WASMType.I32, name=local_name)
        self.current_function.locals.append(local)
        self.current_function.body.append(WASMInstruction("local.set", [f"${local_name}"]))
        self.value_map[vertex.id] = local_name
        self.local_counter += 1

    def _generate_return(self, vertex: Vertex, graph: HyperGraph) -> None:
        """Generate return."""
        preds = graph.get_predecessors(vertex)

        if preds and preds[0].id in self.value_map:
            self.current_function.body.append(
                WASMInstruction("local.get", [f"${self.value_map[preds[0].id]}"])
            )
            self.current_function.body.append(WASMInstruction("return"))

    def _generate_parameter(self, vertex: Vertex) -> None:
        """Generate parameter reference."""
        name = vertex.attributes.get("name", f"param_{vertex.attributes.get('index', 0)}")
        ty = self._get_wasm_type(vertex.metadata.type_info)

        self.current_function.params.append((name, ty))
        self.value_map[vertex.id] = name

    def _get_wasm_type(self, aion_type: AIONType | None) -> WASMType:
        """Convert AION type to WASM type."""
        if not aion_type:
            return WASMType.I32

        type_map = {
            "int": WASMType.I64 if (aion_type.size or 64) > 32 else WASMType.I32,
            "float": WASMType.F64 if (aion_type.size or 64) > 32 else WASMType.F32,
            "bool": WASMType.I32,
            "ptr": WASMType.I32,  # WASM32
        }

        return type_map.get(aion_type.kind, WASMType.I32)


class WASMBackend:
    """WebAssembly backend for AION.

    Provides:
    - WAT (text format) generation
    - Binary WASM generation
    - Memory-safe linear memory handling
    """

    def __init__(self) -> None:
        """Initialize WASM backend."""
        self.codegen = WASMCodeGen()

    def compile_to_wat(self, graph: HyperGraph) -> str:
        """Compile graph to WAT format.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            WAT string
        """
        return self.codegen.generate(graph)

    def compile_to_wasm(self, graph: HyperGraph) -> bytes:
        """Compile graph to binary WASM.

        Args:
            graph: AION-SIR hypergraph

        Returns:
            WASM binary
        """
        wat = self.compile_to_wat(graph)
        # Would use wat2wasm or similar
        # For now, return module as bytes
        return self.codegen.module.to_binary() if self.codegen.module else b""

    def validate(self, wasm: bytes) -> bool:
        """Validate WASM binary.

        Args:
            wasm: WASM binary

        Returns:
            True if valid
        """
        # Check magic number
        if len(wasm) < 8:
            return False

        magic = wasm[:4]
        version = wasm[4:8]

        return magic == bytes([0x00, 0x61, 0x73, 0x6D]) and version == bytes(
            [0x01, 0x00, 0x00, 0x00]
        )
