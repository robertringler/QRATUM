"""Bridge between QDL constructs and world model interfaces."""

from __future__ import annotations

from typing import Any

from qdl.compiler import Compiler


def compile_worldmodel_call(source: str, worldmodel: Any):
    compiler = Compiler(source)
    fn = compiler.lower()
    return fn(), worldmodel
