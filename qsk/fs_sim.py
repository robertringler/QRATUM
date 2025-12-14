"""Deterministic in-memory virtual filesystem."""
from __future__ import annotations

from dataclasses import dataclass, field
import builtins


@dataclass
class VFSNode:
    name: str
    is_dir: bool
    children: dict[str, "VFSNode"] = field(default_factory=dict)
    content: str = ""

    def path_hash(self) -> str:
        if self.is_dir:
            child_repr = ":".join(sorted(self.children.keys()))
            return f"dir:{self.name}:{child_repr}"
        return f"file:{self.name}:{len(self.content)}"


class VirtualFileSystem:
    def __init__(self) -> None:
        self.root = VFSNode(name="/", is_dir=True)

    def _walk(self, path: str, create: bool = False, is_dir: bool = False) -> VFSNode:
        parts = [p for p in path.split("/") if p]
        node = self.root
        for part in parts:
            if part not in node.children:
                if not create:
                    raise FileNotFoundError(path)
                node.children[part] = VFSNode(name=part, is_dir=is_dir if part == parts[-1] else True)
            node = node.children[part]
        if is_dir and not node.is_dir:
            raise IsADirectoryError(path)
        return node

    def mkdir(self, path: str) -> None:
        self._walk(path, create=True, is_dir=True)

    def write(self, path: str, content: str) -> None:
        node = self._walk(path, create=True, is_dir=False)
        node.is_dir = False
        node.content = content

    def read(self, path: str) -> str:
        node = self._walk(path)
        if node.is_dir:
            raise IsADirectoryError(path)
        return node.content

    def list(self, path: str = "/") -> builtins.list[str]:
        node = self._walk(path if path != "/" else "/", create=False, is_dir=True)
        return sorted(node.children.keys())

    def fingerprint(self) -> str:
        def _hash(node: VFSNode) -> list[str]:
            if node.is_dir:
                acc = [node.path_hash()]
                for child in sorted(node.children.values(), key=lambda c: c.name):
                    acc.extend(_hash(child))
                return acc
            return [node.path_hash()]

        return "|".join(_hash(self.root))
