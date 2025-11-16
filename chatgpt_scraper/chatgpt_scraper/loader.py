"""Helpers for locating and loading ChatGPT export payloads."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List

from .models import RawConversation, RawMessage

_TEMP_DIRS: list[tempfile.TemporaryDirectory] = []


def _register_temp_dir(tmp: tempfile.TemporaryDirectory) -> None:
    """Keep a reference so the directory is cleaned up at interpreter shutdown."""

    _TEMP_DIRS.append(tmp)


def find_export_root(path: Path) -> Path:
    """Return the directory that contains the exported JSON files.

    Args:
        path: Path to a directory or a `.zip` archive that contains the export.

    Returns:
        Path to a directory that can be scanned for `conversations.json` or other
        exported files.
    """

    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        return candidate
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="chatgpt_export_")
        with zipfile.ZipFile(candidate) as zf:
            zf.extractall(tmp.name)
        _register_temp_dir(tmp)
        root = Path(tmp.name)
        contents = [child for child in root.iterdir() if child.name != "__MACOSX"]
        if len(contents) == 1 and contents[0].is_dir():
            return contents[0]
        return root
    raise FileNotFoundError(f"No export found at {candidate}")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _conversations_from_payload(payload) -> List[RawConversation]:
    conversations: List[RawConversation] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if "conversations" in payload and isinstance(payload["conversations"], list):
            items = payload["conversations"]
        else:
            items = [payload]
    else:
        items = []
    for item in items:
        conversations.append(RawConversation.model_validate(item))
    return conversations


def load_raw_conversations(root: Path) -> List[RawConversation]:
    """Load every conversation from the export root."""

    root = Path(root)
    conversations: List[RawConversation] = []
    conversations_file = root / "conversations.json"
    if conversations_file.exists():
        payload = _load_json(conversations_file)
        conversations.extend(_conversations_from_payload(payload))
    else:
        conversations_dir = root / "conversations"
        if conversations_dir.is_dir():
            for child in sorted(conversations_dir.glob("*.json")):
                payload = _load_json(child)
                conversations.extend(_conversations_from_payload(payload))
    if not conversations:
        raise FileNotFoundError(
            f"No conversations were located in {root}. Make sure the export contains a"
            " `conversations.json` file or a `conversations/` directory."
        )
    return conversations


def iter_raw_messages(conv: RawConversation) -> Iterable[RawMessage]:
    """Yield all messages contained in the provided conversation."""

    if conv.messages:
        for message in conv.messages:
            yield RawMessage.model_validate(message)
        return
    if conv.mapping:
        for node in conv.mapping.values():
            message = node.get("message") if isinstance(node, dict) else None
            if message:
                yield RawMessage.model_validate(message)
        return
    fallback_messages = conv.model_dump().get("messages")
    if isinstance(fallback_messages, list):
        for message in fallback_messages:
            yield RawMessage.model_validate(message)


__all__ = ["find_export_root", "load_raw_conversations", "iter_raw_messages"]
