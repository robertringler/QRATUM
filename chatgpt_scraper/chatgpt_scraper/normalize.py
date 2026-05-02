"""Normalization helpers for ChatGPT exports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .loader import iter_raw_messages
from .models import (ConversationSummary, NormalizedTurn, RawConversation,
                     RawMessage)

_ALLOWED_ROLES = {"user", "assistant", "system", "tool"}


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            # Allow trailing "Z" or offsets.
            cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _normalize_role(message: RawMessage) -> str:
    candidates = [
        message.role,
        (message.author or {}).get("role") if message.author else None,
        (message.metadata or {}).get("role"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        value = str(candidate).lower()
        if value in _ALLOWED_ROLES:
            return value
        if value.startswith("assistant"):
            return "assistant"
        if value.startswith("user"):
            return "user"
        if value.startswith("system"):
            return "system"
        if value.startswith("tool") or value.startswith("function"):
            return "tool"
    return "assistant"


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            return "\n".join(filter(None, (_flatten_content(part) for part in parts)))
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, (list, tuple)):
        return "\n".join(filter(None, (_flatten_content(part) for part in content)))
    return str(content)


def _message_timestamp(message: RawMessage, fallback: datetime | None) -> datetime:
    timestamp = _coerce_datetime(message.create_time)
    if timestamp:
        return timestamp
    metadata_ts = (
        _coerce_datetime(message.metadata.get("create_time")) if message.metadata else None
    )
    if metadata_ts:
        return metadata_ts
    if fallback:
        return fallback
    return datetime.now(tz=timezone.utc)


def normalize_conversation(conv: RawConversation) -> list[NormalizedTurn]:
    """Normalize every message in the conversation."""

    fallback_ts = _coerce_datetime(conv.create_time) or _coerce_datetime(conv.update_time)
    conversation_id = _conversation_id(conv)
    turns: list[NormalizedTurn] = []
    for raw_message in iter_raw_messages(conv):
        timestamp = _message_timestamp(raw_message, fallback_ts)
        role = _normalize_role(raw_message)
        content = _flatten_content(raw_message.content)
        model_name = None
        if raw_message.metadata:
            model_name = raw_message.metadata.get("model_slug") or raw_message.metadata.get("model")
        if not model_name and isinstance(raw_message.metadata, dict):
            candidate = raw_message.metadata.get("default_model")
            model_name = candidate or model_name
        metadata: dict[str, Any] = {}
        if raw_message.metadata:
            metadata.update(raw_message.metadata)
        metadata.update(
            {
                "message_id": raw_message.id,
                "recipient": raw_message.recipient,
                "weight": raw_message.weight,
            }
        )
        turns.append(
            NormalizedTurn(
                conversation_id=conversation_id,
                timestamp=timestamp,
                role=role,
                content=content,
                model=model_name,
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
        )
    return turns


def _conversation_id(conv: RawConversation) -> str:
    if conv.id:
        return str(conv.id)
    dump = conv.model_dump()
    for key in ("conversation_id", "id"):
        value = dump.get(key)
        if value:
            return str(value)
    return f"conversation-{id(conv)}"


def summarize_conversation(
    conv_id: str, turns: Sequence[NormalizedTurn], raw_title: str | None = None
) -> ConversationSummary:
    """Build a `ConversationSummary` from normalized turns."""

    if not turns:
        now = datetime.now(tz=timezone.utc)
        return ConversationSummary(
            conversation_id=conv_id,
            title="Untitled conversation",
            num_messages=0,
            start_time=now,
            end_time=now,
            models_used=[],
        )
    title = raw_title or turns[0].content.strip().splitlines()[0]
    models = sorted({turn.model for turn in turns if turn.model})
    return ConversationSummary(
        conversation_id=conv_id,
        title=title if title else f"Conversation {conv_id}",
        num_messages=len(turns),
        start_time=min(turn.timestamp for turn in turns),
        end_time=max(turn.timestamp for turn in turns),
        models_used=models,
    )


def build_ledger(
    conversations: Sequence[RawConversation],
) -> tuple[list[NormalizedTurn], list[ConversationSummary]]:
    """Normalize every conversation and build per-conversation summaries."""

    all_turns: list[NormalizedTurn] = []
    summaries: list[ConversationSummary] = []
    for conv in conversations:
        conv_turns = normalize_conversation(conv)
        all_turns.extend(conv_turns)
        summaries.append(summarize_conversation(_conversation_id(conv), conv_turns, conv.title))
    return all_turns, summaries


def write_jsonl(turns: Sequence[NormalizedTurn], path: Path) -> None:
    """Write normalized turns to a JSON Lines file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for turn in turns:
            fh.write(turn.model_dump_json())
            fh.write("\n")


def write_csv(summaries: Sequence[ConversationSummary], path: Path) -> None:
    """Write conversation summaries to a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "conversation_id",
        "title",
        "num_messages",
        "start_time",
        "end_time",
        "models_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "conversation_id": summary.conversation_id,
                    "title": summary.title,
                    "num_messages": summary.num_messages,
                    "start_time": summary.start_time.isoformat(),
                    "end_time": summary.end_time.isoformat(),
                    "models_used": ",".join(summary.models_used),
                }
            )


def summarize_models(turns: Sequence[NormalizedTurn], top_n: int = 5) -> list[tuple[str, int]]:
    """Return the most frequently used models across turns."""

    counter: Counter[str] = Counter(turn.model for turn in turns if turn.model)
    return counter.most_common(top_n)


__all__ = [
    "normalize_conversation",
    "summarize_conversation",
    "build_ledger",
    "write_jsonl",
    "write_csv",
    "summarize_models",
]
