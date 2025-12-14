"""Input/Output helpers for the Empirical Evidence Extractor."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, MutableMapping, Sequence

from .models import ClassificationLabel, MessageRecord, SentenceRecord

LOGGER = logging.getLogger(__name__)


class ConversationLoader:
    """Loads ChatGPT conversation exports into normalized message records."""

    def __init__(self, input_path: Path) -> None:
        self.input_path = Path(input_path)

    def load(self) -> list[MessageRecord]:
        """Load and normalize messages from the conversations export."""

        LOGGER.info("Loading conversations from %s", self.input_path)
        data = json.loads(self.input_path.read_text(encoding="utf-8"))
        conversations: Sequence[MutableMapping[str, object]]
        if isinstance(data, dict) and "conversations" in data:
            conversations = list(_ensure_dict_list(data["conversations"]))
        elif isinstance(data, list):
            conversations = list(_ensure_dict_list(data))
        else:
            raise ValueError("Unsupported conversations.json structure")

        records: list[MessageRecord] = []
        for conv_index, conversation in enumerate(conversations):
            conv_id = _resolve_conversation_id(conversation, conv_index)
            for message_index, message in enumerate(self._iter_messages(conversation)):
                text = _extract_text(message)
                if not text.strip():
                    continue
                role = _extract_role(message)
                records.append(
                    MessageRecord(
                        conversation_id=conv_id,
                        message_index=message_index,
                        role=role,
                        content=text.strip(),
                    )
                )
        LOGGER.info("Loaded %d normalized messages", len(records))
        return records

    def _iter_messages(
        self, conversation: MutableMapping[str, object]
    ) -> Iterator[MutableMapping[str, object]]:
        """Yield raw message dictionaries from a conversation payload."""

        if isinstance(conversation.get("messages"), list):
            yield from _ensure_dict_list(conversation["messages"])
            return

        mapping = conversation.get("mapping")
        if isinstance(mapping, dict):
            nodes = [
                node for node in mapping.values() if isinstance(node, dict) and node.get("message")
            ]
            nodes.sort(key=lambda node: node.get("message", {}).get("create_time", 0) or 0)
            for node in nodes:
                message = node.get("message")
                if isinstance(message, dict):
                    yield message
            return

        if isinstance(conversation.get("items"), list):
            yield from _ensure_dict_list(conversation["items"])
            return

        LOGGER.debug("Conversation %s had no recognizable message container", conversation)


def _ensure_dict_list(items: object) -> Iterator[MutableMapping[str, object]]:
    for item in items or []:  # type: ignore[assignment]
        if isinstance(item, dict):
            yield item


def _resolve_conversation_id(conversation: MutableMapping[str, object], index: int) -> str:
    for key in ("conversation_id", "conversationId", "id", "title"):
        value = conversation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"conversation_{index:05d}"


def _extract_role(message: MutableMapping[str, object]) -> str:
    author = message.get("author")
    if isinstance(author, dict):
        role = author.get("role")
        if isinstance(role, str):
            return role
    role_field = message.get("role")
    if isinstance(role_field, str):
        return role_field
    return "unknown"


def _extract_text(message: MutableMapping[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            text_parts = [part for part in parts if isinstance(part, str)]
            if text_parts:
                return "\n".join(text_parts)
        text = content.get("text")
        if isinstance(text, str):
            return text
    body = message.get("body")
    if isinstance(body, str):
        return body
    return ""


class LedgerWriter:
    """Persists classified sentences into a JSON Lines ledger."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def write(self, records: Iterable[SentenceRecord]) -> Path:
        ledger_path = self.output_dir / "empirical_knowledge_ledger.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing ledger to %s", ledger_path)
        with ledger_path.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = asdict(record)
                payload["classification"] = record.classification.value
                payload["source_path"] = str(record.source_path)
                payload["artifact_path"] = str(ledger_path)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return ledger_path


class SummaryWriter:
    """Aggregates classification statistics into a CSV summary."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def write(self, records: Iterable[SentenceRecord]) -> Path:
        summary_path = self.output_dir / "corpus_classification_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        counts: Counter[str] = Counter()
        domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
        buffered: list[SentenceRecord] = list(records)
        for record in buffered:
            label = record.classification.value
            counts[label] += 1
            for domain in record.domains:
                domain_counts[label][domain] += 1

        LOGGER.info("Writing summary to %s", summary_path)
        lines: list[str] = ["classification,count,top_domains"]
        for label in ClassificationLabel:
            label_value = label.value
            count = counts.get(label_value, 0)
            top_domains = "; ".join(
                f"{domain}:{domain_counts[label_value][domain]}"
                for domain in domain_counts[label_value].most_common(3)
            )
            lines.append(f'{label_value},{count},"{top_domains}"')
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary_path
