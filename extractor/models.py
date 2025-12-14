"""Data models for the Empirical Evidence Extractor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ClassificationLabel(str, Enum):
    """Enumeration of supported classification labels."""

    EMPIRICAL_CONFIRMED = "EMPIRICAL_CONFIRMED"
    EMPIRICAL_PLAUSIBLE = "EMPIRICAL_PLAUSIBLE"
    SPECULATIVE = "SPECULATIVE"
    NARRATIVE_META = "NARRATIVE_META"
    UNCLASSIFIABLE = "UNCLASSIFIABLE"


@dataclass(frozen=True)
class MessageRecord:
    """Normalized representation of a message in a conversation."""

    conversation_id: str
    message_index: int
    role: str
    content: str


@dataclass(frozen=True)
class SentenceRecord:
    """Represents a classified sentence."""

    statement_id: str
    conversation_id: str
    message_index: int
    sentence_index: int
    text: str
    classification: ClassificationLabel
    domains: list[str]
    source_path: Path

