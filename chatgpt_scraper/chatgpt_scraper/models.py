"""Typed models used throughout the export normalization pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RawMessage(BaseModel):
    """Representation of a single message from the export."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    author: dict[str, Any] | None = None
    role: str | None = None
    create_time: Any | None = Field(
        default=None, description="Raw timestamp as stored in the export"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: Any = None
    weight: float | None = None
    recipient: str | None = None


class RawConversation(BaseModel):
    """Representation of a conversation inside the export."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    title: str | None = None
    create_time: Any | None = None
    update_time: Any | None = None
    mapping: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None


class NormalizedTurn(BaseModel):
    """Normalized representation of a single conversation turn."""

    conversation_id: str
    timestamp: datetime
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    """Aggregated metadata for a conversation."""

    conversation_id: str
    title: str
    num_messages: int
    start_time: datetime
    end_time: datetime
    models_used: list[str]


__all__ = [
    "RawConversation",
    "RawMessage",
    "NormalizedTurn",
    "ConversationSummary",
]
