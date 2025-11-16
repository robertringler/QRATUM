"""Typed models used throughout the export normalization pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RawMessage(BaseModel):
    """Representation of a single message from the export."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    author: Optional[Dict[str, Any]] = None
    role: Optional[str] = None
    create_time: Optional[Any] = Field(default=None, description="Raw timestamp as stored in the export")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: Any = None
    weight: Optional[float] = None
    recipient: Optional[str] = None


class RawConversation(BaseModel):
    """Representation of a conversation inside the export."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    title: Optional[str] = None
    create_time: Optional[Any] = None
    update_time: Optional[Any] = None
    mapping: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None


class NormalizedTurn(BaseModel):
    """Normalized representation of a single conversation turn."""

    conversation_id: str
    timestamp: datetime
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    """Aggregated metadata for a conversation."""

    conversation_id: str
    title: str
    num_messages: int
    start_time: datetime
    end_time: datetime
    models_used: List[str]


__all__ = [
    "RawConversation",
    "RawMessage",
    "NormalizedTurn",
    "ConversationSummary",
]
