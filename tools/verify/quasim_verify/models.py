"""Data models for verification tool."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CheckResult(BaseModel):
    """Result of a single verification check."""

    id: str
    passed: bool
    severity: str = "error"  # error|warn|info
    details: dict[str, Any] = Field(default_factory=dict)
    evidence_paths: list[str] = Field(default_factory=list)


class Report(BaseModel):
    """Complete verification report."""

    version: str
    started_at: datetime
    finished_at: datetime
    results: list[CheckResult]
    summary: dict[str, Any]


class VerifyConfig(BaseModel):
    """Configuration for verification run."""

    version: int
    inputs: dict[str, Any]
    checks: list[dict[str, str]]
    policy: dict[str, Any]
    outputs: dict[str, Any]
