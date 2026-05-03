"""prompts.py — built-in prompt fixtures.

The spec ships three canonical prompts; we expose them as the default
prompt matrix and let callers extend or replace via the runner API.
Prompts are kept in code (not YAML on disk) so the eval harness has
zero filesystem dependencies for the smoke tests; richer corpora can
still be loaded externally and passed to :class:`EvalRunner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PromptSpec:
    """A prompt + the topical "utterance" used for relevance scoring.

    The spec uses the symbol ``U`` for the user utterance against
    which off-topic-ness is measured.  We default ``utterance`` to
    the prompt itself; callers can override when the relevance
    reference differs from the prompt (e.g. system prompts).

    ``topic_tokens`` is the on-topic vocabulary; the eval harness
    treats these tokens as low-relevance distance to ``utterance``
    when the deterministic embedder cannot provide useful signal.
    """

    id: str
    prompt: str
    utterance: str = ""
    topic_tokens: Tuple[str, ...] = ()

    def __post_init__(self) -> None:  # pragma: no cover — trivial
        if not self.utterance:
            object.__setattr__(self, "utterance", self.prompt)


_DEFAULT: Tuple[PromptSpec, ...] = (
    PromptSpec(
        id="postgres_pooling",
        prompt="Explain Postgres pooling",
        topic_tokens=(
            "postgres",
            "pooling",
            "connection",
            "pool",
            "pgbouncer",
            "transaction",
            "session",
            "database",
        ),
    ),
    PromptSpec(
        id="sorting_algorithm",
        prompt="Write a sorting algorithm",
        topic_tokens=(
            "sort",
            "sorting",
            "algorithm",
            "array",
            "compare",
            "swap",
            "partition",
            "merge",
        ),
    ),
    PromptSpec(
        id="quantum_tunneling",
        prompt="Summarize quantum tunneling",
        topic_tokens=(
            "quantum",
            "tunneling",
            "barrier",
            "wavefunction",
            "energy",
            "potential",
            "particle",
            "probability",
        ),
    ),
)


def default_prompts() -> Tuple[PromptSpec, ...]:
    """Return the canonical three-prompt fixture from the spec."""
    return _DEFAULT


__all__ = ["PromptSpec", "default_prompts"]
