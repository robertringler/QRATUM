"""personas.py — built-in persona configurations.

Three personas mirror the spec's matrix: ``neutral`` (the baseline),
``nerdy`` (jargon-leaning), ``formal`` (register-leaning).  Each
persona carries:

* a ``name`` — used as the matrix-row label;
* ``temperature`` — the decoding parameter (the runner's
  baseline-pairing invariant requires the neutral persona's
  temperature to be ≤ 0.3);
* ``stylistic_tokens`` — tokens the deterministic emission model
  injects with elevated frequency so the cluster engine has
  persona-vs-baseline signal to discover.  The ``nerdy`` and
  ``formal`` token sets are deliberately *off-topic* relative to the
  spec's three prompts so the drift engine has something to detect.

Production deployments swap the deterministic emission model out for a
real LLM and the ``stylistic_tokens`` field becomes irrelevant — but
the persona name + temperature still parameterise the matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Persona:
    """Persona row in the eval matrix."""

    name: str
    temperature: float
    stylistic_tokens: Tuple[str, ...] = ()
    is_baseline: bool = False


_DEFAULT_PERSONAS: Tuple[Persona, ...] = (
    Persona(
        name="neutral",
        temperature=0.2,
        stylistic_tokens=(),
        is_baseline=True,
    ),
    Persona(
        name="nerdy",
        temperature=0.7,
        # Off-topic stylistic tokens: a "nerdy" persona drifts toward
        # fantasy-roleplay vocabulary that has nothing to do with the
        # spec's three prompts.  This is the canonical drift the
        # spec's running example targets.
        stylistic_tokens=(
            "goblin",
            "troll",
            "ogre",
            "gremlin",
            "wizard",
            "dragon",
            "elf",
            "potion",
        ),
    ),
    Persona(
        name="formal",
        temperature=0.3,
        # A "formal" persona drifts toward bureaucratic register.
        stylistic_tokens=(
            "henceforth",
            "heretofore",
            "aforementioned",
            "pursuant",
            "notwithstanding",
            "moreover",
            "hereinafter",
            "whereby",
        ),
    ),
)


def default_personas() -> Tuple[Persona, ...]:
    """Return the canonical three-persona fixture."""
    return _DEFAULT_PERSONAS


__all__ = ["Persona", "default_personas"]
