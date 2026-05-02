"""LLM proposer backends for the Reality Interface Controller.

The RIC is a deterministic bounded control operator. LLMs are wired in as
*proposers* (suggesting candidate magnitudes per action type) rather than as
the policy itself — so safety and stability dominance are always enforced
locally regardless of LLM behavior.

Each backend implements :class:`qagents.reality_interface.Proposer` and
*MUST* fall back deterministically when the underlying SDK or API key is
missing. This ensures the engine runs in any environment (CI, sandbox).

Available backends
------------------
- :class:`DeterministicLLM`     — always-available baseline, no network.
- :class:`OpenAIProposer`       — uses ``OPENAI_API_KEY``; falls back to
  :class:`DeterministicLLM` when SDK or key missing.
- :class:`AnthropicProposer`    — uses ``ANTHROPIC_API_KEY``; same fallback.
- :class:`GeminiProposer`       — uses ``GOOGLE_API_KEY``; same fallback.
- :class:`LocalLLMProposer`     — calls a local HTTP endpoint (e.g. Ollama,
  llama.cpp server); same fallback.

The :func:`available_backends` helper builds one proposer per supported
provider, all of which are guaranteed to return a valid magnitude mapping.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from qagents.reality_interface import ACTION_TYPES, ActionType, Proposer

# ---------------------------------------------------------------------------
# Deterministic baseline
# ---------------------------------------------------------------------------


def _heuristic_magnitudes(
    intent: Mapping[str, Any],
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> dict[ActionType, float]:
    """Compact, deterministic heuristic.

    - ``control``  scaled by intent priority
    - ``adjust``   = control / 4
    - ``hold``     = 0
    - ``abort``    = 0  (RIC selects abort itself via Rule 1)
    """

    priority = float(intent.get("priority", 0.5))
    base = max(0.0, min(1.0, priority))
    return {
        "control": base,
        "adjust": base * 0.25,
        "hold": 0.0,
        "abort": 0.0,
    }


@dataclass
class DeterministicLLM:
    """Deterministic baseline 'LLM' — same I/O surface, no network."""

    name: str = "deterministic"

    def propose(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> dict[ActionType, float]:
        return _heuristic_magnitudes(intent, world_state, system_limits)


# ---------------------------------------------------------------------------
# Generic LLM-backed proposer with deterministic fallback
# ---------------------------------------------------------------------------


def _build_prompt(
    intent: Mapping[str, Any],
    world_state: Mapping[str, Any],
    system_limits: Mapping[str, Any],
) -> str:
    payload = {
        "intent": dict(intent),
        "world_state": {
            k: v for k, v in world_state.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        },
        "system_limits": dict(system_limits),
    }
    return (
        "You are a magnitude proposer for a Reality Interface Controller. "
        "Return ONLY a JSON object with numeric magnitudes >= 0 for keys "
        f"{list(ACTION_TYPES)}. Input:\n{json.dumps(payload)}"
    )


def _parse_magnitudes(text: str) -> dict[ActionType, float] | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[ActionType, float] = {}
    for key in ACTION_TYPES:
        try:
            out[key] = max(0.0, float(data.get(key, 0.0)))
        except (TypeError, ValueError):
            return None
    return out


class _BaseLLMProposer:
    """Base class enforcing deterministic fallback contract."""

    name: str = "base"

    def __init__(self, fallback: Proposer | None = None) -> None:
        self._fallback: Proposer = fallback or DeterministicLLM(name=f"{self.name}-fallback")

    # Subclasses override this with real network logic. Returning ``None``
    # signals "not available; use fallback".
    def _call_llm(self, prompt: str) -> str | None:  # pragma: no cover - overridden
        return None

    def propose(
        self,
        intent: Mapping[str, Any],
        world_state: Mapping[str, Any],
        system_limits: Mapping[str, Any],
    ) -> dict[ActionType, float]:
        try:
            text = self._call_llm(_build_prompt(intent, world_state, system_limits))
        except Exception:
            text = None
        if text is not None:
            parsed = _parse_magnitudes(text)
            if parsed is not None:
                return parsed
        return self._fallback.propose(intent, world_state, system_limits)


class OpenAIProposer(_BaseLLMProposer):
    """OpenAI Chat Completions backend with deterministic fallback."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        fallback: Proposer | None = None,
    ) -> None:
        super().__init__(fallback=fallback)
        self.model = model
        self._api_key = os.environ.get(api_key_env)

    def _call_llm(self, prompt: str) -> str | None:
        if not self._api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError:
            return None
        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content


class AnthropicProposer(_BaseLLMProposer):
    """Anthropic Messages backend with deterministic fallback."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key_env: str = "ANTHROPIC_API_KEY",
        fallback: Proposer | None = None,
    ) -> None:
        super().__init__(fallback=fallback)
        self.model = model
        self._api_key = os.environ.get(api_key_env)

    def _call_llm(self, prompt: str) -> str | None:
        if not self._api_key:
            return None
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError:
            return None
        client = anthropic.Anthropic(api_key=self._api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        # SDK returns a list of content blocks; concatenate text blocks.
        parts = []
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts) if parts else None


class GeminiProposer(_BaseLLMProposer):
    """Google Gemini backend with deterministic fallback."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key_env: str = "GOOGLE_API_KEY",
        fallback: Proposer | None = None,
    ) -> None:
        super().__init__(fallback=fallback)
        self.model = model
        self._api_key = os.environ.get(api_key_env)

    def _call_llm(self, prompt: str) -> str | None:
        if not self._api_key:
            return None
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
        except ImportError:
            return None
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self.model)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None)


class LocalLLMProposer(_BaseLLMProposer):
    """HTTP-based local LLM proposer (e.g. Ollama / llama.cpp server)."""

    name = "local"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "llama3",
        timeout_s: float = 5.0,
        fallback: Proposer | None = None,
    ) -> None:
        super().__init__(fallback=fallback)
        self.endpoint = endpoint or os.environ.get("LOCAL_LLM_ENDPOINT")
        self.model = model
        self.timeout_s = timeout_s

    def _call_llm(self, prompt: str) -> str | None:
        if not self.endpoint:
            return None
        try:
            import urllib.request
        except ImportError:  # pragma: no cover
            return None
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310
                raw = resp.read().decode("utf-8")
        except Exception:
            return None
        # Ollama-style: {"response": "..."}; OpenAI-style proxies: {"choices":[...]}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(data, dict):
            if "response" in data:
                return str(data["response"])
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                return msg.get("content") or choices[0].get("text")
        return None


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------


def available_backends() -> list[Proposer]:
    """Return one proposer per supported provider, all callable.

    The deterministic baseline is always included. Network-backed proposers
    fall back to the deterministic baseline when their SDK/API key is absent,
    so this list is safe to iterate in any environment.
    """

    return [
        DeterministicLLM(),
        OpenAIProposer(),
        AnthropicProposer(),
        GeminiProposer(),
        LocalLLMProposer(),
    ]


__all__ = [
    "AnthropicProposer",
    "DeterministicLLM",
    "GeminiProposer",
    "LocalLLMProposer",
    "OpenAIProposer",
    "available_backends",
]
