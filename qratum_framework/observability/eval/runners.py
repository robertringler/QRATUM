"""runners.py — the prompt × persona matrix executor.

Implements the spec's harness loop::

    for prompt in PROMPTS:
        for model in MODELS:
            output = run_model(prompt, model)
            metrics = evaluate(output, baseline=baseline_model(prompt),
                               clusters=cluster_engine_state)
            store(metrics)

with a baseline-pairing invariant: a verdict for ``(prompt, persona)``
is only emitted when the runner has also produced a baseline run for
``(prompt, neutral, τ ≤ 0.3)``.  This refusal-to-emit-without-baseline
is what lets the regression detector reason about lift in a principled
way.

The default emission model is :class:`SimpleEmissionModel` — a
deterministic, dependency-free token sampler that interleaves the
prompt's topic tokens with the persona's stylistic tokens at a
persona-temperature-controlled mix rate.  In CI this is what produces
the planted drift signal the regression test detects.  Production
runners pass a callable wrapping a real LLM.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from qratum_framework.observability.clusters.state import (
    discover_clusters,
)
from qratum_framework.observability.drift.engine import DriftConfig, DriftEngine
from qratum_framework.observability.drift.report import DriftReport
from qratum_framework.observability.drift.signals import (
    LogprobScorer,
    deterministic_logprob,
)
from qratum_framework.observability.eval.metrics import MatrixMetrics, aggregate_run
from qratum_framework.observability.eval.personas import Persona, default_personas
from qratum_framework.observability.eval.prompts import PromptSpec, default_prompts
from qratum_framework.trace import MerkleLedger

#: Public callable shape for an LLM emission model.
ModelCallable = Callable[[PromptSpec, Persona], Sequence[str]]


# ---------------------------------------------------------------------------
# Built-in deterministic emission model
# ---------------------------------------------------------------------------


class SimpleEmissionModel:
    """A dependency-free deterministic emission model for CI.

    The emitted token stream is a deterministic interleaving of:

    * ``topic_tokens`` from ``prompt`` (always present);
    * ``stylistic_tokens`` from ``persona`` (mixed in at a rate that
      grows with ``persona.temperature``);
    * ``filler`` words taken from a fixed neutral pool (for length).

    No real RNG is touched — the order is derived from a SHA-256 of
    ``(prompt.id, persona.name)`` so that every (prompt, persona)
    pair has a stable but distinct emission stream.
    """

    DEFAULT_FILLER: Tuple[str, ...] = (
        "the",
        "and",
        "of",
        "to",
        "is",
        "in",
        "with",
        "by",
        "for",
        "on",
    )

    def __init__(
        self,
        *,
        n_tokens: int = 64,
        filler: Optional[Sequence[str]] = None,
    ) -> None:
        self.n_tokens = int(n_tokens)
        self.filler = tuple(filler or self.DEFAULT_FILLER)

    def __call__(self, prompt: PromptSpec, persona: Persona) -> Sequence[str]:
        # Drift rate ∈ [0, 1]; neutral baseline (τ=0.2) emits 0% stylistic
        # tokens by construction; nerdy (τ=0.7) emits ~35%.
        drift_rate = max(0.0, min(1.0, (persona.temperature - 0.2) * 0.7))
        seed_bytes = hashlib.sha256(
            f"{prompt.id}|{persona.name}".encode()
        ).digest()
        topic = list(prompt.topic_tokens) or [prompt.prompt.split()[0].lower()]
        stylistic = list(persona.stylistic_tokens)

        out: List[str] = []
        for i in range(self.n_tokens):
            byte = seed_bytes[i % len(seed_bytes)]
            r = byte / 255.0
            if stylistic and r < drift_rate:
                out.append(stylistic[i % len(stylistic)])
            elif r < drift_rate + 0.4:
                out.append(topic[i % len(topic)])
            else:
                out.append(self.filler[i % len(self.filler)])
        return tuple(out)


# ---------------------------------------------------------------------------
# Matrix row record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixRow:
    """Per-(prompt, persona) row in the matrix output.

    Holds enough state for the regression detector + report writer to
    reason about each cell without re-running the model.
    """

    prompt_id: str
    persona: str
    is_baseline: bool
    tokens: Tuple[str, ...]
    drift: DriftReport
    metrics: MatrixMetrics

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt_id,
            "model": self.persona,
            "is_baseline": bool(self.is_baseline),
            "anomaly_rate": float(self.metrics.anomaly_rate),
            "cluster_activation_index": float(
                self.metrics.cluster_activation_index
            ),
            "lift_cluster_rate": float(self.metrics.lift_cluster_rate),
            "drift": self.drift.to_dict(),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BaselineMissingError(RuntimeError):
    """Raised when a non-baseline persona has no paired baseline run."""


class EvalRunner:
    """Cartesian (prompt × persona) executor.

    Construct with optional fixtures + an emission model, then call
    :meth:`run` to get a list of :class:`MatrixRow`.  The runner
    enforces the baseline-pairing invariant (see module docstring) and
    is fully deterministic.
    """

    def __init__(
        self,
        *,
        prompts: Optional[Sequence[PromptSpec]] = None,
        personas: Optional[Sequence[Persona]] = None,
        model: Optional[ModelCallable] = None,
        drift_config: Optional[DriftConfig] = None,
        ledger: Optional[MerkleLedger] = None,
    ) -> None:
        self.prompts = tuple(prompts) if prompts is not None else default_prompts()
        self.personas = (
            tuple(personas) if personas is not None else default_personas()
        )
        self.model: ModelCallable = model or SimpleEmissionModel()
        self.drift_config = drift_config or DriftConfig()
        self._ledger = ledger
        # Validate baseline-pairing invariant up-front.
        baselines = [p for p in self.personas if p.is_baseline]
        if not baselines:
            raise BaselineMissingError(
                "No baseline persona supplied; eval matrix is meaningless without one."
            )
        for b in baselines:
            if b.temperature > 0.3:
                raise BaselineMissingError(
                    f"baseline persona {b.name!r} has temperature {b.temperature}>0.3; "
                    "baselines must use τ ≤ 0.3 (spec invariant)."
                )

    # ------------------------------------------------------------------

    @property
    def baselines(self) -> Tuple[Persona, ...]:
        return tuple(p for p in self.personas if p.is_baseline)

    @property
    def ledger(self) -> Optional[MerkleLedger]:
        return self._ledger

    # ------------------------------------------------------------------

    def run(self) -> List[MatrixRow]:
        """Execute the matrix and return per-cell rows."""
        # Step 1: emit baseline streams once per prompt.  These define
        # the lift/cluster reference for every non-baseline run.
        baseline_persona = self.baselines[0]
        baseline_emissions: Dict[str, Tuple[str, ...]] = {}
        for prompt in self.prompts:
            baseline_emissions[prompt.id] = tuple(self.model(prompt, baseline_persona))

        rows: List[MatrixRow] = []
        step = 0
        for prompt in self.prompts:
            base_tokens = baseline_emissions[prompt.id]
            for persona in self.personas:
                tokens = (
                    base_tokens
                    if persona.is_baseline
                    else tuple(self.model(prompt, persona))
                )

                # Cluster discovery: persona stream vs. paired baseline.
                # For the baseline row itself, both streams are the
                # baseline so no clusters can emerge — the report is
                # therefore trivially clean.
                if persona.is_baseline:
                    cluster_state = discover_clusters(
                        persona_tokens=tokens,
                        baseline_tokens=tokens,
                        window=min(64, max(8, len(tokens))),
                    )
                else:
                    cluster_state = discover_clusters(
                        persona_tokens=tokens,
                        baseline_tokens=base_tokens,
                        window=min(64, max(8, len(tokens))),
                    )

                # Build deterministic logprob scorers from token freqs.
                persona_freqs = _token_frequencies(tokens)
                baseline_freqs = _token_frequencies(base_tokens)
                persona_lp: LogprobScorer = deterministic_logprob(persona_freqs)
                baseline_lp: LogprobScorer = deterministic_logprob(baseline_freqs)

                engine = DriftEngine(
                    config=self.drift_config, ledger=self._ledger
                )
                report, _ = engine.score_and_log(
                    step=step,
                    tokens=tokens,
                    utterance=prompt.utterance,
                    clusters=cluster_state,
                    persona_model=persona_lp,
                    baseline_model=baseline_lp,
                    persona=persona.name,
                )
                metrics = aggregate_run(
                    report=report, cluster_state=cluster_state, tokens=tokens
                )
                rows.append(
                    MatrixRow(
                        prompt_id=prompt.id,
                        persona=persona.name,
                        is_baseline=bool(persona.is_baseline),
                        tokens=tuple(tokens),
                        drift=report,
                        metrics=metrics,
                    )
                )
                step += 1
        return rows


def _token_frequencies(tokens: Sequence[str]) -> Mapping[str, float]:
    """Plain count map used by the deterministic logprob surrogate."""
    out: Dict[str, float] = {}
    for t in tokens:
        out[t] = out.get(t, 0.0) + 1.0
    return out


__all__ = [
    "BaselineMissingError",
    "EvalRunner",
    "MatrixRow",
    "ModelCallable",
    "SimpleEmissionModel",
]
