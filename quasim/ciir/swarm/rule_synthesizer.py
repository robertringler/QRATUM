r"""Agent RS — Rule Synthesizer.

Constructs executable physical rules including meta-rules capable
of modifying rules.

Built-in rule templates:
- Linear evolution: s' = A·s
- Nonlinear interaction: s' = s + α·f(s)
- Stochastic perturbation: s' = s + σ·η
- Conservation-aware: project onto constraint surface
- Meta-rules: parameter adaptation, rule pruning
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.swarm.memory import KnowledgeGraph, NodeType
from quasim.ciir.swarm.physics_lang import MetaRule, RuleDecl, UpdateMode


class RuleSynthesizer:
    """Construct executable physical rules."""

    def __init__(self, dim: int, seed: int = 42) -> None:
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    # ---- Rule constructors --------------------------------------

    def linear_evolution_rule(
        self,
        name: str = "LinearEvolution",
        matrix: NDArray | None = None,
    ) -> RuleDecl:
        r"""s' = A·s where A is a contraction (||A|| ≤ 1)."""
        if matrix is None:
            # Random orthogonal + slight contraction
            A = np.linalg.qr(self.rng.standard_normal((self.dim, self.dim)))[0]
            A *= 0.99  # Ensure contraction
        else:
            A = matrix.copy()

        def apply_fn(
            state: NDArray, params: dict, rng: np.random.Generator,
        ) -> NDArray:
            return params["A"] @ state

        return RuleDecl(
            name=name,
            apply_fn=apply_fn,
            params={"A": A},
            mode=UpdateMode.DETERMINISTIC,
            priority=10,
        )

    def nonlinear_interaction_rule(
        self,
        name: str = "NonlinearInteraction",
        alpha: float = 0.1,
    ) -> RuleDecl:
        r"""s' = s + α·tanh(W·s) — bounded nonlinear update."""
        W = self.rng.standard_normal((self.dim, self.dim)) * 0.5

        def apply_fn(
            state: NDArray, params: dict, rng: np.random.Generator,
        ) -> NDArray:
            return state + params["alpha"] * np.tanh(params["W"] @ state)

        return RuleDecl(
            name=name,
            apply_fn=apply_fn,
            params={"alpha": alpha, "W": W},
            mode=UpdateMode.DETERMINISTIC,
            priority=5,
        )

    def stochastic_perturbation_rule(
        self,
        name: str = "StochasticPerturbation",
        sigma: float = 0.01,
    ) -> RuleDecl:
        r"""s' = s + σ·η where η ~ N(0, I)."""

        def apply_fn(
            state: NDArray, params: dict, rng: np.random.Generator,
        ) -> NDArray:
            noise = rng.standard_normal(state.shape) * params["sigma"]
            return state + noise

        return RuleDecl(
            name=name,
            apply_fn=apply_fn,
            params={"sigma": sigma},
            mode=UpdateMode.PROBABILISTIC,
            priority=1,
        )

    def conservation_projection_rule(
        self,
        name: str = "ConservationProjection",
        target_norm: float = 1.0,
    ) -> RuleDecl:
        r"""Project state onto unit sphere: s' = s/||s|| · target_norm."""

        def apply_fn(
            state: NDArray, params: dict, rng: np.random.Generator,
        ) -> NDArray:
            n = np.linalg.norm(state)
            if n > 1e-30:
                return state / n * params["target_norm"]
            return state

        return RuleDecl(
            name=name,
            apply_fn=apply_fn,
            params={"target_norm": target_norm},
            mode=UpdateMode.DETERMINISTIC,
            priority=100,  # High priority — execute last
        )

    # ---- Meta-rule constructors ---------------------------------

    def adaptive_parameter_meta_rule(
        self,
        param_name: str = "alpha",
        decay: float = 0.99,
    ) -> MetaRule:
        r"""Meta-rule: decay a parameter over generations."""

        def condition(program: object, history: list) -> bool:
            return len(history) > 10

        def transform(rules: list[RuleDecl]) -> list[RuleDecl]:
            for rule in rules:
                if param_name in rule.params:
                    rule.params[param_name] *= decay
            return rules

        return MetaRule(
            name=f"Adapt_{param_name}",
            condition=condition,
            transform=transform,
        )

    def synthesize_default_ruleset(self) -> list[RuleDecl]:
        """Create a default balanced rule set."""
        return [
            self.linear_evolution_rule(),
            self.nonlinear_interaction_rule(),
            self.stochastic_perturbation_rule(),
            self.conservation_projection_rule(),
        ]

    def publish_to_memory(
        self,
        memory: KnowledgeGraph,
        rules: list[RuleDecl],
        generation: int = 0,
    ) -> list[int]:
        """Write rules to knowledge graph."""
        ids = []
        for rule in rules:
            node = memory.add_node(
                node_type=NodeType.RULE,
                label=rule.name,
                data=rule,
                generation=generation,
            )
            ids.append(node.id)
        return ids
