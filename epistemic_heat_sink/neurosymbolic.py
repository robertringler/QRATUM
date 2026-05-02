"""Neurosymbolic Reasoning Module.

Implements neurosymbolic reasoning with concept bottlenecks for
interpretable and verifiable inference in the QRATUM epistemic substrate.

Concept Bottleneck Architecture:
1. Input → Neural Encoder → Concept Layer (interpretable)
2. Concept Layer → Symbolic Reasoner → Output
3. Full provenance and explanation generation

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np


class ConceptType(Enum):
    """Types of symbolic concepts in the bottleneck layer."""

    BINARY = auto()  # True/False concepts
    CATEGORICAL = auto()  # Multi-class concepts
    CONTINUOUS = auto()  # Real-valued concepts
    RELATIONAL = auto()  # Relationship between entities


class ReasoningMode(Enum):
    """Modes of neurosymbolic reasoning."""

    DEDUCTIVE = auto()  # From general to specific
    INDUCTIVE = auto()  # From specific to general
    ABDUCTIVE = auto()  # Inference to best explanation
    ANALOGICAL = auto()  # Reasoning by analogy


@dataclass(frozen=True)
class SymbolicConcept:
    """A symbolic concept in the concept bottleneck layer.

    Concepts are interpretable intermediate representations
    that enable both neural and symbolic reasoning.
    """

    name: str
    concept_type: ConceptType
    activation: float  # Activation level [0, 1]
    confidence: float = 1.0  # Confidence in the activation
    description: str = ""

    def __post_init__(self) -> None:
        """Validate concept values."""
        if not 0 <= self.activation <= 1:
            raise ValueError(f"Activation must be in [0, 1], got {self.activation}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_active(self) -> bool:
        """Check if concept is considered active (threshold: 0.5)."""
        return self.activation > 0.5

    @property
    def weighted_activation(self) -> float:
        """Get confidence-weighted activation."""
        return self.activation * self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.concept_type.name,
            "activation": self.activation,
            "confidence": self.confidence,
            "is_active": self.is_active,
            "description": self.description,
        }


@dataclass
class ConceptBottleneck:
    """Concept bottleneck layer for neurosymbolic reasoning.

    The bottleneck enforces that all predictions pass through
    an interpretable concept layer, enabling:
    1. Human inspection of reasoning
    2. Concept-level interventions
    3. Symbolic rule integration
    """

    concepts: list[SymbolicConcept] = field(default_factory=list)
    input_dim: int = 0
    concept_dim: int = 0
    encoder_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Initialize encoder if not provided."""
        if self.encoder_weights is None and self.input_dim > 0 and self.concept_dim > 0:
            # Initialize random projection (placeholder for learned encoder)
            np.random.seed(42)
            self.encoder_weights = np.random.randn(self.input_dim, self.concept_dim) * 0.1

    def encode(self, inputs: np.ndarray) -> list[SymbolicConcept]:
        """Encode inputs through the concept bottleneck.

        Args:
            inputs: Input feature vector

        Returns:
            List of activated concepts
        """
        if self.encoder_weights is None:
            raise ValueError("Encoder not initialized")

        # Project inputs to concept space
        inputs = np.asarray(inputs, dtype=np.float64).flatten()
        if len(inputs) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {len(inputs)}")

        # Apply linear projection + sigmoid
        concept_logits = inputs @ self.encoder_weights
        concept_activations = 1 / (1 + np.exp(-concept_logits))

        # Create concept objects
        concepts = []
        for i, activation in enumerate(concept_activations):
            concept = SymbolicConcept(
                name=f"concept_{i}",
                concept_type=ConceptType.CONTINUOUS,
                activation=float(activation),
                confidence=0.9,  # Placeholder confidence
                description=f"Encoded concept {i}",
            )
            concepts.append(concept)

        self.concepts = concepts
        return concepts

    def intervene(self, concept_name: str, new_activation: float) -> None:
        """Intervene on a concept by setting its activation.

        This enables counterfactual reasoning and debugging.

        Args:
            concept_name: Name of concept to intervene on
            new_activation: New activation value [0, 1]
        """
        for i, concept in enumerate(self.concepts):
            if concept.name == concept_name:
                self.concepts[i] = SymbolicConcept(
                    name=concept.name,
                    concept_type=concept.concept_type,
                    activation=new_activation,
                    confidence=1.0,  # Intervention has full confidence
                    description=f"{concept.description} (intervened)",
                )
                return
        raise ValueError(f"Concept {concept_name} not found")

    def get_active_concepts(self, threshold: float = 0.5) -> list[SymbolicConcept]:
        """Get concepts with activation above threshold."""
        return [c for c in self.concepts if c.activation > threshold]

    def compute_hash(self) -> str:
        """Compute hash of concept state for provenance."""
        data = ":".join(f"{c.name}:{c.activation:.6f}:{c.confidence:.6f}" for c in self.concepts)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_dim": self.input_dim,
            "concept_dim": self.concept_dim,
            "concepts": [c.to_dict() for c in self.concepts],
            "active_concepts": len(self.get_active_concepts()),
            "state_hash": self.compute_hash(),
        }


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""

    step_id: int
    rule_applied: str
    input_concepts: list[str]
    output_concepts: list[str]
    confidence: float
    justification: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "rule_applied": self.rule_applied,
            "input_concepts": self.input_concepts,
            "output_concepts": self.output_concepts,
            "confidence": self.confidence,
            "justification": self.justification,
        }


@dataclass
class ReasoningTrace:
    """Full trace of neurosymbolic reasoning.

    Provides complete provenance for verification and audit.
    """

    trace_id: str
    mode: ReasoningMode
    steps: list[ReasoningStep] = field(default_factory=list)
    input_concepts: list[SymbolicConcept] = field(default_factory=list)
    output_concepts: list[SymbolicConcept] = field(default_factory=list)
    execution_time_ms: float = 0.0

    @property
    def total_confidence(self) -> float:
        """Compute total confidence (product of step confidences)."""
        if not self.steps:
            return 0.0
        confidence = 1.0
        for step in self.steps:
            confidence *= step.confidence
        return confidence

    def add_step(
        self,
        rule: str,
        inputs: list[str],
        outputs: list[str],
        confidence: float,
        justification: str = "",
    ) -> None:
        """Add a reasoning step to the trace."""
        step = ReasoningStep(
            step_id=len(self.steps),
            rule_applied=rule,
            input_concepts=inputs,
            output_concepts=outputs,
            confidence=confidence,
            justification=justification,
        )
        self.steps.append(step)

    def compute_provenance_hash(self) -> str:
        """Compute hash for provenance verification."""
        data = f"{self.trace_id}:{self.mode.name}:{len(self.steps)}"
        for step in self.steps:
            data += f":{step.rule_applied}:{step.confidence}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "mode": self.mode.name,
            "steps": [s.to_dict() for s in self.steps],
            "input_concepts": [c.to_dict() for c in self.input_concepts],
            "output_concepts": [c.to_dict() for c in self.output_concepts],
            "total_confidence": self.total_confidence,
            "execution_time_ms": self.execution_time_ms,
            "provenance_hash": self.compute_provenance_hash(),
        }


class NeurosymbolicReasoner:
    """Neurosymbolic reasoner with concept bottleneck.

    Combines neural encoding with symbolic reasoning for
    interpretable and verifiable inference.
    """

    def __init__(
        self,
        input_dim: int,
        concept_dim: int,
        rules: (
            list[tuple[str, Callable[[list[SymbolicConcept]], list[SymbolicConcept]]]] | None
        ) = None,
    ) -> None:
        """Initialize the neurosymbolic reasoner.

        Args:
            input_dim: Dimension of input features
            concept_dim: Number of concepts in bottleneck
            rules: Optional list of (name, rule_function) tuples
        """
        self.bottleneck = ConceptBottleneck(
            input_dim=input_dim,
            concept_dim=concept_dim,
        )
        self.rules: dict[str, Callable[[list[SymbolicConcept]], list[SymbolicConcept]]] = {}

        if rules:
            for name, rule_fn in rules:
                self.rules[name] = rule_fn

        # Add default rules
        self._add_default_rules()

        self._trace_count = 0

    def _add_default_rules(self) -> None:
        """Add default symbolic reasoning rules."""

        def threshold_rule(concepts: list[SymbolicConcept]) -> list[SymbolicConcept]:
            """Apply threshold to concepts."""
            return [
                SymbolicConcept(
                    name=f"thresholded_{c.name}",
                    concept_type=ConceptType.BINARY,
                    activation=1.0 if c.activation > 0.5 else 0.0,
                    confidence=c.confidence,
                    description=f"Thresholded: {c.is_active}",
                )
                for c in concepts
            ]

        def aggregate_rule(concepts: list[SymbolicConcept]) -> list[SymbolicConcept]:
            """Aggregate multiple concepts."""
            if not concepts:
                return []
            avg_activation = np.mean([c.activation for c in concepts])
            avg_confidence = np.mean([c.confidence for c in concepts])
            return [
                SymbolicConcept(
                    name="aggregated",
                    concept_type=ConceptType.CONTINUOUS,
                    activation=float(avg_activation),
                    confidence=float(avg_confidence),
                    description=f"Aggregated from {len(concepts)} concepts",
                )
            ]

        self.rules["threshold"] = threshold_rule
        self.rules["aggregate"] = aggregate_rule

    def add_rule(
        self,
        name: str,
        rule_fn: Callable[[list[SymbolicConcept]], list[SymbolicConcept]],
    ) -> None:
        """Add a symbolic reasoning rule."""
        self.rules[name] = rule_fn

    def reason(
        self,
        inputs: np.ndarray,
        mode: ReasoningMode = ReasoningMode.DEDUCTIVE,
        rules_to_apply: list[str] | None = None,
    ) -> ReasoningTrace:
        """Perform neurosymbolic reasoning.

        Args:
            inputs: Input feature vector
            mode: Reasoning mode to use
            rules_to_apply: Optional list of rules to apply

        Returns:
            ReasoningTrace with full provenance
        """
        start_time = time.time()
        self._trace_count += 1

        trace = ReasoningTrace(
            trace_id=f"trace_{self._trace_count}_{int(start_time * 1000)}",
            mode=mode,
        )

        # Step 1: Encode through concept bottleneck
        concepts = self.bottleneck.encode(inputs)
        trace.input_concepts = list(concepts)

        trace.add_step(
            rule="encode",
            inputs=["raw_input"],
            outputs=[c.name for c in concepts],
            confidence=0.95,
            justification="Neural encoding through concept bottleneck",
        )

        # Step 2: Apply symbolic rules
        current_concepts = concepts
        rules_sequence = rules_to_apply or list(self.rules.keys())[:2]  # Default: first 2 rules

        for rule_name in rules_sequence:
            if rule_name not in self.rules:
                continue

            rule_fn = self.rules[rule_name]
            input_names = [c.name for c in current_concepts]

            new_concepts = rule_fn(current_concepts)
            output_names = [c.name for c in new_concepts]

            trace.add_step(
                rule=rule_name,
                inputs=input_names,
                outputs=output_names,
                confidence=0.9,
                justification=f"Applied rule: {rule_name}",
            )

            current_concepts = new_concepts

        trace.output_concepts = current_concepts
        trace.execution_time_ms = (time.time() - start_time) * 1000

        return trace

    def explain(self, trace: ReasoningTrace) -> str:
        """Generate human-readable explanation of reasoning.

        Args:
            trace: Reasoning trace to explain

        Returns:
            Human-readable explanation
        """
        lines = [
            f"Reasoning Trace: {trace.trace_id}",
            f"Mode: {trace.mode.name}",
            f"Total Confidence: {trace.total_confidence:.4f}",
            "",
            "Steps:",
        ]

        for step in trace.steps:
            lines.append(
                f"  {step.step_id}. {step.rule_applied}: "
                f"{step.input_concepts} → {step.output_concepts} "
                f"(conf: {step.confidence:.2f})"
            )
            if step.justification:
                lines.append(f"     {step.justification}")

        lines.append("")
        lines.append("Input Concepts:")
        for c in trace.input_concepts:
            lines.append(f"  - {c.name}: {c.activation:.4f} ({c.concept_type.name})")

        lines.append("")
        lines.append("Output Concepts:")
        for c in trace.output_concepts:
            lines.append(f"  - {c.name}: {c.activation:.4f} ({c.concept_type.name})")

        return "\n".join(lines)

    @property
    def trace_count(self) -> int:
        """Get number of reasoning traces generated."""
        return self._trace_count


def create_concept_bottleneck(
    input_dim: int,
    concept_dim: int,
) -> ConceptBottleneck:
    """Factory function to create a concept bottleneck.

    Args:
        input_dim: Input feature dimension
        concept_dim: Number of concepts

    Returns:
        Configured ConceptBottleneck
    """
    return ConceptBottleneck(input_dim=input_dim, concept_dim=concept_dim)
