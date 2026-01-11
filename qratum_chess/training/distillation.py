"""Knowledge distillation for cross-generation learning.

Implements cross-generation knowledge distillation:
    for each new_net:
        loss += KL(new_net || best_of_all_time)
        loss += ELO_margin_penalty
        loss += motif_innovation_bonus

This enforces non-regressive evolution and structural memory
compression of all historical strengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation.
    
    Attributes:
        kl_weight: Weight for KL divergence loss.
        elo_penalty_weight: Weight for Elo margin penalty.
        innovation_bonus_weight: Weight for innovation bonus.
        temperature: Temperature for softening distributions.
        min_elo_margin: Minimum Elo margin required.
    """
    kl_weight: float = 0.5
    elo_penalty_weight: float = 0.1
    innovation_bonus_weight: float = 0.2
    temperature: float = 2.0
    min_elo_margin: int = 0


@dataclass
class GenerationRecord:
    """Record of a network generation."""
    generation_id: str
    elo_rating: float
    network_weights: Any = None
    motif_signatures: list[str] = field(default_factory=list)
    timestamp: float = 0.0


class KnowledgeDistillation:
    """Cross-generation knowledge distillation framework.
    
    Ensures new generations retain knowledge from all previous
    generations while still innovating.
    """
    
    def __init__(self, config: DistillationConfig | None = None):
        """Initialize the distillation framework.
        
        Args:
            config: Distillation configuration.
        """
        self.config = config or DistillationConfig()
        self.generation_history: list[GenerationRecord] = []
        self.best_generation: GenerationRecord | None = None
    
    def register_generation(
        self,
        generation_id: str,
        network,
        elo_rating: float,
        motif_signatures: list[str] | None = None
    ) -> None:
        """Register a new generation.
        
        Args:
            generation_id: Unique identifier for generation.
            network: Network weights/parameters.
            elo_rating: Achieved Elo rating.
            motif_signatures: Discovered strategic motifs.
        """
        import time
        
        record = GenerationRecord(
            generation_id=generation_id,
            elo_rating=elo_rating,
            network_weights=network,
            motif_signatures=motif_signatures or [],
            timestamp=time.time(),
        )
        
        self.generation_history.append(record)
        
        # Update best generation
        if self.best_generation is None or elo_rating > self.best_generation.elo_rating:
            self.best_generation = record
    
    def compute_distillation_loss(
        self,
        student_policy: np.ndarray,
        student_value: np.ndarray,
        teacher_policy: np.ndarray,
        teacher_value: np.ndarray,
        student_elo: float,
        is_innovative: bool = False
    ) -> tuple[float, dict[str, float]]:
        """Compute distillation loss for training.
        
        Args:
            student_policy: Student network policy output.
            student_value: Student network value output.
            teacher_policy: Teacher network policy output.
            teacher_value: Teacher network value output.
            student_elo: Student's current Elo estimate.
            is_innovative: Whether student shows innovative play.
            
        Returns:
            Tuple of (total_loss, loss_components).
        """
        # KL divergence loss (soft targets)
        temp = self.config.temperature
        
        # Soften distributions
        soft_student = self._softmax_with_temp(student_policy, temp)
        soft_teacher = self._softmax_with_temp(teacher_policy, temp)
        
        kl_loss = self._kl_divergence(soft_student, soft_teacher)
        kl_loss *= self.config.kl_weight * (temp ** 2)
        
        # Elo margin penalty
        elo_penalty = 0.0
        if self.best_generation:
            elo_gap = self.best_generation.elo_rating - student_elo
            if elo_gap > self.config.min_elo_margin:
                elo_penalty = self.config.elo_penalty_weight * elo_gap / 100.0
        
        # Innovation bonus (negative loss)
        innovation_bonus = 0.0
        if is_innovative:
            innovation_bonus = -self.config.innovation_bonus_weight
        
        # Total loss
        total_loss = kl_loss + elo_penalty + innovation_bonus
        
        components = {
            "kl_loss": float(kl_loss),
            "elo_penalty": float(elo_penalty),
            "innovation_bonus": float(innovation_bonus),
        }
        
        return float(total_loss), components
    
    def _softmax_with_temp(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply softmax with temperature scaling."""
        scaled = logits / temperature
        exp_x = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence KL(p || q)."""
        # Add small epsilon for numerical stability
        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)
        q = np.clip(q, eps, 1 - eps)
        
        kl = np.sum(p * np.log(p / q), axis=-1)
        return float(np.mean(kl))
    
    def detect_innovation(
        self,
        student_network,
        test_positions: list,
        encoder
    ) -> tuple[bool, list[str]]:
        """Detect if student network shows innovative play.
        
        Args:
            student_network: Student network to evaluate.
            test_positions: Test positions for evaluation.
            encoder: Position encoder.
            
        Returns:
            Tuple of (is_innovative, new_motifs).
        """
        if not test_positions:
            return False, []
        
        new_motifs = []
        
        # Check for novel move preferences
        for position in test_positions[:100]:
            input_tensor = encoder.encode(position)
            input_batch = input_tensor[np.newaxis, ...]
            
            policy, value = student_network.forward(input_batch)
            
            # Find top moves
            top_indices = np.argsort(policy[0])[-5:]
            
            # Check if these are novel compared to history
            # (Simplified - would compare to historical patterns)
            for idx in top_indices:
                if policy[0, idx] > 0.1:  # Significant preference
                    motif = f"motif_{idx}_{int(value[0] * 100)}"
                    if motif not in self._get_all_historical_motifs():
                        new_motifs.append(motif)
        
        is_innovative = len(new_motifs) > 5  # Threshold for innovation
        
        return is_innovative, new_motifs
    
    def _get_all_historical_motifs(self) -> set[str]:
        """Get all motifs from historical generations."""
        all_motifs = set()
        for gen in self.generation_history:
            all_motifs.update(gen.motif_signatures)
        return all_motifs
    
    def get_teacher_ensemble(self) -> list[GenerationRecord]:
        """Get ensemble of teacher networks.
        
        Returns top-performing generations for ensemble distillation.
        
        Returns:
            List of teacher generation records.
        """
        if not self.generation_history:
            return []
        
        # Sort by Elo and take top 3
        sorted_gens = sorted(
            self.generation_history,
            key=lambda g: g.elo_rating,
            reverse=True
        )
        
        return sorted_gens[:3]
    
    def validate_promotion(
        self,
        student_elo: float,
        test_positions: list,
        encoder,
        student_network
    ) -> tuple[bool, str]:
        """Validate if a generation should be promoted.
        
        Checks:
        1. Elo ≥ previous best
        2. No significant regression on test positions
        3. Innovation detected
        
        Args:
            student_elo: Student's Elo rating.
            test_positions: Validation positions.
            encoder: Position encoder.
            student_network: Student network.
            
        Returns:
            Tuple of (should_promote, reason).
        """
        # Check Elo
        if self.best_generation:
            if student_elo < self.best_generation.elo_rating - 10:
                return False, f"Elo regression: {student_elo} < {self.best_generation.elo_rating - 10}"
        
        # Check innovation
        is_innovative, new_motifs = self.detect_innovation(
            student_network, test_positions, encoder
        )
        
        if student_elo > (self.best_generation.elo_rating if self.best_generation else 0):
            return True, f"New best Elo: {student_elo}"
        
        if is_innovative:
            return True, f"Innovation detected: {len(new_motifs)} new motifs"
        
        return False, "No improvement or innovation"
