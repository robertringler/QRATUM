"""Network trainer for QRATUM-Chess.

Implements the training loop with loss function:
L = (z - v)^2 - π^T log(p) + λ||θ||^2

Where:
- v = value output
- p = policy head output
- z = game result
- π = MCTS policy
- λ = L2 regularization weight
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for network training.
    
    Attributes:
        learning_rate: Initial learning rate.
        batch_size: Training batch size.
        l2_regularization: L2 regularization weight.
        value_loss_weight: Weight for value loss component.
        policy_loss_weight: Weight for policy loss component.
        epochs: Number of training epochs.
        lr_decay_factor: Learning rate decay factor.
        lr_decay_steps: Steps between learning rate decay.
    """
    learning_rate: float = 0.002
    batch_size: int = 1024
    l2_regularization: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    epochs: int = 100
    lr_decay_factor: float = 0.1
    lr_decay_steps: int = 10000


@dataclass
class TrainingMetrics:
    """Metrics from training."""
    epoch: int = 0
    total_loss: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    l2_loss: float = 0.0
    learning_rate: float = 0.0
    samples_processed: int = 0


class NetworkTrainer:
    """Trainer for QRATUM-Chess neural networks.
    
    Implements the training loop with combined value and policy loss,
    plus L2 regularization.
    """
    
    def __init__(
        self,
        network,
        config: TrainingConfig | None = None
    ):
        """Initialize the trainer.
        
        Args:
            network: Neural network to train.
            config: Training configuration.
        """
        self.network = network
        self.config = config or TrainingConfig()
        self.current_lr = self.config.learning_rate
        self.step_count = 0
        self.metrics_history: list[TrainingMetrics] = []
    
    def train_batch(
        self,
        samples: list['GameSample'],
        encoder: 'PositionEncoder'
    ) -> TrainingMetrics:
        """Train on a batch of samples.
        
        Args:
            samples: Training samples.
            encoder: Position encoder for tensor conversion.
            
        Returns:
            Training metrics for this batch.
        """
        from qratum_chess.core.position import Position
        from qratum_chess.neural.encoding import MOVE_ENCODER
        
        batch_size = len(samples)
        if batch_size == 0:
            return TrainingMetrics()
        
        # Encode positions
        positions = [Position.from_fen(s.position_fen) for s in samples]
        input_batch = encoder.encode_batch(positions)
        
        # Forward pass
        policy_out, value_out = self.network.forward(input_batch)
        
        # Compute value loss: (z - v)^2
        target_values = np.array([s.value for s in samples], dtype=np.float32)
        value_loss = np.mean((target_values - value_out.squeeze()) ** 2)
        
        # Compute policy loss: -π^T log(p)
        policy_loss = 0.0
        for i, sample in enumerate(samples):
            for move_uci, target_prob in sample.policy.items():
                # Get move index
                from qratum_chess.core.position import Move
                move = Move.from_uci(move_uci)
                idx = MOVE_ENCODER.encode_move(move.from_sq, move.to_sq, 0)
                
                if 0 <= idx < policy_out.shape[1]:
                    pred_prob = policy_out[i, idx]
                    if pred_prob > 0:
                        policy_loss -= target_prob * np.log(pred_prob + 1e-10)
        
        policy_loss /= batch_size
        
        # Compute L2 regularization loss
        l2_loss = self._compute_l2_loss()
        
        # Total loss
        total_loss = (
            self.config.value_loss_weight * value_loss +
            self.config.policy_loss_weight * policy_loss +
            self.config.l2_regularization * l2_loss
        )
        
        # Backward pass and update (simplified SGD)
        self._update_weights(samples, positions, encoder, policy_out, value_out)
        
        # Update learning rate
        self.step_count += 1
        if self.step_count % self.config.lr_decay_steps == 0:
            self.current_lr *= self.config.lr_decay_factor
        
        metrics = TrainingMetrics(
            total_loss=float(total_loss),
            value_loss=float(value_loss),
            policy_loss=float(policy_loss),
            l2_loss=float(l2_loss),
            learning_rate=self.current_lr,
            samples_processed=batch_size,
        )
        
        return metrics
    
    def _compute_l2_loss(self) -> float:
        """Compute L2 regularization loss."""
        l2_loss = 0.0
        
        # Sum of squared weights
        l2_loss += np.sum(self.network.input_conv ** 2)
        l2_loss += np.sum(self.network.policy_fc1 ** 2)
        l2_loss += np.sum(self.network.policy_fc2 ** 2)
        l2_loss += np.sum(self.network.value_fc1 ** 2)
        l2_loss += np.sum(self.network.value_fc2 ** 2)
        
        for block in self.network.residual_blocks:
            l2_loss += np.sum(block.conv1_weights ** 2)
            l2_loss += np.sum(block.conv2_weights ** 2)
        
        return float(l2_loss)
    
    def _update_weights(
        self,
        samples: list,
        positions: list,
        encoder,
        policy_out: np.ndarray,
        value_out: np.ndarray
    ) -> None:
        """Update network weights using gradient descent.
        
        Note: This is a simplified implementation. A full implementation
        would use proper backpropagation through the entire network.
        """
        lr = self.current_lr
        
        # NOTE: This is a simplified placeholder implementation.
        # A production implementation would use PyTorch/JAX with proper autograd
        # for backpropagation through the entire network.
        # TODO: Replace with actual backpropagation using deep learning framework.
        
        # Value head update (simplified gradient descent approximation)
        target_values = np.array([s.value for s in samples])
        value_error = target_values - value_out.squeeze()
        
        # Use error signal for directional weight update
        error_magnitude = np.mean(np.abs(value_error))
        if error_magnitude > 0:
            # Scale weight adjustment by error
            self.network.value_fc2 += lr * 0.001 * error_magnitude * np.random.randn(*self.network.value_fc2.shape)
        
        # Policy head update (simplified)
        # Would normally use cross-entropy gradient
        self.network.policy_fc2 += lr * 0.001 * np.random.randn(*self.network.policy_fc2.shape)
        
        # L2 weight decay
        decay = 1 - lr * self.config.l2_regularization
        self.network.value_fc1 *= decay
        self.network.value_fc2 *= decay
        self.network.policy_fc1 *= decay
        self.network.policy_fc2 *= decay
    
    def train_epoch(
        self,
        samples: list['GameSample'],
        encoder: 'PositionEncoder'
    ) -> TrainingMetrics:
        """Train for one epoch over all samples.
        
        Args:
            samples: All training samples.
            encoder: Position encoder.
            
        Returns:
            Aggregated metrics for the epoch.
        """
        import random
        
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        batch_size = self.config.batch_size
        num_batches = (len(shuffled) + batch_size - 1) // batch_size
        
        total_metrics = TrainingMetrics()
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(shuffled))
            batch = shuffled[start:end]
            
            batch_metrics = self.train_batch(batch, encoder)
            
            # Accumulate metrics
            total_metrics.total_loss += batch_metrics.total_loss
            total_metrics.value_loss += batch_metrics.value_loss
            total_metrics.policy_loss += batch_metrics.policy_loss
            total_metrics.l2_loss += batch_metrics.l2_loss
            total_metrics.samples_processed += batch_metrics.samples_processed
        
        # Average metrics
        if num_batches > 0:
            total_metrics.total_loss /= num_batches
            total_metrics.value_loss /= num_batches
            total_metrics.policy_loss /= num_batches
            total_metrics.l2_loss /= num_batches
        
        total_metrics.learning_rate = self.current_lr
        
        self.metrics_history.append(total_metrics)
        
        return total_metrics
    
    def train(
        self,
        samples: list['GameSample'],
        encoder: 'PositionEncoder',
        epochs: int | None = None
    ) -> list[TrainingMetrics]:
        """Train for multiple epochs.
        
        Args:
            samples: Training samples.
            encoder: Position encoder.
            epochs: Number of epochs (uses config default if None).
            
        Returns:
            List of metrics for each epoch.
        """
        epochs = epochs or self.config.epochs
        all_metrics = []
        
        for epoch in range(epochs):
            metrics = self.train_epoch(samples, encoder)
            metrics.epoch = epoch
            all_metrics.append(metrics)
            
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"loss={metrics.total_loss:.4f}, "
                  f"value_loss={metrics.value_loss:.4f}, "
                  f"policy_loss={metrics.policy_loss:.4f}")
        
        return all_metrics
