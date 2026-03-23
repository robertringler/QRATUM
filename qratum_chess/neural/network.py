"""Neural evaluation network architecture.

Implements the dual-head residual policy-value network for chess evaluation.
Supports both standard evaluation and the Tri-Modal Cognitive Core architecture
described in Stage III.

Network Architecture:
- Initial Conv2D 3×3, 256 filters
- 20-40 Residual Blocks (configurable)
- Policy Head: Conv2D 1×1 → FC → Softmax
- Value Head: Conv2D 1×1 → FC → tanh

The network can operate in three modes:
1. Standard: Single policy-value network
2. NNUE-style: Fast tactical evaluator
3. Tri-Modal: Combined Tactical, Strategic, and Conceptual cortices
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture.

    Attributes:
        num_input_channels: Number of input tensor channels.
        num_filters: Number of convolutional filters.
        num_residual_blocks: Number of residual blocks in trunk.
        policy_head_filters: Filters in policy head convolution.
        policy_fc_size: Size of policy head FC layer.
        value_head_filters: Filters in value head convolution.
        value_fc_size: Size of value head FC layers.
        num_moves: Size of policy output (legal move space).
        dropout_rate: Dropout rate for regularization.
    """

    num_input_channels: int = 28
    num_filters: int = 256
    num_residual_blocks: int = 20
    policy_head_filters: int = 32
    policy_fc_size: int = 2048
    value_head_filters: int = 8
    value_fc_size: int = 256
    num_moves: int = 4672
    dropout_rate: float = 0.1


class ResidualBlock:
    """Residual block with skip connection.

    Structure: Conv → BN → ReLU → Conv → BN → Add(skip) → ReLU
    """

    def __init__(self, num_filters: int):
        """Initialize residual block.

        Args:
            num_filters: Number of convolutional filters.
        """
        self.num_filters = num_filters
        # Weight matrices (simplified - would use proper convolution in full impl)
        self.conv1_weights = np.random.randn(num_filters, num_filters, 3, 3) * 0.01
        self.conv2_weights = np.random.randn(num_filters, num_filters, 3, 3) * 0.01
        self.bn1_gamma = np.ones(num_filters)
        self.bn1_beta = np.zeros(num_filters)
        self.bn2_gamma = np.ones(num_filters)
        self.bn2_beta = np.zeros(num_filters)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through residual block.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Output tensor of same shape.
        """
        residual = x

        # Conv1 → BN → ReLU
        out = self._conv2d(x, self.conv1_weights)
        out = self._batch_norm(out, self.bn1_gamma, self.bn1_beta)
        out = self._relu(out)

        # Conv2 → BN
        out = self._conv2d(out, self.conv2_weights)
        out = self._batch_norm(out, self.bn2_gamma, self.bn2_beta)

        # Skip connection
        out = out + residual

        # Final ReLU
        out = self._relu(out)

        return out

    def _conv2d(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Simplified 3×3 convolution with same padding."""
        batch, channels, h, w = x.shape
        out_channels = weights.shape[0]

        # Pad input
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")

        # Simple convolution (vectorized version would be faster)
        output = np.zeros((batch, out_channels, h, w))
        for b in range(batch):
            for oc in range(out_channels):
                for i in range(h):
                    for j in range(w):
                        patch = padded[b, :, i : i + 3, j : j + 3]
                        output[b, oc, i, j] = np.sum(patch * weights[oc])

        return output

    def _batch_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Simplified batch normalization."""
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)


class PolicyValueNetwork:
    """Dual-head policy-value neural network.

    Architecture follows AlphaZero-style residual network with
    separate policy and value heads.
    """

    def __init__(self, config: NetworkConfig | None = None):
        """Initialize the network.

        Args:
            config: Network configuration. Uses defaults if None.
        """
        self.config = config or NetworkConfig()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/He initialization."""
        cfg = self.config

        # Initial convolution
        self.input_conv = np.random.randn(cfg.num_filters, cfg.num_input_channels, 3, 3) * np.sqrt(
            2.0 / (cfg.num_input_channels * 9)
        )
        self.input_bn_gamma = np.ones(cfg.num_filters)
        self.input_bn_beta = np.zeros(cfg.num_filters)

        # Residual blocks
        self.residual_blocks = [
            ResidualBlock(cfg.num_filters) for _ in range(cfg.num_residual_blocks)
        ]

        # Policy head
        self.policy_conv = np.random.randn(
            cfg.policy_head_filters, cfg.num_filters, 1, 1
        ) * np.sqrt(2.0 / cfg.num_filters)
        self.policy_bn_gamma = np.ones(cfg.policy_head_filters)
        self.policy_bn_beta = np.zeros(cfg.policy_head_filters)
        self.policy_fc1 = np.random.randn(
            cfg.policy_head_filters * 64, cfg.policy_fc_size
        ) * np.sqrt(2.0 / (cfg.policy_head_filters * 64))
        self.policy_fc2 = np.random.randn(cfg.policy_fc_size, cfg.num_moves) * np.sqrt(
            2.0 / cfg.policy_fc_size
        )

        # Value head
        self.value_conv = np.random.randn(cfg.value_head_filters, cfg.num_filters, 1, 1) * np.sqrt(
            2.0 / cfg.num_filters
        )
        self.value_bn_gamma = np.ones(cfg.value_head_filters)
        self.value_bn_beta = np.zeros(cfg.value_head_filters)
        self.value_fc1 = np.random.randn(cfg.value_head_filters * 64, cfg.value_fc_size) * np.sqrt(
            2.0 / (cfg.value_head_filters * 64)
        )
        self.value_fc2 = np.random.randn(cfg.value_fc_size, 1) * np.sqrt(2.0 / cfg.value_fc_size)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, 8, 8).

        Returns:
            Tuple of (policy logits, value) where:
            - policy: Shape (batch, num_moves)
            - value: Shape (batch, 1)
        """
        # Input convolution
        out = self._conv2d(x, self.input_conv)
        out = self._batch_norm(out, self.input_bn_gamma, self.input_bn_beta)
        out = self._relu(out)

        # Residual tower
        for block in self.residual_blocks:
            out = block.forward(out)

        # Policy head
        policy = self._policy_head(out)

        # Value head
        value = self._value_head(out)

        return policy, value

    def _policy_head(self, x: np.ndarray) -> np.ndarray:
        """Compute policy output."""
        batch = x.shape[0]

        # Conv 1×1
        out = self._conv2d_1x1(x, self.policy_conv)
        out = self._batch_norm(out, self.policy_bn_gamma, self.policy_bn_beta)
        out = self._relu(out)

        # Flatten
        out = out.reshape(batch, -1)

        # FC layers
        out = self._relu(out @ self.policy_fc1)
        out = out @ self.policy_fc2

        # Softmax
        out = self._softmax(out)

        return out

    def _value_head(self, x: np.ndarray) -> np.ndarray:
        """Compute value output."""
        batch = x.shape[0]

        # Conv 1×1
        out = self._conv2d_1x1(x, self.value_conv)
        out = self._batch_norm(out, self.value_bn_gamma, self.value_bn_beta)
        out = self._relu(out)

        # Flatten
        out = out.reshape(batch, -1)

        # FC layers
        out = self._relu(out @ self.value_fc1)
        out = np.tanh(out @ self.value_fc2)

        return out

    def _conv2d(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """3×3 convolution with same padding."""
        batch, channels, h, w = x.shape
        out_channels = weights.shape[0]

        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        output = np.zeros((batch, out_channels, h, w))

        for b in range(batch):
            for oc in range(out_channels):
                for i in range(h):
                    for j in range(w):
                        patch = padded[b, :, i : i + 3, j : j + 3]
                        output[b, oc, i, j] = np.sum(patch * weights[oc])

        return output

    def _conv2d_1x1(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """1×1 convolution."""
        batch, channels, h, w = x.shape
        out_channels = weights.shape[0]

        # Reshape for efficient computation
        x_flat = x.reshape(batch, channels, h * w)
        w_flat = weights.reshape(out_channels, channels)

        output = np.einsum("bci,oc->boi", x_flat, w_flat)
        return output.reshape(batch, out_channels, h, w)

    def _batch_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Batch normalization."""
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def get_num_parameters(self) -> int:
        """Count total number of parameters."""
        count = 0
        count += self.input_conv.size
        count += self.input_bn_gamma.size + self.input_bn_beta.size

        for block in self.residual_blocks:
            count += block.conv1_weights.size + block.conv2_weights.size
            count += block.bn1_gamma.size + block.bn1_beta.size
            count += block.bn2_gamma.size + block.bn2_beta.size

        count += self.policy_conv.size
        count += self.policy_bn_gamma.size + self.policy_bn_beta.size
        count += self.policy_fc1.size + self.policy_fc2.size

        count += self.value_conv.size
        count += self.value_bn_gamma.size + self.value_bn_beta.size
        count += self.value_fc1.size + self.value_fc2.size

        return count


class NeuralEvaluator:
    """High-level interface for neural position evaluation.

    Wraps the PolicyValueNetwork with position encoding and move decoding.
    """

    def __init__(self, config: NetworkConfig | None = None):
        """Initialize the neural evaluator.

        Args:
            config: Network configuration. Uses defaults if None.
        """
        from qratum_chess.neural.encoding import PositionEncoder

        self.config = config or NetworkConfig()
        self.network = PolicyValueNetwork(self.config)
        self.encoder = PositionEncoder()

    def evaluate(self, position: Position) -> tuple[np.ndarray, float]:
        """Evaluate a position.

        Args:
            position: Chess position to evaluate.

        Returns:
            Tuple of (policy distribution, value estimate).
        """
        # Encode position
        input_tensor = self.encoder.encode(position)
        input_batch = input_tensor[np.newaxis, ...]  # Add batch dimension

        # Forward pass
        policy, value = self.network.forward(input_batch)

        return policy[0], float(value[0, 0])

    def evaluate_batch(self, positions: list[Position]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of positions.

        Args:
            positions: List of chess positions.

        Returns:
            Tuple of (policy distributions, value estimates).
        """
        # Encode all positions
        input_batch = self.encoder.encode_batch(positions)

        # Forward pass
        policy, value = self.network.forward(input_batch)

        return policy, value.squeeze(-1)

    def get_move_probabilities(
        self, position: Position, legal_moves: list[Move]
    ) -> dict[Move, float]:
        """Get probability distribution over legal moves.

        Args:
            position: Current position.
            legal_moves: List of legal moves.

        Returns:
            Dictionary mapping moves to probabilities.
        """
        from qratum_chess.neural.encoding import MOVE_ENCODER

        policy, _ = self.evaluate(position)

        # Get probabilities for legal moves only
        move_probs = {}
        total_prob = 0.0

        for move in legal_moves:
            promo = 0
            if move.promotion:
                promo_map = {1: 1, 2: 2, 3: 3, 4: 0}  # Knight=1, Bishop=2, Rook=3, Queen=0
                promo = promo_map.get(int(move.promotion), 0)

            idx = MOVE_ENCODER.encode_move(move.from_sq, move.to_sq, promo)
            if 0 <= idx < len(policy):
                prob = float(policy[idx])
                move_probs[move] = prob
                total_prob += prob

        # Normalize
        if total_prob > 0:
            for move in move_probs:
                move_probs[move] /= total_prob

        return move_probs
