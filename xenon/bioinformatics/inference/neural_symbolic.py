"""Neural-symbolic inference with constraint regularization.

This module provides graph neural network (GNN) based inference with symbolic
constraint integration. Constraints dynamically regularize neural embeddings
during training.

Mathematical Foundation:
    Loss = L_task + λ * L_constraint

    Where:
    - L_task: Primary task loss (e.g., classification, regression)
    - L_constraint: Symbolic constraint violation penalty
    - λ: Constraint weight (configurable)

    Constraints can enforce:
    - Conservation laws (e.g., mass, energy, information)
    - Physical bounds (e.g., non-negativity, upper limits)
    - Structural relationships (e.g., pathway topology)

Determinism:
    All random operations (weight initialization, dropout, data shuffling)
    are seeded via the global SeedManager for reproducibility.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Try new package name first, fallback to old for compatibility
try:
    from qratum.common.seeding import SeedManager
except ImportError:
    from quasim.common.seeding import SeedManager

# Optional PyTorch import with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Neural-symbolic inference will use classical fallback.",
        UserWarning,
    )


class ConstraintType(Enum):
    """Types of symbolic constraints."""

    NON_NEGATIVE = "non_negative"
    UPPER_BOUND = "upper_bound"
    CONSERVATION = "conservation"
    SYMMETRY = "symmetry"
    MONOTONICITY = "monotonicity"


@dataclass
class GraphEmbedding:
    """Graph embedding result.

    Attributes:
        node_embeddings: Embedding vectors for nodes (shape: [n_nodes, embedding_dim])
        edge_embeddings: Optional embedding vectors for edges
        graph_embedding: Global graph embedding (shape: [embedding_dim])
        attention_weights: Optional attention weights from GNN layers
    """

    node_embeddings: np.ndarray
    edge_embeddings: Optional[np.ndarray] = None
    graph_embedding: Optional[np.ndarray] = None
    attention_weights: Optional[Dict[str, np.ndarray]] = None


@dataclass
class InferenceResult:
    """Result of neural-symbolic inference.

    Attributes:
        predictions: Model predictions
        embeddings: Graph embeddings
        constraint_violations: Constraint violation metrics
        confidence: Prediction confidence scores
        backend: Backend used ("pytorch" or "classical")
    """

    predictions: np.ndarray
    embeddings: GraphEmbedding
    constraint_violations: Dict[str, float]
    confidence: Optional[np.ndarray] = None
    backend: str = "classical"


if TORCH_AVAILABLE:

    class GNNLayer(nn.Module):
        """Graph Neural Network layer with message passing.

        Implements a simplified Graph Attention Network (GAT) layer:
            h_i' = σ(Σ_j α_ij W h_j)

        Where:
        - h_i: Node i embedding
        - α_ij: Attention coefficient between nodes i and j
        - W: Learnable weight matrix
        - σ: Activation function
        """

        def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
            """Initialize GNN layer.

            Args:
                in_features: Input feature dimension
                out_features: Output feature dimension
                num_heads: Number of attention heads
            """

            super().__init__()
            self.num_heads = num_heads
            self.out_features = out_features

            # Multi-head attention
            self.linear = nn.Linear(in_features, out_features * num_heads)
            self.attention = nn.Linear(2 * out_features, 1)

        def forward(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass with message passing.

            Args:
                node_features: Node feature matrix [n_nodes, in_features]
                edge_index: Edge connectivity [2, n_edges]

            Returns:
                Updated node features and attention weights
            """

            # Linear transformation
            h = self.linear(node_features)  # [n_nodes, out_features * num_heads]
            h = h.view(-1, self.num_heads, self.out_features)  # [n_nodes, num_heads, out_features]

            # Compute attention coefficients
            edge_src, edge_dst = edge_index[0], edge_index[1]
            h_src = h[edge_src]  # [n_edges, num_heads, out_features]
            h_dst = h[edge_dst]  # [n_edges, num_heads, out_features]

            # Concatenate source and destination for attention
            h_cat = torch.cat([h_src, h_dst], dim=-1)  # [n_edges, num_heads, 2*out_features]

            # Compute attention scores
            attention_scores = self.attention(h_cat).squeeze(-1)  # [n_edges, num_heads]
            attention_weights = F.softmax(attention_scores, dim=0)  # Normalize per head

            # Aggregate messages
            messages = h_src * attention_weights.unsqueeze(-1)  # [n_edges, num_heads, out_features]

            # Sum messages per node
            aggregated = torch.zeros_like(h)
            for i in range(edge_dst.shape[0]):
                dst_node = edge_dst[i]
                aggregated[dst_node] += messages[i]

            # Average across heads
            output = aggregated.mean(dim=1)  # [n_nodes, out_features]

            return F.relu(output), attention_weights.mean(dim=1)

    class NeuralSymbolicModel(nn.Module):
        """Neural-symbolic model with GNN and constraint regularization."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 3,
            num_heads: int = 4,
        ):
            """Initialize neural-symbolic model.

            Args:
                input_dim: Input feature dimension
                hidden_dim: Hidden layer dimension
                output_dim: Output dimension
                num_layers: Number of GNN layers
                num_heads: Number of attention heads per layer
            """

            super().__init__()

            self.num_layers = num_layers

            # GNN layers
            self.layers = nn.ModuleList()
            self.layers.append(GNNLayer(input_dim, hidden_dim, num_heads))
            for _ in range(num_layers - 1):
                self.layers.append(GNNLayer(hidden_dim, hidden_dim, num_heads))

            # Output layer
            self.output_layer = nn.Linear(hidden_dim, output_dim)

            # Global pooling
            self.pool = nn.Linear(hidden_dim, hidden_dim)

        def forward(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                node_features: Node feature matrix
                edge_index: Edge connectivity

            Returns:
                Node embeddings, graph embedding, predictions
            """

            h = node_features
            attention_weights_list = []

            # Apply GNN layers
            for layer in self.layers:
                h, attention_weights = layer(h, edge_index)
                attention_weights_list.append(attention_weights)

            # Global pooling for graph-level embedding
            graph_emb = self.pool(h.mean(dim=0, keepdim=True))

            # Predictions
            predictions = self.output_layer(h)

            return h, graph_emb, predictions


class NeuralSymbolicEngine:
    """Neural-symbolic inference engine with constraint regularization.

    Provides GNN-based inference with symbolic constraint integration.
    All operations are deterministic via seed management.

    Attributes:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of GNN layers
        constraint_weight: Weight for constraint regularization
        seed_manager: Deterministic seed management
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        num_layers: int = 3,
        constraint_weight: float = 0.1,
        seed: int = 42,
    ):
        """Initialize neural-symbolic engine.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GNN layers
            constraint_weight: Weight for constraint regularization (λ)
            seed: Random seed for reproducibility
        """

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.constraint_weight = constraint_weight
        self.seed_manager = SeedManager(seed)

        # SeedManager already sets global seed in __init__

        # Initialize model
        if TORCH_AVAILABLE:
            # Set PyTorch-specific seeds
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            self.model = NeuralSymbolicModel(input_dim, hidden_dim, output_dim, num_layers)
            self.optimizer = Adam(self.model.parameters(), lr=0.001)
            self.backend = "pytorch"
        else:
            self.model = None
            self.backend = "classical"

        # Constraint storage
        self.constraints: List[Tuple[ConstraintType, Callable]] = []

    def add_constraint(
        self,
        constraint_type: ConstraintType,
        constraint_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Add a symbolic constraint to the model.

        Constraints are applied during training to regularize embeddings.

        Args:
            constraint_type: Type of constraint
            constraint_fn: Function that computes constraint violation
                          Should return a scalar loss value

        Example:
            >>> def non_negative_constraint(embeddings):
            ...     return F.relu(-embeddings).sum()  # Penalize negative values
            >>> engine.add_constraint(ConstraintType.NON_NEGATIVE, non_negative_constraint)
        """

        self.constraints.append((constraint_type, constraint_fn))

    def _compute_constraint_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute total constraint violation loss.

        Args:
            embeddings: Node embeddings

        Returns:
            Total constraint loss
        """

        if not TORCH_AVAILABLE:
            return 0.0

        total_loss = torch.tensor(0.0, device=embeddings.device)

        for constraint_type, constraint_fn in self.constraints:
            violation = constraint_fn(embeddings)
            total_loss += violation

        return total_loss

    def train_step(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Perform one training step.

        Args:
            node_features: Node feature matrix [n_nodes, input_dim]
            edge_index: Edge connectivity [2, n_edges]
            labels: Target labels [n_nodes, output_dim]

        Returns:
            Dictionary of loss components
        """

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Cannot train neural model.")

        # Convert to tensors
        node_features_t = torch.FloatTensor(node_features)
        edge_index_t = torch.LongTensor(edge_index)
        labels_t = torch.FloatTensor(labels)

        # Forward pass
        self.model.train()
        embeddings, graph_emb, predictions = self.model(node_features_t, edge_index_t)

        # Task loss
        task_loss = F.mse_loss(predictions, labels_t)

        # Constraint loss
        constraint_loss = self._compute_constraint_loss(embeddings)

        # Total loss
        total_loss = task_loss + self.constraint_weight * constraint_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "constraint_loss": constraint_loss.item(),
        }

    def infer(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> InferenceResult:
        """Perform inference on a graph.

        Args:
            node_features: Node feature matrix [n_nodes, input_dim]
            edge_index: Edge connectivity [2, n_edges]

        Returns:
            Inference result with predictions and embeddings
        """

        if TORCH_AVAILABLE and self.model is not None:
            return self._infer_neural(node_features, edge_index)
        else:
            return self._infer_classical(node_features, edge_index)

    def _infer_neural(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> InferenceResult:
        """Neural inference with PyTorch."""

        # Convert to tensors
        node_features_t = torch.FloatTensor(node_features)
        edge_index_t = torch.LongTensor(edge_index)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            embeddings, graph_emb, predictions = self.model(node_features_t, edge_index_t)

        # Compute constraint violations
        violations = {}
        for constraint_type, constraint_fn in self.constraints:
            violation = constraint_fn(embeddings)
            violations[constraint_type.value] = violation.item()

        # Convert to numpy
        node_embeddings_np = embeddings.cpu().numpy()
        graph_emb_np = graph_emb.cpu().numpy().squeeze(0) if graph_emb is not None else None
        predictions_np = predictions.cpu().numpy()

        # Create embedding object
        embedding_obj = GraphEmbedding(
            node_embeddings=node_embeddings_np,
            graph_embedding=graph_emb_np,
        )

        return InferenceResult(
            predictions=predictions_np,
            embeddings=embedding_obj,
            constraint_violations=violations,
            backend="pytorch",
        )

    def _infer_classical(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> InferenceResult:
        """Classical fallback inference using simple aggregation."""

        # Simple message passing aggregation
        n_nodes = node_features.shape[0]
        in_features = node_features.shape[1]

        # Pad or truncate input to hidden_dim
        if in_features < self.hidden_dim:
            node_embeddings = np.pad(
                node_features, ((0, 0), (0, self.hidden_dim - in_features)), mode="constant"
            )
        else:
            node_embeddings = node_features[:, : self.hidden_dim].copy()

        # Aggregate neighbor features
        aggregated = np.zeros((n_nodes, self.hidden_dim))
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < n_nodes and dst < n_nodes:
                # Simple average aggregation
                aggregated[dst] += node_embeddings[src]

        # Combine original and aggregated
        node_embeddings = (node_embeddings + aggregated) / 2.0

        # Simple linear projection for predictions
        if self.output_dim <= self.hidden_dim:
            predictions = node_embeddings[:, : self.output_dim]
        else:
            # Pad predictions if output_dim > hidden_dim
            predictions = np.pad(
                node_embeddings, ((0, 0), (0, self.output_dim - self.hidden_dim)), mode="constant"
            )

        # Graph embedding as mean
        graph_emb = node_embeddings.mean(axis=0)

        # Create embedding object
        embedding_obj = GraphEmbedding(
            node_embeddings=node_embeddings,
            graph_embedding=graph_emb,
        )

        return InferenceResult(
            predictions=predictions,
            embeddings=embedding_obj,
            constraint_violations={},
            backend="classical",
        )

    def validate_equivalence(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        tolerance: float = 1e-6,
    ) -> Tuple[bool, float]:
        """Validate classical-neural equivalence (when applicable).

        Args:
            node_features: Node feature matrix
            edge_index: Edge connectivity
            tolerance: Maximum allowed difference

        Returns:
            (is_equivalent, difference)
        """

        if not TORCH_AVAILABLE:
            # No neural backend, always equivalent to itself
            return True, 0.0

        # Run both backends
        result_neural = self._infer_neural(node_features, edge_index)
        result_classical = self._infer_classical(node_features, edge_index)

        # Compare predictions
        diff = np.linalg.norm(result_neural.predictions - result_classical.predictions)
        is_equivalent = diff < tolerance

        if not is_equivalent:
            warnings.warn(
                f"Neural-classical equivalence violated: diff={diff:.2e} > {tolerance:.2e}",
                UserWarning,
            )

        return is_equivalent, diff
