"""Meta-Controller Kernel (MCK) for QuASIM Phase VIII.

Implements reinforcement learning-based autonomous parameter tuning
for Φ-valuation, compliance, and orchestration optimization.
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quasim.audit.log import audit_event


@dataclass
class MCKState:
    """State representation for Meta-Controller Kernel."""

    phi_variance: float
    compliance_score: float
    resource_utilization: float
    error_rate: float
    timestamp: str


@dataclass
class MCKAction:
    """Action representation for Meta-Controller Kernel."""

    parameter_name: str
    adjustment: float
    reason: str


class MetaControllerKernel:
    """Reinforcement learning-based meta-controller for autonomous tuning.

    The MCK monitors system telemetry and autonomously adjusts parameters
    to minimize Φ_QEVF variance while maintaining compliance constraints.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize the Meta-Controller Kernel.

        Parameters
        ----------
        learning_rate : float
            Learning rate for Q-learning updates
        discount_factor : float
            Discount factor for future rewards
        exploration_rate : float
            Epsilon for epsilon-greedy exploration
        seed : Optional[int]
            Random seed for deterministic replay
        """

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.seed = seed

        # Set random seed for deterministic behavior
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Q-table: state -> action -> value
        self.q_table: Dict[str, Dict[str, float]] = {}

        # Experience replay buffer
        self.experience_buffer: List[Tuple[MCKState, MCKAction, float, MCKState]] = []

        # Performance history
        self.performance_history: List[Dict[str, Any]] = []

        # Current state
        self.current_state: Optional[MCKState] = None

    def observe_state(
        self,
        phi_variance: float,
        compliance_score: float,
        resource_utilization: float,
        error_rate: float,
    ) -> MCKState:
        """Observe current system state.

        Parameters
        ----------
        phi_variance : float
            Current Φ_QEVF variance
        compliance_score : float
            Current compliance score (0-100)
        resource_utilization : float
            Current resource utilization (0-1)
        error_rate : float
            Current error rate (0-1)

        Returns
        -------
        MCKState
            Current state representation
        """

        state = MCKState(
            phi_variance=phi_variance,
            compliance_score=compliance_score,
            resource_utilization=resource_utilization,
            error_rate=error_rate,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
        )

        self.current_state = state

        # Log state observation
        audit_event(
            "mck.state_observed",
            {
                "phi_variance": phi_variance,
                "compliance_score": compliance_score,
                "resource_utilization": resource_utilization,
                "error_rate": error_rate,
            },
        )

        return state

    def select_action(self, state: MCKState, epsilon: Optional[float] = None) -> MCKAction:
        """Select action using epsilon-greedy policy.

        Parameters
        ----------
        state : MCKState
            Current state
        epsilon : Optional[float]
            Override exploration rate

        Returns
        -------
        MCKAction
            Selected action
        """

        epsilon = epsilon if epsilon is not None else self.exploration_rate

        # Discretize state for Q-table lookup
        state_key = self._discretize_state(state)

        # Initialize Q-values for unseen states
        if state_key not in self.q_table:
            self.q_table[state_key] = self._initialize_q_values()

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Explore: random action
            action = self._random_action()
            reason = "exploration"
        else:
            # Exploit: best known action
            action = self._best_action(state_key)
            reason = "exploitation"

        action.reason = reason

        # Log action selection
        audit_event(
            "mck.action_selected",
            {
                "parameter": action.parameter_name,
                "adjustment": action.adjustment,
                "reason": reason,
            },
        )

        return action

    def update_q_value(
        self,
        state: MCKState,
        action: MCKAction,
        reward: float,
        next_state: MCKState,
    ) -> None:
        """Update Q-value using Q-learning rule.

        Parameters
        ----------
        state : MCKState
            Previous state
        action : MCKAction
            Action taken
        reward : float
            Reward received
        next_state : MCKState
            Resulting state
        """

        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        action_key = f"{action.parameter_name}:{action.adjustment:.2f}"

        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = self._initialize_q_values()
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = self._initialize_q_values()

        # Q-learning update
        current_q = self.q_table[state_key].get(action_key, 0.0)
        max_next_q = (
            max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        )

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action_key] = new_q

        # Store experience
        self.experience_buffer.append((state, action, reward, next_state))

        # Log update
        audit_event(
            "mck.q_value_updated",
            {
                "state_key": state_key,
                "action_key": action_key,
                "reward": reward,
                "new_q_value": new_q,
            },
        )

    def compute_reward(self, prev_state: MCKState, new_state: MCKState) -> float:
        """Compute reward based on state transition.

        Reward function prioritizes:
        1. Minimizing Φ_QEVF variance
        2. Maintaining high compliance
        3. Efficient resource utilization
        4. Low error rate

        Parameters
        ----------
        prev_state : MCKState
            Previous state
        new_state : MCKState
            New state

        Returns
        -------
        float
            Computed reward
        """

        # Variance reduction (primary objective)
        variance_improvement = prev_state.phi_variance - new_state.phi_variance
        variance_reward = variance_improvement * 10.0

        # Compliance maintenance (must stay above threshold)
        compliance_penalty = 0.0
        if new_state.compliance_score < 95.0:
            compliance_penalty = (95.0 - new_state.compliance_score) * 2.0

        # Resource efficiency
        resource_reward = 0.0
        if 0.4 <= new_state.resource_utilization <= 0.8:
            resource_reward = 1.0
        elif new_state.resource_utilization > 0.9:
            resource_reward = -2.0

        # Error rate penalty
        error_penalty = new_state.error_rate * 5.0

        # Total reward
        reward = variance_reward - compliance_penalty + resource_reward - error_penalty

        return reward

    def save_checkpoint(self, path: Path) -> None:
        """Save MCK checkpoint for reproducibility.

        Parameters
        ----------
        path : Path
            Checkpoint file path
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q_table": self.q_table,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "seed": self.seed,
            "performance_history": self.performance_history,
            "experience_buffer": [
                (
                    {
                        "phi_variance": s.phi_variance,
                        "compliance_score": s.compliance_score,
                        "resource_utilization": s.resource_utilization,
                        "error_rate": s.error_rate,
                        "timestamp": s.timestamp,
                    },
                    {
                        "parameter_name": a.parameter_name,
                        "adjustment": a.adjustment,
                        "reason": a.reason,
                    },
                    r,
                    {
                        "phi_variance": ns.phi_variance,
                        "compliance_score": ns.compliance_score,
                        "resource_utilization": ns.resource_utilization,
                        "error_rate": ns.error_rate,
                        "timestamp": ns.timestamp,
                    },
                )
                for s, a, r, ns in self.experience_buffer
            ],
        }

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        audit_event("mck.checkpoint_saved", {"path": str(path)})

    def load_checkpoint(self, path: Path) -> None:
        """Load MCK checkpoint for deterministic replay.

        Parameters
        ----------
        path : Path
            Checkpoint file path
        """

        with open(path) as f:
            checkpoint = json.load(f)

        self.q_table = checkpoint["q_table"]
        self.learning_rate = checkpoint["learning_rate"]
        self.discount_factor = checkpoint["discount_factor"]
        self.exploration_rate = checkpoint["exploration_rate"]
        self.seed = checkpoint["seed"]
        self.performance_history = checkpoint["performance_history"]

        # Restore experience buffer
        self.experience_buffer = [
            (
                MCKState(**s),
                MCKAction(**a),
                r,
                MCKState(**ns),
            )
            for s, a, r, ns in checkpoint["experience_buffer"]
        ]

        audit_event("mck.checkpoint_loaded", {"path": str(path)})

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.

        Returns
        -------
        Dict[str, Any]
            Performance metrics
        """

        if not self.experience_buffer:
            return {
                "episodes": 0,
                "avg_reward": 0.0,
                "phi_variance_reduction": 0.0,
                "compliance_maintained": True,
            }

        # Calculate metrics
        total_reward = sum(r for _, _, r, _ in self.experience_buffer)
        avg_reward = total_reward / len(self.experience_buffer)

        # Phi variance reduction
        initial_variance = self.experience_buffer[0][0].phi_variance
        final_variance = self.experience_buffer[-1][3].phi_variance
        variance_reduction = initial_variance - final_variance

        # Compliance check
        compliance_maintained = all(
            ns.compliance_score >= 95.0 for _, _, _, ns in self.experience_buffer
        )

        return {
            "episodes": len(self.experience_buffer),
            "avg_reward": avg_reward,
            "phi_variance_reduction": variance_reduction,
            "compliance_maintained": compliance_maintained,
            "q_table_size": len(self.q_table),
        }

    def _discretize_state(self, state: MCKState) -> str:
        """Discretize continuous state for Q-table."""

        phi_bin = int(state.phi_variance * 10)
        comp_bin = int(state.compliance_score / 10)
        util_bin = int(state.resource_utilization * 10)
        err_bin = int(state.error_rate * 10)
        return f"{phi_bin}:{comp_bin}:{util_bin}:{err_bin}"

    def _initialize_q_values(self) -> Dict[str, float]:
        """Initialize Q-values for a new state."""

        return {
            "phi_tolerance:0.05": 0.0,
            "phi_tolerance:0.10": 0.0,
            "phi_tolerance:-0.05": 0.0,
            "resource_limit:0.05": 0.0,
            "resource_limit:-0.05": 0.0,
            "batch_size:10": 0.0,
            "batch_size:-10": 0.0,
        }

    def _random_action(self) -> MCKAction:
        """Select random action for exploration."""

        actions = [
            ("phi_tolerance", 0.05),
            ("phi_tolerance", 0.10),
            ("phi_tolerance", -0.05),
            ("resource_limit", 0.05),
            ("resource_limit", -0.05),
            ("batch_size", 10),
            ("batch_size", -10),
        ]
        param, adj = random.choice(actions)
        return MCKAction(parameter_name=param, adjustment=adj, reason="exploration")

    def _best_action(self, state_key: str) -> MCKAction:
        """Select best action based on Q-values."""

        q_values = self.q_table[state_key]
        best_action_key = max(q_values, key=q_values.get)

        # Parse action key
        param, adj_str = best_action_key.split(":")
        adj = float(adj_str)

        return MCKAction(parameter_name=param, adjustment=adj, reason="exploitation")
