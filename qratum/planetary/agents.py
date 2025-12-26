"""
Multi-Agent Simulation Module

Defines planetary-scale agent types for infrastructure operations:
- Node Agents: Manage physical infrastructure
- Data Validator Agents: Validate data contracts
- AI Governance Agents: Autonomous decision-making
- Economic Agents: Manage tokenized flows
- Integration Agents: Cross-domain interoperability
- Simulation/Optimization Agents: Predictive modeling
- Security Agents: Threat detection and response

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Protocol


class AgentType(Enum):
    """Types of planetary agents."""

    NODE = "node"
    DATA_VALIDATOR = "data_validator"
    AI_GOVERNANCE = "ai_governance"
    ECONOMIC = "economic"
    INTEGRATION = "integration"
    SIMULATION = "simulation"
    SECURITY = "security"


class AgentStatus(Enum):
    """Agent operational status."""

    IDLE = "idle"
    ACTIVE = "active"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class AgentObservation:
    """Observation from the environment for agent decision-making.

    Attributes:
        tick: Simulation tick number
        view: Current environmental view
        provenance: Source of observation
    """

    tick: int
    view: dict[str, Any]
    provenance: str = ""


@dataclass
class AgentState:
    """Mutable agent state tracked during operations.

    Attributes:
        agent_id: Unique agent identifier
        memory: Persistent agent memory
        rewards: Accumulated rewards
        actions: History of actions taken
        observations: History of observations
    """

    agent_id: str
    memory: dict[str, Any] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[AgentObservation] = field(default_factory=list)

    def remember(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        self.memory[key] = value

    def add_reward(self, reward: float) -> float:
        """Add a reward and return cumulative reward."""
        self.rewards.append(reward)
        return sum(self.rewards)

    def record_action(self, action: dict[str, Any]) -> None:
        """Record an action taken."""
        self.actions.append(action)

    def record_observation(self, observation: AgentObservation) -> None:
        """Record an observation."""
        self.observations.append(observation)


class AgentPolicy(Protocol):
    """Protocol for agent decision policies."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Make a decision based on observation and state."""
        ...


@dataclass
class PlanetaryAgent:
    """Base class for all planetary agents.

    Attributes:
        agent_id: Unique agent identifier
        agent_type: Type of agent
        policy: Decision-making policy
        state: Agent state
        status: Current operational status
        created_at: Agent creation timestamp
        metadata: Additional agent metadata
    """

    agent_id: str
    agent_type: AgentType
    policy: AgentPolicy | None = None
    state: AgentState = field(init=False)
    status: AgentStatus = AgentStatus.IDLE
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize agent state."""
        self.state = AgentState(agent_id=self.agent_id)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def observe(self, observation: AgentObservation) -> None:
        """Process an observation."""
        self.state.record_observation(observation)
        self.state.remember("last_view", observation.view)

    def act(self, observation: AgentObservation) -> dict[str, Any]:
        """Take an action based on observation."""
        self.observe(observation)
        self.status = AgentStatus.PROCESSING

        if self.policy:
            action = self.policy.decide(observation, self.state)
        else:
            action = self._default_action(observation)

        self.state.record_action(action)
        self.status = AgentStatus.IDLE
        return action

    def _default_action(self, observation: AgentObservation) -> dict[str, Any]:
        """Default action when no policy is set."""
        return {
            "action_type": "noop",
            "agent_id": self.agent_id,
            "tick": observation.tick,
        }

    def reward(self, value: float) -> float:
        """Add a reward to the agent."""
        return self.state.add_reward(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "memory_size": len(self.state.memory),
            "total_rewards": sum(self.state.rewards),
            "actions_taken": len(self.state.actions),
            "metadata": self.metadata,
        }


class NodeAgentPolicy:
    """Policy for node management agents."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Decide on node management actions."""
        view = observation.view
        node_status = view.get("node_status", "unknown")
        load = view.get("load_percent", 0)

        action_type = "monitor"
        if load > 90:
            action_type = "scale_up"
        elif load < 20:
            action_type = "scale_down"
        elif node_status == "degraded":
            action_type = "repair"

        return {
            "action_type": action_type,
            "agent_id": state.agent_id,
            "tick": observation.tick,
            "target_load": 50,
            "priority": min(1.0, load / 100),
        }


class NodeAgent(PlanetaryAgent):
    """Agent for managing physical infrastructure nodes.

    Handles scaling, maintenance, and health monitoring of nodes.
    """

    def __init__(
        self,
        agent_id: str,
        managed_nodes: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.NODE,
            policy=NodeAgentPolicy(),
            **kwargs,
        )
        self.managed_nodes = managed_nodes or []

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["managed_nodes"] = self.managed_nodes
        return base


class DataValidatorPolicy:
    """Policy for data validation agents."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Decide on data validation actions."""
        view = observation.view
        data_hash = view.get("data_hash", "")
        schema_valid = view.get("schema_valid", False)

        if not schema_valid:
            return {
                "action_type": "reject",
                "agent_id": state.agent_id,
                "tick": observation.tick,
                "reason": "schema_invalid",
            }

        return {
            "action_type": "validate",
            "agent_id": state.agent_id,
            "tick": observation.tick,
            "data_hash": data_hash,
            "vote": True,
        }


class DataValidatorAgent(PlanetaryAgent):
    """Agent for validating data contracts through Proof-of-Data consensus.

    Participates in consensus voting for data validation.
    """

    def __init__(
        self,
        agent_id: str,
        validation_contracts: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.DATA_VALIDATOR,
            policy=DataValidatorPolicy(),
            **kwargs,
        )
        self.validation_contracts = validation_contracts or []

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["validation_contracts"] = self.validation_contracts
        return base


class AIGovernancePolicy:
    """Policy for AI governance agents."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Decide on governance actions."""
        view = observation.view
        proposal = view.get("proposal", {})
        risk_score = view.get("risk_score", 0.5)

        if risk_score > 0.8:
            action_type = "reject"
            reason = "high_risk"
        elif risk_score < 0.2:
            action_type = "approve"
            reason = "low_risk"
        else:
            action_type = "escalate"
            reason = "requires_review"

        return {
            "action_type": action_type,
            "agent_id": state.agent_id,
            "tick": observation.tick,
            "proposal_id": proposal.get("id", ""),
            "reason": reason,
            "risk_score": risk_score,
        }


class AIGovernanceAgent(PlanetaryAgent):
    """Agent for autonomous AI governance decisions.

    Makes real-time decisions on policy proposals and system operations.
    """

    def __init__(
        self,
        agent_id: str,
        governance_scope: list[str] | None = None,
        decision_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.AI_GOVERNANCE,
            policy=AIGovernancePolicy(),
            **kwargs,
        )
        self.governance_scope = governance_scope or ["policy", "operations"]
        self.decision_threshold = decision_threshold

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["governance_scope"] = self.governance_scope
        base["decision_threshold"] = self.decision_threshold
        return base


class EconomicPolicy:
    """Policy for economic agents."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Decide on economic actions."""
        view = observation.view
        token_balance = view.get("token_balance", 0)
        market_price = view.get("market_price", 1.0)
        demand = view.get("demand", 0.5)

        if demand > 0.8 and token_balance > 1000:
            action_type = "sell"
            amount = int(token_balance * 0.1)
        elif demand < 0.3 and market_price < state.memory.get("avg_price", 1.0):
            action_type = "buy"
            amount = 100
        else:
            action_type = "hold"
            amount = 0

        state.remember("avg_price", market_price)

        return {
            "action_type": action_type,
            "agent_id": state.agent_id,
            "tick": observation.tick,
            "amount": amount,
            "price": market_price,
        }


class EconomicAgent(PlanetaryAgent):
    """Agent for managing tokenized economic flows.

    Handles token transactions, revenue optimization, and economic stability.
    """

    def __init__(
        self,
        agent_id: str,
        initial_balance: float = 10000.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ECONOMIC,
            policy=EconomicPolicy(),
            **kwargs,
        )
        self.token_balance = initial_balance

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["token_balance"] = self.token_balance
        return base


class IntegrationPolicy:
    """Policy for integration agents."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Decide on integration actions."""
        view = observation.view
        source_domain = view.get("source_domain", "")
        target_domain = view.get("target_domain", "")
        data_format = view.get("data_format", "json")

        if data_format not in ["json", "cbor", "protobuf"]:
            action_type = "transform"
        else:
            action_type = "relay"

        return {
            "action_type": action_type,
            "agent_id": state.agent_id,
            "tick": observation.tick,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "transform_format": "json" if action_type == "transform" else None,
        }


class IntegrationAgent(PlanetaryAgent):
    """Agent for cross-domain integration and interoperability.

    Manages data transformation and routing between sectors.
    """

    def __init__(
        self,
        agent_id: str,
        supported_domains: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.INTEGRATION,
            policy=IntegrationPolicy(),
            **kwargs,
        )
        self.supported_domains = supported_domains or [
            "energy", "transport", "healthcare", "finance"
        ]

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["supported_domains"] = self.supported_domains
        return base


class SimulationPolicy:
    """Policy for simulation/optimization agents."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Decide on simulation actions."""
        view = observation.view
        scenario = view.get("scenario", "baseline")
        iterations = view.get("iterations", 1000)
        convergence = view.get("convergence", 0.0)

        if convergence > 0.95:
            action_type = "report"
        elif convergence > 0.5:
            action_type = "refine"
            iterations = int(iterations * 0.5)
        else:
            action_type = "simulate"

        return {
            "action_type": action_type,
            "agent_id": state.agent_id,
            "tick": observation.tick,
            "scenario": scenario,
            "iterations": iterations,
            "convergence": convergence,
        }


class SimulationAgent(PlanetaryAgent):
    """Agent for predictive simulation and optimization.

    Runs simulations and optimizes system parameters.
    """

    def __init__(
        self,
        agent_id: str,
        simulation_types: list[str] | None = None,
        max_iterations: int = 10000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.SIMULATION,
            policy=SimulationPolicy(),
            **kwargs,
        )
        self.simulation_types = simulation_types or [
            "monte_carlo", "agent_based", "system_dynamics"
        ]
        self.max_iterations = max_iterations

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["simulation_types"] = self.simulation_types
        base["max_iterations"] = self.max_iterations
        return base


class SecurityPolicy:
    """Policy for security agents."""

    def decide(
        self, observation: AgentObservation, state: AgentState
    ) -> dict[str, Any]:
        """Decide on security actions."""
        view = observation.view
        threat_level = view.get("threat_level", 0.0)
        anomaly_score = view.get("anomaly_score", 0.0)
        attack_vector = view.get("attack_vector", "")

        if threat_level > 0.9:
            action_type = "isolate"
            priority = "critical"
        elif threat_level > 0.7 or anomaly_score > 0.8:
            action_type = "investigate"
            priority = "high"
        elif anomaly_score > 0.5:
            action_type = "monitor"
            priority = "medium"
        else:
            action_type = "scan"
            priority = "low"

        return {
            "action_type": action_type,
            "agent_id": state.agent_id,
            "tick": observation.tick,
            "threat_level": threat_level,
            "anomaly_score": anomaly_score,
            "attack_vector": attack_vector,
            "priority": priority,
        }


class SecurityAgent(PlanetaryAgent):
    """Agent for security monitoring and threat response.

    Handles threat detection, anomaly analysis, and incident response.
    """

    def __init__(
        self,
        agent_id: str,
        security_domains: list[str] | None = None,
        threat_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.SECURITY,
            policy=SecurityPolicy(),
            **kwargs,
        )
        self.security_domains = security_domains or [
            "network", "data", "identity", "compliance"
        ]
        self.threat_threshold = threat_threshold

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["security_domains"] = self.security_domains
        base["threat_threshold"] = self.threat_threshold
        return base


@dataclass
class AgentNetwork:
    """Network of interconnected planetary agents.

    Manages agent interactions, communication, and emergent behavior.

    Attributes:
        network_id: Unique network identifier
        agents: Dictionary of agents in the network
        connections: Adjacency list of agent connections
        feedback_loops: Defined feedback loops
        tick: Current simulation tick
    """

    network_id: str
    agents: dict[str, PlanetaryAgent] = field(default_factory=dict)
    connections: dict[str, list[str]] = field(default_factory=dict)
    feedback_loops: list[dict[str, Any]] = field(default_factory=list)
    tick: int = 0

    def add_agent(self, agent: PlanetaryAgent) -> None:
        """Add an agent to the network."""
        self.agents[agent.agent_id] = agent
        if agent.agent_id not in self.connections:
            self.connections[agent.agent_id] = []

    def connect_agents(self, agent1_id: str, agent2_id: str) -> None:
        """Create a bidirectional connection between agents."""
        if agent1_id in self.agents and agent2_id in self.agents:
            if agent2_id not in self.connections.get(agent1_id, []):
                self.connections.setdefault(agent1_id, []).append(agent2_id)
            if agent1_id not in self.connections.get(agent2_id, []):
                self.connections.setdefault(agent2_id, []).append(agent1_id)

    def define_feedback_loop(
        self,
        name: str,
        agents: list[str],
        feedback_type: str = "positive",
    ) -> dict[str, Any]:
        """Define a feedback loop between agents.

        Args:
            name: Loop name
            agents: Ordered list of agent IDs in the loop
            feedback_type: Type of feedback (positive/negative)

        Returns:
            Feedback loop definition
        """
        loop = {
            "name": name,
            "agents": agents,
            "feedback_type": feedback_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.feedback_loops.append(loop)

        # Create connections for the loop
        for i in range(len(agents)):
            self.connect_agents(agents[i], agents[(i + 1) % len(agents)])

        return loop

    def step(self, global_observation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Execute one simulation step for all agents.

        Args:
            global_observation: Global environment state

        Returns:
            Dictionary of actions by agent type
        """
        self.tick += 1
        actions: dict[str, list[dict[str, Any]]] = {}

        for agent_id, agent in self.agents.items():
            # Create agent-specific observation
            local_view = {
                **global_observation,
                "connected_agents": self.connections.get(agent_id, []),
                "tick": self.tick,
            }
            observation = AgentObservation(
                tick=self.tick,
                view=local_view,
                provenance="network_step",
            )

            # Get agent action
            action = agent.act(observation)

            # Group by agent type
            agent_type = agent.agent_type.value
            if agent_type not in actions:
                actions[agent_type] = []
            actions[agent_type].append(action)

        return actions

    def get_emergent_behavior(self) -> dict[str, Any]:
        """Analyze emergent behavior in the network.

        Returns:
            Analysis of emergent patterns
        """
        total_rewards = {}
        action_counts = {}

        for agent_id, agent in self.agents.items():
            agent_type = agent.agent_type.value

            # Aggregate rewards
            if agent_type not in total_rewards:
                total_rewards[agent_type] = 0.0
            total_rewards[agent_type] += sum(agent.state.rewards)

            # Count actions
            for action in agent.state.actions:
                action_type = action.get("action_type", "unknown")
                key = f"{agent_type}:{action_type}"
                action_counts[key] = action_counts.get(key, 0) + 1

        return {
            "tick": self.tick,
            "total_agents": len(self.agents),
            "total_connections": sum(len(c) for c in self.connections.values()) // 2,
            "feedback_loops": len(self.feedback_loops),
            "rewards_by_type": total_rewards,
            "action_distribution": action_counts,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get network statistics.

        Returns:
            Dictionary with network statistics
        """
        agents_by_type: dict[str, int] = {}
        for agent in self.agents.values():
            t = agent.agent_type.value
            agents_by_type[t] = agents_by_type.get(t, 0) + 1

        return {
            "network_id": self.network_id,
            "total_agents": len(self.agents),
            "agents_by_type": agents_by_type,
            "total_connections": sum(len(c) for c in self.connections.values()) // 2,
            "feedback_loops": len(self.feedback_loops),
            "current_tick": self.tick,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize network to dictionary."""
        return {
            "network_id": self.network_id,
            "tick": self.tick,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "connections": self.connections,
            "feedback_loops": self.feedback_loops,
        }
