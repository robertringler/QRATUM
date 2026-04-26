"""Q-Stack deterministic multi-agent layer."""

from qagents.base import Agent, AgentLog, AgentObservation, AgentPolicy, AgentState, LambdaPolicy
from qagents.interaction import InteractionBus, Message
from qagents.observation import filtered_observation, merge_observations
from qagents.reality_interface import (
    ACTION_TYPES,
    ControlDecision,
    FallbackPlan,
    IntentInterpretation,
    PredictedOutcome,
    RealityInterfaceController,
    SelectedAction,
    default_proposer,
    default_simulator,
)
from qagents.registry import AgentRegistry
from qagents.rewards import aggregate_rewards, shaped_reward
from qagents.strategy import (
    DeterminismChecker,
    PolicyAdapter,
    ScriptedStrategy,
    Strategy,
    StrategyDecision,
    ThresholdStrategy,
)

__all__ = [
    "Agent",
    "AgentObservation",
    "AgentPolicy",
    "AgentLog",
    "AgentState",
    "LambdaPolicy",
    "Strategy",
    "ThresholdStrategy",
    "ScriptedStrategy",
    "PolicyAdapter",
    "DeterminismChecker",
    "StrategyDecision",
    "AgentRegistry",
    "InteractionBus",
    "Message",
    "filtered_observation",
    "merge_observations",
    "aggregate_rewards",
    "shaped_reward",
    "ACTION_TYPES",
    "ControlDecision",
    "FallbackPlan",
    "IntentInterpretation",
    "PredictedOutcome",
    "RealityInterfaceController",
    "SelectedAction",
    "default_proposer",
    "default_simulator",
]
