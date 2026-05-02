"""Q-Stack deterministic multi-agent layer."""

from qagents.adapters import (ciir_proposer, ciir_simulator, crs_proposer,
                              crs_simulator, make_ciir_controller,
                              make_crs_controller, make_qratum_controller,
                              qratum_proposer, qratum_simulator)
from qagents.base import (Agent, AgentLog, AgentObservation, AgentPolicy,
                          AgentState, LambdaPolicy)
from qagents.interaction import InteractionBus, Message
from qagents.observation import filtered_observation, merge_observations
from qagents.reality_interface import (ACTION_TYPES, ControlDecision,
                                       FallbackPlan, IntentInterpretation,
                                       PredictedOutcome,
                                       RealityInterfaceController,
                                       SelectedAction, default_proposer,
                                       default_simulator)
from qagents.registry import AgentRegistry
from qagents.rewards import aggregate_rewards, shaped_reward
from qagents.strategy import (DeterminismChecker, PolicyAdapter,
                              ScriptedStrategy, Strategy, StrategyDecision,
                              ThresholdStrategy)

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
    "make_qratum_controller",
    "qratum_simulator",
    "qratum_proposer",
    "make_ciir_controller",
    "ciir_simulator",
    "ciir_proposer",
    "make_crs_controller",
    "crs_simulator",
    "crs_proposer",
]
