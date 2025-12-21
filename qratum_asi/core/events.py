"""Extended event system for QRATUM-ASI."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict
from datetime import datetime
import hashlib
import json


class ASIEventType(str, Enum):
    """Event types for ASI operations."""

    # Q-REALITY events
    KNOWLEDGE_NODE_CREATED = "knowledge_node_created"
    KNOWLEDGE_NODE_UPDATED = "knowledge_node_updated"
    CAUSAL_LINK_CREATED = "causal_link_created"
    WORLD_MODEL_UPDATED = "world_model_updated"

    # Q-MIND events
    REASONING_STARTED = "reasoning_started"
    REASONING_COMPLETED = "reasoning_completed"
    INFERENCE_MADE = "inference_made"
    CROSS_DOMAIN_SYNTHESIS = "cross_domain_synthesis"

    # Q-EVOLVE events
    IMPROVEMENT_PROPOSED = "improvement_proposed"
    IMPROVEMENT_AUTHORIZED = "improvement_authorized"
    IMPROVEMENT_EXECUTED = "improvement_executed"
    IMPROVEMENT_VALIDATED = "improvement_validated"
    IMPROVEMENT_ROLLED_BACK = "improvement_rolled_back"
    ROLLBACK_POINT_CREATED = "rollback_point_created"

    # Q-WILL events
    GOAL_PROPOSED = "goal_proposed"
    GOAL_AUTHORIZED = "goal_authorized"
    GOAL_REJECTED = "goal_rejected"
    GOAL_COMPLETED = "goal_completed"

    # Q-FORGE events
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    DISCOVERY_MADE = "discovery_made"
    DISCOVERY_VALIDATED = "discovery_validated"
    NOVEL_SYNTHESIS = "novel_synthesis"

    # Safety events
    SAFETY_VIOLATION_DETECTED = "safety_violation_detected"
    BOUNDARY_CHECK_PASSED = "boundary_check_passed"
    AUTHORIZATION_REQUIRED = "authorization_required"
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"


@dataclass(frozen=True)
class ASIEvent:
    """Immutable event record for ASI operations."""

    event_type: ASIEventType
    payload: Dict[str, Any]
    contract_id: str
    timestamp: str
    event_id: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "payload": dict(self.payload),
            "contract_id": self.contract_id,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "index": self.index,
        }

    @staticmethod
    def create(
        event_type: ASIEventType,
        payload: Dict[str, Any],
        contract_id: str,
        index: int,
    ) -> "ASIEvent":
        """Create a new ASI event with computed hash."""
        timestamp = datetime.utcnow().isoformat()
        event_data = {
            "event_type": event_type.value,
            "payload": payload,
            "contract_id": contract_id,
            "timestamp": timestamp,
            "index": index,
        }
        serialized = json.dumps(event_data, sort_keys=True)
        event_id = hashlib.sha256(serialized.encode()).hexdigest()

        return ASIEvent(
            event_type=event_type,
            payload=payload,
            contract_id=contract_id,
            timestamp=timestamp,
            event_id=event_id,
            index=index,
        )
