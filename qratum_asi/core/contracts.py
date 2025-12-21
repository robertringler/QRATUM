"""Extended contract system for QRATUM-ASI."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
import hashlib
import json

from qratum_asi.core.types import ASISafetyLevel, AuthorizationType


@dataclass
class ASIContract:
    """Extended contract for ASI operations.
    
    All ASI operations are contract-bound with immutable records,
    authorization requirements, and safety level classification.
    """

    contract_id: str
    operation_type: str
    safety_level: ASISafetyLevel
    authorization_type: AuthorizationType
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    authorized: bool = False
    authorized_by: Optional[str] = None
    authorization_timestamp: Optional[str] = None
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    rollback_point: Optional[str] = None
    hash: Optional[str] = None

    def __post_init__(self):
        """Compute contract hash."""
        if self.hash is None:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash of contract."""
        contract_data = {
            "contract_id": self.contract_id,
            "operation_type": self.operation_type,
            "safety_level": self.safety_level.value,
            "authorization_type": self.authorization_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
        serialized = json.dumps(contract_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def authorize(self, authorized_by: str) -> None:
        """Authorize this contract."""
        if self.authorized:
            raise ValueError(f"Contract {self.contract_id} already authorized")
        self.authorized = True
        self.authorized_by = authorized_by
        self.authorization_timestamp = datetime.utcnow().isoformat()

    def is_authorized(self) -> bool:
        """Check if contract is authorized."""
        return self.authorized

    def requires_authorization(self) -> bool:
        """Check if contract requires authorization."""
        return self.authorization_type != AuthorizationType.NONE

    def validate(self) -> bool:
        """Validate contract against criteria."""
        # Check authorization requirement
        if self.requires_authorization() and not self.is_authorized():
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary."""
        return {
            "contract_id": self.contract_id,
            "operation_type": self.operation_type,
            "safety_level": self.safety_level.value,
            "authorization_type": self.authorization_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "authorized": self.authorized,
            "authorized_by": self.authorized_by,
            "authorization_timestamp": self.authorization_timestamp,
            "validation_criteria": self.validation_criteria,
            "rollback_point": self.rollback_point,
            "hash": self.hash,
        }
