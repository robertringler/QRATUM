"""Authorization system for QRATUM-ASI."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from qratum_asi.core.types import ASISafetyLevel, AuthorizationType


@dataclass
class AuthorizationRequest:
    """Request for authorization."""

    request_id: str
    operation_type: str
    safety_level: ASISafetyLevel
    authorization_type: AuthorizationType
    requester: str
    justification: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"  # pending, approved, denied
    authorized_by: Optional[Set[str]] = None
    decision_timestamp: Optional[str] = None

    def __post_init__(self):
        """Initialize authorized_by as set."""
        if self.authorized_by is None:
            self.authorized_by = set()


@dataclass
class AuthorizationSystem:
    """Authorization system for ASI operations.
    
    Enforces human oversight requirements based on safety levels.
    All sensitive operations require explicit human authorization.
    """

    pending_requests: Dict[str, AuthorizationRequest] = field(default_factory=dict)
    approved_requests: Dict[str, AuthorizationRequest] = field(default_factory=dict)
    denied_requests: Dict[str, AuthorizationRequest] = field(default_factory=dict)
    authorized_users: Set[str] = field(default_factory=set)

    def request_authorization(
        self,
        request_id: str,
        operation_type: str,
        safety_level: ASISafetyLevel,
        requester: str,
        justification: str,
    ) -> AuthorizationRequest:
        """Submit authorization request."""
        # Determine required authorization type
        auth_type = self._determine_authorization_type(safety_level)

        request = AuthorizationRequest(
            request_id=request_id,
            operation_type=operation_type,
            safety_level=safety_level,
            authorization_type=auth_type,
            requester=requester,
            justification=justification,
        )

        self.pending_requests[request_id] = request
        return request

    def grant_authorization(
        self, request_id: str, authorized_by: str
    ) -> Optional[AuthorizationRequest]:
        """Grant authorization for a request."""
        if request_id not in self.pending_requests:
            return None

        request = self.pending_requests[request_id]

        # Add authorizer
        request.authorized_by.add(authorized_by)

        # Check if enough authorizers
        required_count = self._get_required_authorizer_count(request.authorization_type)
        if len(request.authorized_by) >= required_count:
            request.status = "approved"
            request.decision_timestamp = datetime.utcnow().isoformat()
            self.approved_requests[request_id] = request
            del self.pending_requests[request_id]

        return request

    def deny_authorization(
        self, request_id: str, denied_by: str, reason: str
    ) -> Optional[AuthorizationRequest]:
        """Deny authorization for a request."""
        if request_id not in self.pending_requests:
            return None

        request = self.pending_requests[request_id]
        request.status = "denied"
        request.decision_timestamp = datetime.utcnow().isoformat()
        request.authorized_by.add(f"{denied_by} (denied: {reason})")

        self.denied_requests[request_id] = request
        del self.pending_requests[request_id]
        return request

    def is_authorized(self, request_id: str) -> bool:
        """Check if request is authorized."""
        return request_id in self.approved_requests

    def add_authorized_user(self, user_id: str) -> None:
        """Add user to authorized users list."""
        self.authorized_users.add(user_id)

    def remove_authorized_user(self, user_id: str) -> None:
        """Remove user from authorized users list."""
        self.authorized_users.discard(user_id)

    def get_pending_requests(self) -> List[AuthorizationRequest]:
        """Get all pending authorization requests."""
        return list(self.pending_requests.values())

    def _determine_authorization_type(self, safety_level: ASISafetyLevel) -> AuthorizationType:
        """Determine required authorization type based on safety level."""
        mapping = {
            ASISafetyLevel.ROUTINE: AuthorizationType.NONE,
            ASISafetyLevel.ELEVATED: AuthorizationType.SINGLE_HUMAN,
            ASISafetyLevel.SENSITIVE: AuthorizationType.SINGLE_HUMAN,
            ASISafetyLevel.CRITICAL: AuthorizationType.MULTI_HUMAN,
            ASISafetyLevel.EXISTENTIAL: AuthorizationType.BOARD_LEVEL,
        }
        return mapping[safety_level]

    def _get_required_authorizer_count(self, auth_type: AuthorizationType) -> int:
        """Get required number of authorizers."""
        mapping = {
            AuthorizationType.NONE: 0,
            AuthorizationType.SINGLE_HUMAN: 1,
            AuthorizationType.MULTI_HUMAN: 2,
            AuthorizationType.BOARD_LEVEL: 3,
            AuthorizationType.EXTERNAL_OVERSIGHT: 5,
        }
        return mapping[auth_type]
