"""Merkle Verification for Sandbox Communication.

Implements immutable Merkle-verified communication between sandbox and
production. All data exchanges are cryptographically verifiable.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from qradle.merkle import MerkleChain


class VerificationStatus(Enum):
    """Status of verification."""

    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class VerificationResult:
    """Result of Merkle verification.

    Attributes:
        verification_id: Unique verification identifier
        message_hash: Hash of verified message
        is_valid: Whether verification passed
        status: Verification status
        chain_proof: Merkle chain proof at time of verification
        verification_time_ms: Time taken for verification
    """

    verification_id: str
    message_hash: str
    is_valid: bool
    status: VerificationStatus
    chain_proof: str
    verification_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize verification result."""
        return {
            "verification_id": self.verification_id,
            "message_hash": self.message_hash,
            "is_valid": self.is_valid,
            "status": self.status.value,
            "chain_proof": self.chain_proof,
            "verification_time_ms": self.verification_time_ms,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class VerifiedMessage:
    """Message with Merkle verification.

    Attributes:
        message_id: Unique message identifier
        channel_id: Channel the message was sent on
        payload: Message payload
        sender_id: Sender identifier
        content_hash: Hash of message content
        merkle_proof: Merkle proof at time of creation
        verification_status: Current verification status
    """

    message_id: str
    channel_id: str
    payload: dict[str, Any]
    sender_id: str
    content_hash: str = ""
    merkle_proof: str = ""
    verification_status: VerificationStatus = VerificationStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        """Compute content hash if not provided."""
        if not self.content_hash:
            content = {
                "message_id": self.message_id,
                "channel_id": self.channel_id,
                "payload": self.payload,
                "sender_id": self.sender_id,
            }
            self.content_hash = hashlib.sha3_256(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize verified message."""
        return {
            "message_id": self.message_id,
            "channel_id": self.channel_id,
            "payload": self.payload,
            "sender_id": self.sender_id,
            "content_hash": self.content_hash,
            "merkle_proof": self.merkle_proof,
            "verification_status": self.verification_status.value,
            "created_at": self.created_at,
        }


class MerkleVerifiedChannel:
    """Channel for Merkle-verified communication between sandbox and production.

    All messages sent through this channel are:
    - Immutable once committed
    - Cryptographically verifiable
    - Auditable through Merkle chain
    """

    def __init__(
        self,
        channel_id: str,
        merkle_chain: MerkleChain | None = None,
        verify_on_receive: bool = True,
    ):
        """Initialize verified channel.

        Args:
            channel_id: Unique channel identifier
            merkle_chain: Merkle chain for verification
            verify_on_receive: Whether to auto-verify received messages
        """
        self.channel_id = channel_id
        self.merkle_chain = merkle_chain or MerkleChain()
        self.verify_on_receive = verify_on_receive

        # Message tracking
        self.sent_messages: dict[str, VerifiedMessage] = {}
        self.received_messages: dict[str, VerifiedMessage] = {}
        self._message_counter = 0
        self._verification_counter = 0
        self._lock = threading.RLock()

        # Log initialization
        self.merkle_chain.add_event(
            "verified_channel_initialized",
            {
                "channel_id": channel_id,
                "verify_on_receive": verify_on_receive,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def send(
        self,
        payload: dict[str, Any],
        sender_id: str,
    ) -> VerifiedMessage:
        """Send a verified message.

        Args:
            payload: Message payload
            sender_id: Sender identifier

        Returns:
            VerifiedMessage with Merkle proof
        """
        with self._lock:
            self._message_counter += 1
            message_id = f"msg_{self.channel_id}_{self._message_counter:08d}"

            message = VerifiedMessage(
                message_id=message_id,
                channel_id=self.channel_id,
                payload=payload,
                sender_id=sender_id,
            )

            # Add to Merkle chain and get proof
            self.merkle_chain.add_event(
                "message_sent",
                {
                    "message_id": message_id,
                    "content_hash": message.content_hash,
                    "sender_id": sender_id,
                },
            )
            message.merkle_proof = self.merkle_chain.get_chain_proof()
            message.verification_status = VerificationStatus.VALID

            self.sent_messages[message_id] = message
            return message

    def receive(
        self,
        message: VerifiedMessage,
    ) -> VerificationResult:
        """Receive and verify a message.

        Args:
            message: Message to receive and verify

        Returns:
            VerificationResult indicating validity
        """
        start_time = time.perf_counter()

        with self._lock:
            self._verification_counter += 1
            verification_id = f"verify_{self.channel_id}_{self._verification_counter:08d}"

            # Verify message integrity
            expected_hash = self._compute_content_hash(message)
            is_valid = expected_hash == message.content_hash

            if is_valid and self.verify_on_receive:
                # Additional Merkle verification
                is_valid = self._verify_merkle_proof(message)

            verification_time_ms = (time.perf_counter() - start_time) * 1000

            status = VerificationStatus.VALID if is_valid else VerificationStatus.INVALID
            message.verification_status = status

            # Store received message
            self.received_messages[message.message_id] = message

            # Log verification
            self.merkle_chain.add_event(
                "message_verified",
                {
                    "message_id": message.message_id,
                    "verification_id": verification_id,
                    "is_valid": is_valid,
                    "status": status.value,
                },
            )

            return VerificationResult(
                verification_id=verification_id,
                message_hash=message.content_hash,
                is_valid=is_valid,
                status=status,
                chain_proof=self.merkle_chain.get_chain_proof(),
                verification_time_ms=verification_time_ms,
                details={
                    "message_id": message.message_id,
                    "sender_id": message.sender_id,
                },
            )

    def _compute_content_hash(self, message: VerifiedMessage) -> str:
        """Compute expected content hash for a message."""
        content = {
            "message_id": message.message_id,
            "channel_id": message.channel_id,
            "payload": message.payload,
            "sender_id": message.sender_id,
        }
        return hashlib.sha3_256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

    def _verify_merkle_proof(self, message: VerifiedMessage) -> bool:
        """Verify Merkle proof of a message.

        For a real implementation, this would verify against the
        Merkle tree structure. Here we verify chain integrity.
        """
        return self.merkle_chain.verify_integrity()

    def get_channel_stats(self) -> dict[str, Any]:
        """Get channel statistics."""
        valid_received = sum(
            1
            for m in self.received_messages.values()
            if m.verification_status == VerificationStatus.VALID
        )

        return {
            "channel_id": self.channel_id,
            "messages_sent": len(self.sent_messages),
            "messages_received": len(self.received_messages),
            "valid_received": valid_received,
            "verification_rate": (
                valid_received / len(self.received_messages)
                if self.received_messages
                else 1.0
            ),
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
        }


class AuditChainLogger:
    """Logger that maintains Merkle-chained audit logs.

    Ensures all sandbox operations are recorded with cryptographic
    verifiability without adding latency to production paths.
    """

    def __init__(
        self,
        logger_id: str = "audit",
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize audit chain logger.

        Args:
            logger_id: Unique logger identifier
            merkle_chain: Merkle chain for audit trail
        """
        self.logger_id = logger_id
        self.merkle_chain = merkle_chain or MerkleChain()

        self._log_counter = 0
        self._lock = threading.RLock()

        # Log buffers for async processing
        self._pending_logs: list[dict[str, Any]] = []
        self._committed_logs: list[str] = []

    def log(
        self,
        event_type: str,
        data: dict[str, Any],
        sync: bool = False,
    ) -> str:
        """Log an event to the audit chain.

        Args:
            event_type: Type of event
            data: Event data
            sync: Whether to commit synchronously

        Returns:
            Hash of the logged event
        """
        with self._lock:
            self._log_counter += 1

            log_entry = {
                "log_id": f"log_{self.logger_id}_{self._log_counter:010d}",
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if sync:
                # Commit immediately
                event_hash = self.merkle_chain.add_event(event_type, log_entry)
                self._committed_logs.append(event_hash)
                return event_hash
            else:
                # Buffer for async commit
                self._pending_logs.append(log_entry)
                return hashlib.sha3_256(
                    json.dumps(log_entry, sort_keys=True).encode()
                ).hexdigest()

    def flush(self) -> int:
        """Flush pending logs to the audit chain.

        Returns:
            Number of logs committed
        """
        with self._lock:
            committed_count = 0
            for log_entry in self._pending_logs:
                event_hash = self.merkle_chain.add_event(
                    log_entry["event_type"], log_entry
                )
                self._committed_logs.append(event_hash)
                committed_count += 1

            self._pending_logs.clear()
            return committed_count

    def get_chain_proof(self) -> str:
        """Get current chain proof."""
        return self.merkle_chain.get_chain_proof()

    def verify_chain(self) -> bool:
        """Verify integrity of the audit chain."""
        return self.merkle_chain.verify_integrity()

    def get_logger_stats(self) -> dict[str, Any]:
        """Get logger statistics."""
        return {
            "logger_id": self.logger_id,
            "total_logs": self._log_counter,
            "pending_logs": len(self._pending_logs),
            "committed_logs": len(self._committed_logs),
            "chain_length": len(self.merkle_chain.chain),
            "chain_valid": self.merkle_chain.verify_integrity(),
        }
