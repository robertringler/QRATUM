"""QRATUM Platform Orchestrator.

Central orchestrator routing intents to vertical modules and managing
overall platform lifecycle.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Optional

from platform.core.base import ExecutionResult, VerticalModuleBase
from platform.core.intent import PlatformContract, PlatformIntent, VerticalType


class QRATUMPlatform:
    """Main platform orchestrator.

    Routes intents to appropriate vertical modules and manages
    platform-wide operations.

    Attributes:
        modules: Registry of vertical modules
        platform_seed: Platform-wide random seed
    """

    def __init__(self, seed: int = 42):
        """Initialize QRATUM Platform.

        Args:
            seed: Platform-wide random seed for determinism
        """
        self.modules: Dict[VerticalType, VerticalModuleBase] = {}
        self.platform_seed = seed
        self._intent_count = 0

    def register_module(self, vertical: VerticalType, module: VerticalModuleBase) -> None:
        """Register a vertical module with the platform.

        Args:
            vertical: Vertical type
            module: Module instance
        """
        self.modules[vertical] = module

    def unregister_module(self, vertical: VerticalType) -> None:
        """Unregister a vertical module.

        Args:
            vertical: Vertical type to unregister
        """
        if vertical in self.modules:
            del self.modules[vertical]

    def create_contract(
        self,
        intent: PlatformIntent,
        authorized: bool = True,
        expiry_hours: Optional[int] = None,
    ) -> PlatformContract:
        """Create execution contract from intent.

        Args:
            intent: Platform intent
            authorized: Whether to authorize immediately
            expiry_hours: Hours until expiration (None = no expiration)

        Returns:
            Platform contract
        """
        # Compute contract hash
        intent_data = {
            "vertical": intent.vertical.value,
            "operation": intent.operation,
            "parameters": intent.parameters,
            "attestations": sorted(intent.compliance_attestations),
            "user_id": intent.user_id,
            "timestamp": intent.timestamp.isoformat(),
        }
        intent_json = json.dumps(intent_data, sort_keys=True, default=str)
        contract_hash = hashlib.sha256(intent_json.encode()).hexdigest()

        # Calculate expiry
        expiry_timestamp = None
        if expiry_hours is not None:
            from datetime import timedelta

            expiry_timestamp = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

        return PlatformContract(
            intent=intent,
            contract_hash=contract_hash,
            authorized=authorized,
            authorization_timestamp=datetime.now(timezone.utc),
            expiry_timestamp=expiry_timestamp,
        )

    def execute_intent(self, intent: PlatformIntent) -> ExecutionResult:
        """Execute an intent by routing to appropriate vertical module.

        Args:
            intent: Platform intent to execute

        Returns:
            Execution result

        Raises:
            ValueError: If vertical module not registered
        """
        self._intent_count += 1

        # Check if module is registered
        if intent.vertical not in self.modules:
            raise ValueError(f"Vertical module not registered: {intent.vertical.value}")

        # Create and authorize contract
        contract = self.create_contract(intent, authorized=True)

        # Route to module
        module = self.modules[intent.vertical]
        result = module.execute(contract)

        return result

    def execute_contract(self, contract: PlatformContract) -> ExecutionResult:
        """Execute a pre-created contract.

        Args:
            contract: Platform contract to execute

        Returns:
            Execution result

        Raises:
            ValueError: If vertical module not registered or contract invalid
        """
        # Check contract validity
        if not contract.is_valid():
            raise ValueError("Contract is not valid (not authorized or expired)")

        # Check if module is registered
        if contract.intent.vertical not in self.modules:
            raise ValueError(f"Vertical module not registered: {contract.intent.vertical.value}")

        # Route to module
        module = self.modules[contract.intent.vertical]
        result = module.execute(contract)

        return result

    def get_registered_verticals(self) -> list[VerticalType]:
        """Get list of registered vertical modules.

        Returns:
            List of registered vertical types
        """
        return list(self.modules.keys())

    def get_platform_stats(self) -> Dict[str, any]:
        """Get platform statistics.

        Returns:
            Dictionary of platform statistics
        """
        stats = {
            "registered_modules": len(self.modules),
            "verticals": [v.value for v in self.modules.keys()],
            "total_intents_processed": self._intent_count,
            "platform_seed": self.platform_seed,
        }

        # Add per-module stats
        module_stats = {}
        for vertical, module in self.modules.items():
            module_stats[vertical.value] = {
                "event_count": len(module.event_chain),
                "audit_valid": module.verify_audit_trail(),
                "merkle_root": module.event_chain.get_merkle_root()[:16],
            }

        stats["module_stats"] = module_stats
        return stats
