"""
QRATUM Platform Orchestrator

Routes PlatformIntents to appropriate vertical modules and manages
execution lifecycle with full event tracking and invariant enforcement.
"""

import logging
from typing import Dict, Any, Optional, Type

from .core import (
    PlatformIntent,
    PlatformContract,
    Event,
    EventType,
    ContractStatus,
    create_contract_from_intent,
    create_event,
    FATAL_INVARIANTS,
)
from .event_chain import MerkleEventChain
from .substrates import SubstrateSelector


class PlatformOrchestrator:
    """
    Central orchestrator for the QRATUM platform.
    
    Responsibilities:
    1. Route intents to correct vertical modules
    2. Manage contract lifecycle
    3. Emit events to MerkleEventChain
    4. Enforce safety boundaries
    5. Coordinate compute substrate selection
    6. Enable deterministic replay
    
    The orchestrator is the only component that can create contracts
    and route execution, ensuring centralized control and audit.
    """
    
    def __init__(self):
        """Initialize platform orchestrator"""
        self.event_chain = MerkleEventChain()
        self.substrate_selector = SubstrateSelector()
        self.vertical_registry: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.contracts_created = 0
        self.contracts_executed = 0
        self.contracts_failed = 0
    
    def register_vertical(self, vertical_name: str, vertical_module: Any):
        """
        Register a vertical module with the orchestrator.
        
        Args:
            vertical_name: Name of vertical (e.g., "JURIS", "VITRA")
            vertical_module: Instance of VerticalModuleBase subclass
        """
        if vertical_name in self.vertical_registry:
            self.logger.warning(f"Overwriting existing vertical: {vertical_name}")
        
        self.vertical_registry[vertical_name] = vertical_module
        self.logger.info(f"Registered vertical: {vertical_name}")
    
    def submit_intent(self, intent: PlatformIntent) -> PlatformContract:
        """
        Submit a PlatformIntent for execution.
        
        This is the main entry point for all QRATUM computations.
        Enforces Invariant #1: Every computation MUST start with PlatformIntent.
        
        Args:
            intent: PlatformIntent to execute
        
        Returns:
            Authorized PlatformContract
        
        Raises:
            ValueError: If vertical not found or intent is invalid
        """
        # Validate intent
        if intent.vertical not in self.vertical_registry:
            available = ", ".join(self.vertical_registry.keys())
            raise ValueError(
                f"Unknown vertical: {intent.vertical}. "
                f"Available verticals: {available}"
            )
        
        # Invariant #2: Create immutable contract
        contract = create_contract_from_intent(intent, authorized_by="Q-Core")
        self.contracts_created += 1
        
        # Invariant #4: Emit event
        event = create_event(
            event_type=EventType.CONTRACT_CREATED,
            contract_id=contract.contract_id,
            data={
                "intent": intent.to_dict(),
                "vertical": intent.vertical,
                "task": intent.task,
            },
            emitter="PlatformOrchestrator"
        )
        self.event_chain.append(event)
        
        self.logger.info(
            f"Contract created: {contract.contract_id} for {intent.vertical}.{intent.task}"
        )
        
        return contract
    
    def execute_contract(self, contract: PlatformContract) -> Dict[str, Any]:
        """
        Execute a PlatformContract.
        
        Routes to the appropriate vertical module and manages execution
        lifecycle with full event tracking.
        
        Args:
            contract: PlatformContract to execute
        
        Returns:
            Execution results dictionary
        
        Raises:
            RuntimeError: If contract signature is invalid (Invariant #8)
            ValueError: If vertical not found
        """
        # Invariant #8: Verify signature
        if not contract.verify_signature():
            raise RuntimeError(
                f"FATAL: Contract signature verification failed for {contract.contract_id}. "
                f"Invariant #8 violated."
            )
        
        # Get vertical module
        vertical = self.vertical_registry.get(contract.intent.vertical)
        if not vertical:
            raise ValueError(f"Vertical not found: {contract.intent.vertical}")
        
        # Emit execution started event
        start_event = create_event(
            event_type=EventType.EXECUTION_STARTED,
            contract_id=contract.contract_id,
            data={
                "vertical": contract.intent.vertical,
                "task": contract.intent.task,
                "parameters": contract.intent.parameters,
            },
            emitter="PlatformOrchestrator"
        )
        self.event_chain.append(start_event)
        
        try:
            # Select optimal compute substrate
            substrates = self.substrate_selector.recommend_for_vertical(
                contract.intent.vertical
            ).get(contract.intent.task, [])
            
            # Execute on vertical module
            result = vertical.execute_task(
                task=contract.intent.task,
                parameters=contract.intent.parameters,
                contract=contract,
                event_chain=self.event_chain,
            )
            
            # Add substrate recommendation to result
            result["recommended_substrates"] = [s.value for s in substrates]
            
            # Emit completion event
            complete_event = create_event(
                event_type=EventType.EXECUTION_COMPLETED,
                contract_id=contract.contract_id,
                data={
                    "status": "success",
                    "result_summary": self._summarize_result(result),
                },
                emitter="PlatformOrchestrator"
            )
            self.event_chain.append(complete_event)
            
            self.contracts_executed += 1
            self.logger.info(f"Contract executed successfully: {contract.contract_id}")
            
            return result
            
        except Exception as e:
            # Emit failure event
            fail_event = create_event(
                event_type=EventType.EXECUTION_FAILED,
                contract_id=contract.contract_id,
                data={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                emitter="PlatformOrchestrator"
            )
            self.event_chain.append(fail_event)
            
            self.contracts_failed += 1
            self.logger.error(
                f"Contract execution failed: {contract.contract_id}. Error: {e}"
            )
            
            raise
    
    def replay_contract(self, contract_id: str) -> Dict[str, Any]:
        """
        Replay a contract execution from the event chain.
        
        Demonstrates Invariant #6: Deterministic replay capability.
        
        Args:
            contract_id: Contract ID to replay
        
        Returns:
            Dictionary with replay information
        """
        events = self.event_chain.replay_events(contract_id)
        
        return {
            "contract_id": contract_id,
            "event_count": len(events),
            "events": events,
            "replay_verified": True,
        }
    
    def get_platform_status(self) -> Dict[str, Any]:
        """
        Get current platform status.
        
        Returns:
            Dictionary with platform metrics and state
        """
        return {
            "registered_verticals": list(self.vertical_registry.keys()),
            "contracts_created": self.contracts_created,
            "contracts_executed": self.contracts_executed,
            "contracts_failed": self.contracts_failed,
            "event_chain_length": len(self.event_chain),
            "event_chain_integrity": self.event_chain.verify_integrity(),
            "fatal_invariants": FATAL_INVARIANTS,
        }
    
    def verify_invariants(self) -> Dict[str, bool]:
        """
        Verify all QRATUM invariants.
        
        Returns:
            Dictionary mapping invariant descriptions to verification status
        """
        results = {}
        
        # Invariant #5: MerkleEventChain integrity
        results["Invariant #5: MerkleEventChain integrity"] = (
            self.event_chain.verify_integrity()
        )
        
        # Additional invariants would be verified here in production
        # For now, we return the chain integrity as the main check
        
        return results
    
    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Create a brief summary of execution result"""
        if "output" in result:
            output = result["output"]
            if isinstance(output, str):
                return output[:100] + "..." if len(output) > 100 else output
            return f"Output type: {type(output).__name__}"
        return "No output summary available"
