"""
QRATUM Vertical Module Base Class

Abstract base class that all vertical modules must extend.
Enforces QRATUM invariants and provides common functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..platform.core import EventType, PlatformContract, create_event
from ..platform.event_chain import MerkleEventChain


class VerticalModuleBase(ABC):
    """
    Abstract base class for all QRATUM vertical modules.
    
    Every vertical must:
    1. Define its domain and capabilities
    2. Implement task execution methods
    3. Emit events for all significant operations
    4. Enforce safety boundaries
    5. Validate compliance requirements
    
    Attributes:
        vertical_name: Unique identifier for this vertical
        description: Human-readable description
        safety_disclaimer: Domain-specific safety warning
        prohibited_uses: List of prohibited applications
        required_compliance: List of compliance requirements
    """

    def __init__(
        self,
        vertical_name: str,
        description: str,
        safety_disclaimer: str,
        prohibited_uses: List[str],
        required_compliance: List[str],
    ):
        """
        Initialize vertical module.
        
        Args:
            vertical_name: Unique name (e.g., "JURIS", "VITRA")
            description: Module description
            safety_disclaimer: Safety warning for outputs
            prohibited_uses: List of prohibited applications
            required_compliance: List of compliance requirements
        """
        self.vertical_name = vertical_name
        self.description = description
        self.safety_disclaimer = safety_disclaimer
        self.prohibited_uses = prohibited_uses
        self.required_compliance = required_compliance

    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """
        Return list of supported task names.
        
        Returns:
            List of task identifiers
        """
        pass

    @abstractmethod
    def execute_task(
        self,
        task: str,
        parameters: Dict[str, Any],
        contract: PlatformContract,
        event_chain: MerkleEventChain,
    ) -> Dict[str, Any]:
        """
        Execute a task within this vertical.
        
        This is the main execution method. Must:
        1. Validate task and parameters
        2. Emit task started event
        3. Perform computation
        4. Emit task completed event
        5. Return results with safety disclaimer
        
        Args:
            task: Task identifier
            parameters: Task-specific parameters
            contract: Executing contract
            event_chain: Event chain for logging
        
        Returns:
            Dictionary with execution results
        
        Raises:
            ValueError: If task unknown or parameters invalid
        """
        pass

    def validate_safety(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate that request doesn't violate safety boundaries.
        
        Args:
            parameters: Task parameters to validate
        
        Returns:
            True if safe, False if prohibited
        """
        # Default implementation - can be overridden
        return True

    def validate_compliance(self, parameters: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check compliance requirements.
        
        Args:
            parameters: Task parameters
        
        Returns:
            Dictionary mapping compliance requirements to status
        """
        # Default implementation - all compliant
        return dict.fromkeys(self.required_compliance, True)

    def emit_task_event(
        self,
        event_type: EventType,
        contract_id: str,
        task: str,
        data: Dict[str, Any],
        event_chain: MerkleEventChain,
    ):
        """
        Emit a task-related event to the chain.
        
        Args:
            event_type: Type of event
            contract_id: Associated contract ID
            task: Task name
            data: Event data
            event_chain: Event chain to append to
        """
        event = create_event(
            event_type=event_type,
            contract_id=contract_id,
            data={
                "task": task,
                "vertical": self.vertical_name,
                **data,
            },
            emitter=self.vertical_name,
        )
        event_chain.append(event)

    def format_output(self, output: Any) -> Dict[str, Any]:
        """
        Format output with safety disclaimer and compliance info.
        
        Args:
            output: Raw output from computation
        
        Returns:
            Formatted output dictionary
        """
        return {
            "output": output,
            "vertical": self.vertical_name,
            "safety_disclaimer": self.safety_disclaimer,
            "prohibited_uses": self.prohibited_uses,
            "required_compliance": self.required_compliance,
            "recommendation": "Domain expert review recommended before production use",
        }
