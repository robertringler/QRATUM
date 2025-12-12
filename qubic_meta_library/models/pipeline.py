"""Pipeline execution configuration model for Qubic Meta Library."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pipeline:
    """
    Represents an execution pipeline for prompt processing.

    Attributes:
        id: Pipeline identifier
        name: Pipeline name
        phase: Execution phase (1-4)
        prompts: List of prompt IDs to execute
        keystones: List of keystone prompt IDs
        platform: Target platform (QuASIM/QStack/QNimbus)
        start_date: Pipeline start date
        end_date: Pipeline end date
        status: Pipeline status (pending/running/completed/failed)
        dependencies: List of prerequisite pipeline IDs
    """

    id: str
    name: str
    phase: int
    prompts: list[int] = field(default_factory=list)
    keystones: list[int] = field(default_factory=list)
    platform: str = "QuASIM"
    start_date: str = ""
    end_date: str = ""
    status: str = "pending"
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate pipeline attributes."""
        if not 1 <= self.phase <= 4:
            raise ValueError(f"Phase must be between 1 and 4, got {self.phase}")
        if self.platform not in ["QuASIM", "QStack", "QNimbus"]:
            raise ValueError(f"Platform must be QuASIM, QStack, or QNimbus, got {self.platform}")
        if self.status not in ["pending", "running", "completed", "failed"]:
            raise ValueError(
                f"Status must be pending, running, completed, or failed, got {self.status}"
            )

    def is_ready(self, completed_pipelines: set[str]) -> bool:
        """
        Check if pipeline is ready to execute.

        Args:
            completed_pipelines: Set of completed pipeline IDs

        Returns:
            True if all dependencies are completed
        """
        return all(dep in completed_pipelines for dep in self.dependencies)

    def add_prompt(self, prompt_id: int, is_keystone: bool = False):
        """
        Add a prompt to the pipeline.

        Args:
            prompt_id: Prompt ID to add
            is_keystone: Whether prompt is a keystone
        """
        if prompt_id not in self.prompts:
            self.prompts.append(prompt_id)
        if is_keystone and prompt_id not in self.keystones:
            self.keystones.append(prompt_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "phase": self.phase,
            "prompts": self.prompts,
            "keystones": self.keystones,
            "platform": self.platform,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pipeline":
        """Create Pipeline instance from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            phase=data["phase"],
            prompts=data.get("prompts", []),
            keystones=data.get("keystones", []),
            platform=data.get("platform", "QuASIM"),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            status=data.get("status", "pending"),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
        )
