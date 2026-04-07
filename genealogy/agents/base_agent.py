"""
Base class for genealogical research agents.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.person import Person, Relationship
from ..models.record import Citation, GenealogicalRecord, Source


class ResearchAgent(ABC):
    """
    Abstract base class for specialized genealogical research agents.

    Each agent is responsible for a specific time period or type of research,
    working independently but coordinated by the Chief Orchestrator.
    """

    def __init__(self, agent_id: str, scope: str):
        """
        Initialize the research agent.

        Args:
            agent_id: Unique identifier for this agent
            scope: Description of research scope (time period, geography, etc.)
        """
        self.agent_id = agent_id
        self.scope = scope
        self.persons: Dict[str, Person] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.records: Dict[str, GenealogicalRecord] = {}
        self.sources: Dict[str, Source] = {}
        self.citations: Dict[str, Citation] = {}
        self.findings: List[str] = []
        self.conflicts: List[str] = []
        self.status = "initialized"
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    @abstractmethod
    def execute_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research task for this agent's scope.

        Args:
            context: Research context including target person, known ancestors, etc.

        Returns:
            Dictionary containing research results, findings, and evidence
        """
        pass

    @abstractmethod
    def analyze_evidence(self) -> Dict[str, Any]:
        """
        Analyze collected evidence and assess proof strength.

        Returns:
            Dictionary containing analysis results
        """
        pass

    def add_person(self, person: Person) -> None:
        """Add a person to the research database."""
        self.persons[person.person_id] = person

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the research database."""
        self.relationships[relationship.relationship_id] = relationship

    def add_record(self, record: GenealogicalRecord) -> None:
        """Add a genealogical record."""
        self.records[record.record_id] = record

    def add_source(self, source: Source) -> None:
        """Add a source."""
        self.sources[source.source_id] = source

    def add_citation(self, citation: Citation) -> None:
        """Add a citation."""
        self.citations[citation.citation_id] = citation

    def add_finding(self, finding: str) -> None:
        """Record a research finding."""
        self.findings.append(finding)

    def add_conflict(self, conflict: str) -> None:
        """Record a conflict or discrepancy in evidence."""
        self.conflicts.append(conflict)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of research findings.

        Returns:
            Dictionary containing formatted report data
        """
        return {
            "agent_id": self.agent_id,
            "scope": self.scope,
            "status": self.status,
            "execution_time": self._calculate_execution_time(),
            "persons_found": len(self.persons),
            "relationships_established": len(self.relationships),
            "records_analyzed": len(self.records),
            "sources_consulted": len(self.sources),
            "citations_created": len(self.citations),
            "findings": self.findings,
            "conflicts": self.conflicts,
            "persons": {pid: str(p) for pid, p in self.persons.items()},
            "relationships": {rid: str(r) for rid, r in self.relationships.items()},
        }

    def _calculate_execution_time(self) -> str:
        """Calculate and format execution time."""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            return f"{duration.total_seconds():.2f} seconds"
        return "Not completed"

    def start_execution(self) -> None:
        """Mark the start of execution."""
        self.start_time = datetime.now()
        self.status = "executing"

    def complete_execution(self) -> None:
        """Mark the completion of execution."""
        self.end_time = datetime.now()
        self.status = "completed"
