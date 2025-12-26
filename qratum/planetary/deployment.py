"""
Deployment Roadmap Module

Implements phased deployment strategy from seed network to
exponential global scaling with adaptive nodes and revenue loops.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PhaseType(Enum):
    """Deployment phase types."""

    PHASE_I = "phase_1_seed"
    PHASE_II = "phase_2_integration"
    PHASE_III = "phase_3_governance"
    PHASE_IV = "phase_4_scaling"


class MilestoneStatus(Enum):
    """Milestone completion status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


@dataclass
class Milestone:
    """Deployment milestone.

    Attributes:
        milestone_id: Unique milestone identifier
        name: Milestone name
        description: Milestone description
        phase: Phase this milestone belongs to
        status: Current status
        target_date: Target completion date
        completion_date: Actual completion date
        dependencies: List of dependent milestone IDs
        metrics: Success metrics
    """

    milestone_id: str
    name: str
    description: str
    phase: PhaseType
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED
    target_date: str = ""
    completion_date: str = ""
    dependencies: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def mark_in_progress(self) -> None:
        """Mark milestone as in progress."""
        self.status = MilestoneStatus.IN_PROGRESS

    def mark_completed(self) -> None:
        """Mark milestone as completed."""
        self.status = MilestoneStatus.COMPLETED
        self.completion_date = datetime.now(timezone.utc).isoformat()

    def mark_blocked(self, reason: str = "") -> None:
        """Mark milestone as blocked.

        Args:
            reason: Blocking reason
        """
        self.status = MilestoneStatus.BLOCKED
        self.metrics["block_reason"] = reason

    def is_ready(self, completed_milestones: set[str]) -> bool:
        """Check if milestone dependencies are met.

        Args:
            completed_milestones: Set of completed milestone IDs

        Returns:
            True if all dependencies are met
        """
        return all(dep in completed_milestones for dep in self.dependencies)

    def to_dict(self) -> dict[str, Any]:
        """Serialize milestone to dictionary."""
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "description": self.description,
            "phase": self.phase.value,
            "status": self.status.value,
            "target_date": self.target_date,
            "completion_date": self.completion_date,
            "dependencies": self.dependencies,
            "metrics": self.metrics,
        }


@dataclass
class DeploymentPhase:
    """Deployment phase with milestones and metrics.

    Attributes:
        phase_type: Type of deployment phase
        name: Phase name
        description: Phase description
        milestones: List of milestones in this phase
        start_date: Phase start date
        target_end_date: Target end date
        actual_end_date: Actual end date
        success_criteria: Criteria for phase success
        kpis: Key performance indicators
    """

    phase_type: PhaseType
    name: str
    description: str
    milestones: list[Milestone] = field(default_factory=list)
    start_date: str = ""
    target_end_date: str = ""
    actual_end_date: str = ""
    success_criteria: dict[str, Any] = field(default_factory=dict)
    kpis: dict[str, float] = field(default_factory=dict)

    def add_milestone(self, milestone: Milestone) -> None:
        """Add a milestone to the phase.

        Args:
            milestone: Milestone to add
        """
        self.milestones.append(milestone)

    def get_progress(self) -> float:
        """Calculate phase completion progress.

        Returns:
            Progress percentage (0-100)
        """
        if not self.milestones:
            return 0.0

        completed = sum(1 for m in self.milestones if m.status == MilestoneStatus.COMPLETED)
        return (completed / len(self.milestones)) * 100

    def is_complete(self) -> bool:
        """Check if phase is complete.

        Returns:
            True if all milestones are completed
        """
        return all(m.status == MilestoneStatus.COMPLETED for m in self.milestones)

    def complete_phase(self) -> None:
        """Mark phase as complete."""
        self.actual_end_date = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize phase to dictionary."""
        return {
            "phase_type": self.phase_type.value,
            "name": self.name,
            "description": self.description,
            "milestones": [m.to_dict() for m in self.milestones],
            "start_date": self.start_date,
            "target_end_date": self.target_end_date,
            "actual_end_date": self.actual_end_date,
            "progress": self.get_progress(),
            "success_criteria": self.success_criteria,
            "kpis": self.kpis,
        }


@dataclass
class MilestoneTracker:
    """Tracks milestones across all phases.

    Attributes:
        tracker_id: Unique tracker identifier
        milestones: Dictionary of all milestones
        dependencies_graph: Milestone dependency graph
        events: List of tracking events
    """

    tracker_id: str
    milestones: dict[str, Milestone] = field(default_factory=dict)
    dependencies_graph: dict[str, list[str]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def register_milestone(self, milestone: Milestone) -> None:
        """Register a milestone.

        Args:
            milestone: Milestone to register
        """
        self.milestones[milestone.milestone_id] = milestone
        self.dependencies_graph[milestone.milestone_id] = milestone.dependencies

    def update_status(self, milestone_id: str, status: MilestoneStatus) -> None:
        """Update milestone status.

        Args:
            milestone_id: Milestone identifier
            status: New status
        """
        if milestone_id in self.milestones:
            self.milestones[milestone_id].status = status
            if status == MilestoneStatus.COMPLETED:
                self.milestones[milestone_id].mark_completed()

            self.events.append(
                {
                    "event_type": "status_change",
                    "milestone_id": milestone_id,
                    "new_status": status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def get_ready_milestones(self) -> list[Milestone]:
        """Get milestones ready to start.

        Returns:
            List of ready milestones
        """
        completed = {
            mid for mid, m in self.milestones.items() if m.status == MilestoneStatus.COMPLETED
        }

        ready = []
        for mid, milestone in self.milestones.items():
            if milestone.status == MilestoneStatus.NOT_STARTED:
                if milestone.is_ready(completed):
                    ready.append(milestone)

        return ready

    def get_blocked_milestones(self) -> list[Milestone]:
        """Get blocked milestones.

        Returns:
            List of blocked milestones
        """
        return [m for m in self.milestones.values() if m.status == MilestoneStatus.BLOCKED]

    def get_critical_path(self) -> list[str]:
        """Calculate critical path through milestones.

        Returns:
            List of milestone IDs on critical path
        """
        # Simple critical path: milestones with most dependents
        dependents_count = dict.fromkeys(self.milestones, 0)

        for deps in self.dependencies_graph.values():
            for dep in deps:
                if dep in dependents_count:
                    dependents_count[dep] += 1

        # Sort by dependents (descending) and return
        sorted_milestones = sorted(dependents_count.items(), key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in sorted_milestones[:5]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker to dictionary."""
        return {
            "tracker_id": self.tracker_id,
            "milestones": {k: v.to_dict() for k, v in self.milestones.items()},
            "dependencies_graph": self.dependencies_graph,
            "events": self.events[-50:],
            "critical_path": self.get_critical_path(),
        }


class DeploymentRoadmap:
    """Full deployment roadmap manager.

    Manages the complete deployment lifecycle across all phases.

    Attributes:
        roadmap_id: Unique roadmap identifier
        phases: Dictionary of deployment phases
        milestone_tracker: Milestone tracking system
        created_at: Roadmap creation timestamp
    """

    def __init__(self, roadmap_id: str | None = None) -> None:
        """Initialize deployment roadmap.

        Args:
            roadmap_id: Optional roadmap ID
        """
        self.roadmap_id = roadmap_id or self._generate_id()
        self.phases: dict[PhaseType, DeploymentPhase] = {}
        self.milestone_tracker = MilestoneTracker(tracker_id=f"tracker_{self.roadmap_id}")
        self.created_at = datetime.now(timezone.utc).isoformat()

        self._initialize_phases()

    def _generate_id(self) -> str:
        """Generate unique roadmap ID."""
        return f"roadmap_{secrets.token_hex(8)}"

    def _initialize_phases(self) -> None:
        """Initialize all deployment phases with milestones."""
        # Phase I: Seed Network Deployment
        phase1 = DeploymentPhase(
            phase_type=PhaseType.PHASE_I,
            name="Seed Network Deployment",
            description="Initial network deployment and data validation setup",
            success_criteria={
                "min_nodes": 10,
                "min_validators": 5,
                "data_contracts": 3,
                "uptime_percent": 95,
            },
        )
        self._add_phase1_milestones(phase1)
        self.phases[PhaseType.PHASE_I] = phase1

        # Phase II: Cross-Sector Integration
        phase2 = DeploymentPhase(
            phase_type=PhaseType.PHASE_II,
            name="Cross-Sector Integration",
            description="Multi-domain adoption and integration",
            success_criteria={
                "domains_integrated": 4,
                "active_integrations": 20,
                "data_throughput_gbps": 100,
            },
        )
        self._add_phase2_milestones(phase2)
        self.phases[PhaseType.PHASE_II] = phase2

        # Phase III: Autonomous AI Governance
        phase3 = DeploymentPhase(
            phase_type=PhaseType.PHASE_III,
            name="Autonomous AI Governance",
            description="AI-driven governance and self-optimizing operations",
            success_criteria={
                "ai_governance_nodes": 10,
                "autonomous_decisions_daily": 1000,
                "optimization_improvement_percent": 20,
            },
        )
        self._add_phase3_milestones(phase3)
        self.phases[PhaseType.PHASE_III] = phase3

        # Phase IV: Exponential Global Scaling
        phase4 = DeploymentPhase(
            phase_type=PhaseType.PHASE_IV,
            name="Exponential Global Scaling",
            description="Planet-scale deployment with adaptive nodes",
            success_criteria={
                "global_nodes": 10000,
                "coverage_percent": 95,
                "revenue_usd_annual": 1_000_000_000,
            },
        )
        self._add_phase4_milestones(phase4)
        self.phases[PhaseType.PHASE_IV] = phase4

    def _add_phase1_milestones(self, phase: DeploymentPhase) -> None:
        """Add Phase I milestones."""
        milestones = [
            Milestone(
                milestone_id="p1_m1_core_infra",
                name="Core Infrastructure Setup",
                description="Deploy initial satellite and terrestrial nodes",
                phase=PhaseType.PHASE_I,
                metrics={"nodes_deployed": 0, "target": 10},
            ),
            Milestone(
                milestone_id="p1_m2_validators",
                name="Validator Network",
                description="Establish validator node network",
                phase=PhaseType.PHASE_I,
                dependencies=["p1_m1_core_infra"],
                metrics={"validators": 0, "target": 5},
            ),
            Milestone(
                milestone_id="p1_m3_data_contracts",
                name="Data Contracts",
                description="Deploy initial Proof-of-Data contracts",
                phase=PhaseType.PHASE_I,
                dependencies=["p1_m2_validators"],
                metrics={"contracts": 0, "target": 3},
            ),
            Milestone(
                milestone_id="p1_m4_monitoring",
                name="Monitoring System",
                description="Deploy infrastructure monitoring",
                phase=PhaseType.PHASE_I,
                dependencies=["p1_m1_core_infra"],
                metrics={"uptime_percent": 0, "target": 95},
            ),
        ]

        for m in milestones:
            phase.add_milestone(m)
            self.milestone_tracker.register_milestone(m)

    def _add_phase2_milestones(self, phase: DeploymentPhase) -> None:
        """Add Phase II milestones."""
        milestones = [
            Milestone(
                milestone_id="p2_m1_energy_integration",
                name="Energy Sector Integration",
                description="Integrate with energy grid systems",
                phase=PhaseType.PHASE_II,
                dependencies=["p1_m3_data_contracts"],
                metrics={"integrations": 0, "target": 5},
            ),
            Milestone(
                milestone_id="p2_m2_healthcare_integration",
                name="Healthcare Integration",
                description="Integrate with healthcare systems (HIPAA compliant)",
                phase=PhaseType.PHASE_II,
                dependencies=["p1_m3_data_contracts"],
                metrics={"integrations": 0, "target": 5},
            ),
            Milestone(
                milestone_id="p2_m3_finance_integration",
                name="Finance Integration",
                description="Integrate with financial systems (PCI-DSS)",
                phase=PhaseType.PHASE_II,
                dependencies=["p1_m3_data_contracts"],
                metrics={"integrations": 0, "target": 5},
            ),
            Milestone(
                milestone_id="p2_m4_logistics_integration",
                name="Logistics Integration",
                description="Integrate with supply chain systems",
                phase=PhaseType.PHASE_II,
                dependencies=["p1_m3_data_contracts"],
                metrics={"integrations": 0, "target": 5},
            ),
            Milestone(
                milestone_id="p2_m5_interop_layer",
                name="Interoperability Layer",
                description="Deploy cross-domain interoperability",
                phase=PhaseType.PHASE_II,
                dependencies=[
                    "p2_m1_energy_integration",
                    "p2_m2_healthcare_integration",
                    "p2_m3_finance_integration",
                ],
                metrics={"domains_connected": 0, "target": 4},
            ),
        ]

        for m in milestones:
            phase.add_milestone(m)
            self.milestone_tracker.register_milestone(m)

    def _add_phase3_milestones(self, phase: DeploymentPhase) -> None:
        """Add Phase III milestones."""
        milestones = [
            Milestone(
                milestone_id="p3_m1_ai_governance",
                name="AI Governance Nodes",
                description="Deploy AI governance decision nodes",
                phase=PhaseType.PHASE_III,
                dependencies=["p2_m5_interop_layer"],
                metrics={"ai_nodes": 0, "target": 10},
            ),
            Milestone(
                milestone_id="p3_m2_symbolic_attractors",
                name="Symbolic Attractor System",
                description="Deploy symbolic policy optimization",
                phase=PhaseType.PHASE_III,
                dependencies=["p3_m1_ai_governance"],
                metrics={"attractors": 0, "target": 5},
            ),
            Milestone(
                milestone_id="p3_m3_autonomous_ops",
                name="Autonomous Operations",
                description="Enable self-optimizing operations",
                phase=PhaseType.PHASE_III,
                dependencies=["p3_m1_ai_governance", "p3_m2_symbolic_attractors"],
                metrics={"autonomous_decisions": 0, "target": 1000},
            ),
            Milestone(
                milestone_id="p3_m4_feedback_loops",
                name="Feedback Loop Integration",
                description="Establish recursive feedback systems",
                phase=PhaseType.PHASE_III,
                dependencies=["p3_m3_autonomous_ops"],
                metrics={"feedback_loops": 0, "target": 10},
            ),
        ]

        for m in milestones:
            phase.add_milestone(m)
            self.milestone_tracker.register_milestone(m)

    def _add_phase4_milestones(self, phase: DeploymentPhase) -> None:
        """Add Phase IV milestones."""
        milestones = [
            Milestone(
                milestone_id="p4_m1_global_expansion",
                name="Global Node Expansion",
                description="Deploy nodes across all continents",
                phase=PhaseType.PHASE_IV,
                dependencies=["p3_m4_feedback_loops"],
                metrics={"nodes": 0, "target": 10000},
            ),
            Milestone(
                milestone_id="p4_m2_satellite_constellation",
                name="Full Satellite Constellation",
                description="Deploy complete satellite coverage",
                phase=PhaseType.PHASE_IV,
                dependencies=["p4_m1_global_expansion"],
                metrics={"satellites": 0, "target": 1000},
            ),
            Milestone(
                milestone_id="p4_m3_revenue_scaling",
                name="Revenue Scaling",
                description="Achieve target annual revenue",
                phase=PhaseType.PHASE_IV,
                dependencies=["p4_m1_global_expansion"],
                metrics={"revenue_usd": 0, "target": 1_000_000_000},
            ),
            Milestone(
                milestone_id="p4_m4_adaptive_network",
                name="Adaptive Network Operations",
                description="Full adaptive self-healing network",
                phase=PhaseType.PHASE_IV,
                dependencies=[
                    "p4_m1_global_expansion",
                    "p4_m2_satellite_constellation",
                ],
                metrics={"self_heal_events": 0, "uptime_percent": 99.99},
            ),
        ]

        for m in milestones:
            phase.add_milestone(m)
            self.milestone_tracker.register_milestone(m)

    def start_phase(self, phase_type: PhaseType) -> bool:
        """Start a deployment phase.

        Args:
            phase_type: Phase to start

        Returns:
            True if phase started successfully
        """
        if phase_type not in self.phases:
            return False

        phase = self.phases[phase_type]
        phase.start_date = datetime.now(timezone.utc).isoformat()

        # Start first available milestones
        for milestone in phase.milestones:
            completed = {
                m.milestone_id
                for m in self.milestone_tracker.milestones.values()
                if m.status == MilestoneStatus.COMPLETED
            }
            if milestone.is_ready(completed):
                milestone.mark_in_progress()

        return True

    def complete_milestone(self, milestone_id: str) -> bool:
        """Complete a milestone.

        Args:
            milestone_id: Milestone to complete

        Returns:
            True if milestone completed successfully
        """
        self.milestone_tracker.update_status(milestone_id, MilestoneStatus.COMPLETED)

        # Check if this enables other milestones
        for milestone in self.milestone_tracker.get_ready_milestones():
            if milestone.status == MilestoneStatus.NOT_STARTED:
                milestone.mark_in_progress()

        # Check if any phase is now complete
        for phase in self.phases.values():
            if phase.is_complete() and not phase.actual_end_date:
                phase.complete_phase()

        return True

    def get_overall_progress(self) -> dict[str, Any]:
        """Get overall roadmap progress.

        Returns:
            Progress summary across all phases
        """
        total_milestones = sum(len(p.milestones) for p in self.phases.values())
        completed_milestones = sum(
            1
            for m in self.milestone_tracker.milestones.values()
            if m.status == MilestoneStatus.COMPLETED
        )

        phases_progress = {}
        for phase_type, phase in self.phases.items():
            phases_progress[phase_type.value] = {
                "name": phase.name,
                "progress": phase.get_progress(),
                "is_complete": phase.is_complete(),
                "milestones_total": len(phase.milestones),
            }

        return {
            "roadmap_id": self.roadmap_id,
            "overall_progress": (
                (completed_milestones / total_milestones) * 100 if total_milestones > 0 else 0
            ),
            "total_milestones": total_milestones,
            "completed_milestones": completed_milestones,
            "phases": phases_progress,
            "critical_path": self.milestone_tracker.get_critical_path(),
            "blocked_milestones": len(self.milestone_tracker.get_blocked_milestones()),
        }

    def get_next_milestones(self) -> list[dict[str, Any]]:
        """Get next milestones to work on.

        Returns:
            List of ready and in-progress milestones
        """
        ready = self.milestone_tracker.get_ready_milestones()
        in_progress = [
            m
            for m in self.milestone_tracker.milestones.values()
            if m.status == MilestoneStatus.IN_PROGRESS
        ]

        return [
            {
                "milestone_id": m.milestone_id,
                "name": m.name,
                "phase": m.phase.value,
                "status": m.status.value,
                "dependencies_met": m.is_ready(
                    {
                        mid
                        for mid, ms in self.milestone_tracker.milestones.items()
                        if ms.status == MilestoneStatus.COMPLETED
                    }
                ),
            }
            for m in ready + in_progress
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize roadmap to dictionary."""
        return {
            "roadmap_id": self.roadmap_id,
            "phases": {k.value: v.to_dict() for k, v in self.phases.items()},
            "milestone_tracker": self.milestone_tracker.to_dict(),
            "overall_progress": self.get_overall_progress(),
            "created_at": self.created_at,
        }
