"""Snapshot capture functionality for QRATUM-Chess GUI.

Captures high-fidelity snapshots of all GUI panels, telemetry data,
cortex outputs, and quantum visualizations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qratum_chess.gui.panels.anti_holographic import AntiHolographicPanel
    from qratum_chess.gui.panels.board import BoardPanel
    from qratum_chess.gui.panels.control import ControlPanel
    from qratum_chess.gui.panels.motif_tracker import MotifTracker
    from qratum_chess.gui.panels.quantum import QuantumPanel
    from qratum_chess.gui.panels.search_tree import SearchTreePanel
    from qratum_chess.gui.panels.telemetry import TelemetryPanel
    from qratum_chess.gui.panels.tricortex import TriCortexPanel
    from qratum_chess.gui.snapshot.events import SnapshotEvent


class OutputFormat(Enum):
    """Snapshot output formats."""

    PNG = "png"
    JPEG = "jpeg"
    JSON = "json"
    CSV = "csv"
    HTML = "html"


@dataclass
class SnapshotConfig:
    """Configuration for snapshot capture.

    Attributes:
        output_dir: Base directory for snapshot storage
        image_format: Image format for visual captures
        image_quality: JPEG quality (1-100) if using JPEG
        resolution: Image resolution (width, height)
        capture_individual_panels: Capture each panel separately
        capture_combined_view: Capture full GUI combined view
        include_annotations: Add metadata annotations to images
        include_telemetry_csv: Export telemetry data as CSV
        include_cortex_json: Export cortex data as JSON
        auto_trigger: Enable automatic snapshot triggering
        eval_threshold: Evaluation change threshold for auto-trigger
        novelty_threshold: Novelty threshold for auto-trigger
        time_interval: Seconds between interval-based captures (0 to disable)
    """

    output_dir: Path = field(default_factory=lambda: Path("snapshots"))
    image_format: OutputFormat = OutputFormat.PNG
    image_quality: int = 95
    resolution: tuple[int, int] = (1920, 1080)
    capture_individual_panels: bool = True
    capture_combined_view: bool = True
    include_annotations: bool = True
    include_telemetry_csv: bool = True
    include_cortex_json: bool = True
    auto_trigger: bool = True
    eval_threshold: float = 1.0
    novelty_threshold: float = 0.3
    time_interval: float = 0.0


@dataclass
class PanelSnapshot:
    """Captured snapshot of a single panel.

    Attributes:
        panel_name: Name of the panel
        render_data: Panel render data (JSON-serializable)
        image_data: Base64-encoded image data (if visual capture)
        timestamp: Capture timestamp
    """

    panel_name: str
    render_data: dict[str, Any]
    image_data: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class FullSnapshot:
    """Complete snapshot of the QRATUM-Chess system.

    Attributes:
        event: Event that triggered the snapshot
        session_id: Unique session identifier
        snapshot_id: Unique snapshot identifier
        timestamp: Capture timestamp
        panels: Individual panel snapshots
        combined_image: Combined view image data
        metadata: Additional metadata
    """

    event: SnapshotEvent
    session_id: str
    snapshot_id: str
    timestamp: float = field(default_factory=time.time)
    panels: dict[str, PanelSnapshot] = field(default_factory=dict)
    combined_image: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event": self.event.to_dict(),
            "session_id": self.session_id,
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "timestamp_formatted": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)
            ),
            "panels": {
                name: {
                    "panel_name": ps.panel_name,
                    "render_data": ps.render_data,
                    "has_image": ps.image_data is not None,
                    "timestamp": ps.timestamp,
                }
                for name, ps in self.panels.items()
            },
            "has_combined_image": self.combined_image is not None,
            "metadata": self.metadata,
        }


class SnapshotCapture:
    """Captures high-fidelity snapshots of the QRATUM-Chess system.

    This class coordinates snapshot capture across all GUI panels,
    handling both visual captures (PNG/JPEG) and data exports (JSON/CSV).

    Features:
    - Individual panel capture
    - Combined full-GUI capture
    - Metadata annotation
    - Telemetry and cortex data export
    - Automatic triggering based on game events
    """

    def __init__(
        self,
        config: SnapshotConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize snapshot capture.

        Args:
            config: Snapshot configuration
            session_id: Unique session identifier
        """
        self.config = config or SnapshotConfig()
        self.session_id = session_id or self._generate_session_id()

        # Panel references (set via set_panels)
        self._board: BoardPanel | None = None
        self._tricortex: TriCortexPanel | None = None
        self._search_tree: SearchTreePanel | None = None
        self._motif_tracker: MotifTracker | None = None
        self._quantum: QuantumPanel | None = None
        self._anti_holographic: AntiHolographicPanel | None = None
        self._telemetry: TelemetryPanel | None = None
        self._control: ControlPanel | None = None

        # Snapshot counter
        self._snapshot_count = 0

        # Ensure output directory exists
        self._ensure_output_dir()

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"bob_{int(time.time())}_{np.random.randint(1000, 9999)}"

    def _ensure_output_dir(self) -> None:
        """Ensure output directory structure exists."""
        base = self.config.output_dir / self.session_id

        # Create subdirectories
        for subdir in [
            "board",
            "tricortex",
            "search_tree",
            "motif",
            "quantum",
            "anti_holographic",
            "telemetry",
            "combined",
            "data",
            "reports",
        ]:
            (base / subdir).mkdir(parents=True, exist_ok=True)

    def set_panels(
        self,
        board: BoardPanel | None = None,
        tricortex: TriCortexPanel | None = None,
        search_tree: SearchTreePanel | None = None,
        motif_tracker: MotifTracker | None = None,
        quantum: QuantumPanel | None = None,
        anti_holographic: AntiHolographicPanel | None = None,
        telemetry: TelemetryPanel | None = None,
        control: ControlPanel | None = None,
    ) -> None:
        """Set references to GUI panels.

        Args:
            board: Board panel instance
            tricortex: Tri-cortex diagnostics panel
            search_tree: Search tree explorer panel
            motif_tracker: Motif & novelty tracker panel
            quantum: Quantum panel
            anti_holographic: Anti-holographic indicator panel
            telemetry: Telemetry dashboard panel
            control: Control panel
        """
        self._board = board
        self._tricortex = tricortex
        self._search_tree = search_tree
        self._motif_tracker = motif_tracker
        self._quantum = quantum
        self._anti_holographic = anti_holographic
        self._telemetry = telemetry
        self._control = control

    def capture(self, event: SnapshotEvent) -> FullSnapshot:
        """Capture a full snapshot of the system.

        Args:
            event: Event that triggered the snapshot

        Returns:
            Complete snapshot of all panels and data
        """
        self._snapshot_count += 1
        snapshot_id = f"snap_{self._snapshot_count:06d}"

        snapshot = FullSnapshot(
            event=event,
            session_id=self.session_id,
            snapshot_id=snapshot_id,
        )

        # Capture individual panels
        if self.config.capture_individual_panels:
            snapshot.panels = self._capture_all_panels(snapshot_id)

        # Build metadata
        snapshot.metadata = self._build_metadata(event)

        # Save to disk
        self._save_snapshot(snapshot)

        return snapshot

    def _capture_all_panels(self, snapshot_id: str) -> dict[str, PanelSnapshot]:
        """Capture all panel data.

        Args:
            snapshot_id: Unique snapshot identifier

        Returns:
            Dictionary of panel snapshots
        """
        panels = {}

        if self._board:
            panels["board"] = PanelSnapshot(
                panel_name="board",
                render_data=self._board.get_render_data(),
            )

        if self._tricortex:
            panels["tricortex"] = PanelSnapshot(
                panel_name="tricortex",
                render_data=self._tricortex.get_render_data(),
            )

        if self._search_tree:
            panels["search_tree"] = PanelSnapshot(
                panel_name="search_tree",
                render_data=self._search_tree.get_render_data(),
            )

        if self._motif_tracker:
            panels["motif_tracker"] = PanelSnapshot(
                panel_name="motif_tracker",
                render_data=self._motif_tracker.get_render_data(),
            )

        if self._quantum and self._quantum.state.enabled:
            panels["quantum"] = PanelSnapshot(
                panel_name="quantum",
                render_data=self._quantum.get_render_data(),
            )

        if self._anti_holographic:
            panels["anti_holographic"] = PanelSnapshot(
                panel_name="anti_holographic",
                render_data=self._anti_holographic.get_render_data(),
            )

        if self._telemetry:
            panels["telemetry"] = PanelSnapshot(
                panel_name="telemetry",
                render_data=self._telemetry.get_render_data(),
            )

        if self._control:
            panels["control"] = PanelSnapshot(
                panel_name="control",
                render_data=self._control.get_render_data(),
            )

        return panels

    def _build_metadata(self, event: SnapshotEvent) -> dict[str, Any]:
        """Build metadata for snapshot.

        Args:
            event: Triggering event

        Returns:
            Metadata dictionary
        """
        metadata = {
            "session_id": self.session_id,
            "capture_config": {
                "resolution": self.config.resolution,
                "format": self.config.image_format.value,
                "individual_panels": self.config.capture_individual_panels,
                "combined_view": self.config.capture_combined_view,
            },
            "system_info": {
                "timestamp": time.time(),
                "snapshot_count": self._snapshot_count,
            },
        }

        # Add cortex contributions if available
        if self._tricortex:
            tricortex_data = self._tricortex.get_render_data()
            metadata["cortex_contributions"] = tricortex_data.get("contributions", {})

        # Add telemetry metrics if available
        if self._telemetry:
            telemetry_data = self._telemetry.get_render_data()
            metadata["telemetry"] = {
                "nps": telemetry_data.get("performance", {}).get("nps", 0),
                "hash_hit_rate": telemetry_data.get("cache", {}).get("hit_rate", 0),
                "threads": telemetry_data.get("threads", {}).get("active", 1),
            }

        # Add anti-holographic metrics if available
        if self._anti_holographic:
            ah_data = self._anti_holographic.get_render_data()
            metadata["anti_holographic"] = {
                "overall_score": ah_data.get("overall_score", 0),
                "stochasticity": ah_data.get("stochasticity", {}).get("current", 0),
                "destabilization": ah_data.get("destabilization", {}).get("score", 0),
            }

        # Add quantum metrics if available and enabled
        if self._quantum and self._quantum.state.enabled:
            quantum_data = self._quantum.get_render_data()
            metadata["quantum"] = {
                "enabled": True,
                "has_circuit": quantum_data.get("circuit") is not None,
                "has_qubo": quantum_data.get("qubo") is not None,
            }

        return metadata

    def _save_snapshot(self, snapshot: FullSnapshot) -> None:
        """Save snapshot to disk.

        Args:
            snapshot: Snapshot to save
        """
        base = self.config.output_dir / self.session_id
        snapshot_dir = base / "data" / snapshot.snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save main snapshot metadata
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

        # Save individual panel data
        for panel_name, panel_snapshot in snapshot.panels.items():
            panel_path = snapshot_dir / f"{panel_name}.json"
            with open(panel_path, "w") as f:
                json.dump(panel_snapshot.render_data, f, indent=2, default=str)

        # Export telemetry as CSV if configured
        if self.config.include_telemetry_csv and "telemetry" in snapshot.panels:
            self._export_telemetry_csv(snapshot, snapshot_dir)

        # Export cortex data as JSON if configured
        if self.config.include_cortex_json and "tricortex" in snapshot.panels:
            self._export_cortex_json(snapshot, snapshot_dir)

        # Log snapshot event
        self._log_snapshot_event(snapshot)

    def _export_telemetry_csv(self, snapshot: FullSnapshot, output_dir: Path) -> None:
        """Export telemetry data as CSV.

        Args:
            snapshot: Snapshot containing telemetry
            output_dir: Output directory
        """
        telemetry = snapshot.panels.get("telemetry")
        if not telemetry:
            return

        csv_path = output_dir / "telemetry.csv"

        data = telemetry.render_data
        rows = [
            ["metric", "value", "unit"],
            ["nodes_per_second", data.get("performance", {}).get("nps", 0), "nodes/s"],
            ["hash_hit_rate", data.get("cache", {}).get("hit_rate", 0), "ratio"],
            ["active_threads", data.get("threads", {}).get("active", 1), "count"],
            ["nn_latency", data.get("latency", {}).get("nn_eval_ms", 0), "ms"],
            ["memory_used", data.get("memory", {}).get("heap_used_mb", 0), "MB"],
        ]

        with open(csv_path, "w") as f:
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")

    def _export_cortex_json(self, snapshot: FullSnapshot, output_dir: Path) -> None:
        """Export cortex data as JSON.

        Args:
            snapshot: Snapshot containing cortex data
            output_dir: Output directory
        """
        tricortex = snapshot.panels.get("tricortex")
        if not tricortex:
            return

        json_path = output_dir / "cortex_data.json"

        data = {
            "contributions": tricortex.render_data.get("contributions", {}),
            "entropy": tricortex.render_data.get("entropy", {}),
            "heatmaps": tricortex.render_data.get("heatmaps", {}),
            "top_moves": tricortex.render_data.get("top_moves", []),
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _log_snapshot_event(self, snapshot: FullSnapshot) -> None:
        """Log snapshot event to event log.

        Args:
            snapshot: Captured snapshot
        """
        base = self.config.output_dir / self.session_id
        log_path = base / "snapshot_log.jsonl"

        log_entry = {
            "snapshot_id": snapshot.snapshot_id,
            "timestamp": snapshot.timestamp,
            "trigger": snapshot.event.trigger.value,
            "move_number": snapshot.event.move_number,
            "description": snapshot.event.description,
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def generate_report(
        self,
        snapshots: list[FullSnapshot],
        format: OutputFormat = OutputFormat.HTML,
    ) -> Path:
        """Generate a report from captured snapshots.

        Args:
            snapshots: List of snapshots to include
            format: Report output format

        Returns:
            Path to generated report
        """
        base = self.config.output_dir / self.session_id / "reports"
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if format == OutputFormat.HTML:
            return self._generate_html_report(snapshots, base, timestamp)
        elif format == OutputFormat.JSON:
            return self._generate_json_report(snapshots, base, timestamp)
        else:
            return self._generate_json_report(snapshots, base, timestamp)

    def _generate_html_report(
        self,
        snapshots: list[FullSnapshot],
        output_dir: Path,
        timestamp: str,
    ) -> Path:
        """Generate HTML report.

        Args:
            snapshots: Snapshots to include
            output_dir: Output directory
            timestamp: Report timestamp

        Returns:
            Path to report
        """
        report_path = output_dir / f"report_{timestamp}.html"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>QRATUM-Chess Snapshot Report - {timestamp}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #0a0a0f; color: #fff; }}
        h1 {{ color: #00f5ff; }}
        h2 {{ color: #7b2cbf; border-bottom: 1px solid #333; padding-bottom: 10px; }}
        .snapshot {{ background: rgba(15,15,25,0.9); border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .metric {{ display: inline-block; background: #1a1a2e; padding: 10px; margin: 5px; border-radius: 4px; }}
        .metric-label {{ color: #888; font-size: 0.9em; }}
        .metric-value {{ color: #00f5ff; font-size: 1.2em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #333; }}
        th {{ color: #00f5ff; }}
        .trigger {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }}
        .trigger-evaluation_shift {{ background: #ff6b6b; }}
        .trigger-motif_discovery {{ background: #7b2cbf; }}
        .trigger-phase_transition {{ background: #00f5ff; color: #000; }}
        .trigger-novelty_detection {{ background: #00ff88; color: #000; }}
    </style>
</head>
<body>
    <h1>🤖 QRATUM-Chess "Bob" Snapshot Report</h1>
    <p>Session: {self.session_id}</p>
    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total Snapshots: {len(snapshots)}</p>
    
    <h2>📸 Captured Snapshots</h2>
"""

        for snap in snapshots:
            trigger_class = f"trigger-{snap.event.trigger.value}"
            html_content += f"""
    <div class="snapshot">
        <h3>{snap.snapshot_id} - <span class="trigger {trigger_class}">{snap.event.trigger.value}</span></h3>
        <p>{snap.event.description}</p>
        <p>Move: {snap.event.move_number} | Phase: {snap.event.phase.value} | Eval: {snap.event.evaluation:.2f}</p>
        
        <div class="metrics">
"""
            if "cortex_contributions" in snap.metadata:
                contrib = snap.metadata["cortex_contributions"]
                html_content += f"""
            <div class="metric">
                <div class="metric-label">Tactical</div>
                <div class="metric-value">{contrib.get('tactical', 0)*100:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Strategic</div>
                <div class="metric-value">{contrib.get('strategic', 0)*100:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Conceptual</div>
                <div class="metric-value">{contrib.get('conceptual', 0)*100:.1f}%</div>
            </div>
"""
            html_content += """
        </div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        with open(report_path, "w") as f:
            f.write(html_content)

        return report_path

    def _generate_json_report(
        self,
        snapshots: list[FullSnapshot],
        output_dir: Path,
        timestamp: str,
    ) -> Path:
        """Generate JSON report.

        Args:
            snapshots: Snapshots to include
            output_dir: Output directory
            timestamp: Report timestamp

        Returns:
            Path to report
        """
        report_path = output_dir / f"report_{timestamp}.json"

        report_data = {
            "session_id": self.session_id,
            "generated": time.time(),
            "total_snapshots": len(snapshots),
            "snapshots": [s.to_dict() for s in snapshots],
        }

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_path
