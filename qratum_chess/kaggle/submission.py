"""Kaggle competition submission formatting and validation for QRATUM-Chess.

Handles strict CSV formatting, schema validation, and metadata injection.
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qratum_chess.core.position import Move, Position


@dataclass
class SubmissionMetadata:
    """Metadata for a QRATUM submission.

    Attributes:
        qratum_hash: Hash of engine configuration
        engine_version: Engine version string
        novelty_pressure: Novelty pressure parameter
        cortex_weights: Cortex weight configuration
        search_depth: Search depth used
        timestamp: Submission timestamp
        config_snapshot: Full configuration snapshot
    """

    qratum_hash: str
    engine_version: str
    novelty_pressure: float
    cortex_weights: dict[str, float]
    search_depth: int
    timestamp: float
    config_snapshot: dict[str, Any]


class SubmissionFormatter:
    """Formats QRATUM engine output for Kaggle competition submissions.

    Ensures strict column enforcement and proper CSV formatting.
    """

    def __init__(self):
        """Initialize submission formatter."""
        self.metadata: SubmissionMetadata | None = None

    def format_submission(
        self,
        predictions: list[dict[str, Any]],
        output_file: str | Path,
        metadata: SubmissionMetadata | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Path:
        """Format predictions as Kaggle-compliant CSV.

        Args:
            predictions: List of prediction dictionaries
            output_file: Output CSV file path
            metadata: Optional submission metadata
            schema: Optional schema definition

        Returns:
            Path to created submission file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.metadata = metadata

        # Default schema for chess competitions
        if schema is None:
            schema = {"columns": ["id", "move"], "required": True}

        # Write CSV with strict formatting
        with open(output_path, "w", newline="") as f:
            # Write metadata header as comments (if supported)
            if metadata:
                f.write(f"# QRATUM_HASH={metadata.qratum_hash}\n")
                f.write(f"# ENGINE_VERSION={metadata.engine_version}\n")
                f.write(f"# NOVELTY_PRESSURE={metadata.novelty_pressure}\n")
                f.write(
                    f"# CORTEX_WEIGHTS={','.join(f'{k}={v}' for k, v in metadata.cortex_weights.items())}\n"
                )
                f.write(f"# SEARCH_DEPTH={metadata.search_depth}\n")
                f.write(f"# TIMESTAMP={metadata.timestamp}\n")

            # Write CSV data
            writer = csv.DictWriter(f, fieldnames=schema["columns"])
            writer.writeheader()
            writer.writerows(predictions)

        print(f"✓ Submission formatted: {output_path}")
        print(f"  Predictions: {len(predictions)}")
        if metadata:
            print(f"  Engine hash: {metadata.qratum_hash[:12]}...")

        return output_path

    def convert_engine_output(
        self, positions: list[tuple[str, Position]], moves: list[Move], evaluations: list[float]
    ) -> list[dict[str, Any]]:
        """Convert QRATUM engine output to submission format.

        Args:
            positions: List of (id, Position) tuples
            moves: List of best moves
            evaluations: List of evaluation scores

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for (pos_id, position), move, eval_score in zip(positions, moves, evaluations):
            predictions.append(
                {"id": pos_id, "move": move.to_uci() if move else "0000", "evaluation": eval_score}
            )

        return predictions


class SubmissionValidator:
    """Validates Kaggle submissions before upload.

    Performs strict schema validation and legality checks.
    """

    def __init__(self):
        """Initialize submission validator."""
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_submission(
        self,
        submission_file: str | Path,
        positions: dict[str, Position] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> bool:
        """Validate a submission file.

        Args:
            submission_file: Path to submission CSV
            positions: Optional mapping of id -> Position for move validation
            schema: Optional schema definition

        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []

        submission_path = Path(submission_file)

        if not submission_path.exists():
            self.errors.append(f"Submission file not found: {submission_path}")
            return False

        # Default schema
        if schema is None:
            schema = {"columns": ["id", "move"], "required": True}

        try:
            with open(submission_path) as f:
                # Skip comment lines
                lines = [line for line in f if not line.startswith("#")]

                # Re-create file content without comments
                content = "".join(lines)

                # Parse CSV
                reader = csv.DictReader(content.splitlines())
                rows = list(reader)

            # Validate schema
            if not self._validate_schema(rows, schema):
                return False

            # Validate moves if positions provided
            if positions:
                if not self._validate_moves(rows, positions):
                    return False

            # Check for duplicates
            if not self._check_duplicates(rows):
                return False

            if self.warnings:
                print("Validation warnings:")
                for warning in self.warnings:
                    print(f"  ⚠ {warning}")

            if self.errors:
                print("Validation errors:")
                for error in self.errors:
                    print(f"  ✗ {error}")
                return False

            print(f"✓ Submission validated: {len(rows)} predictions")
            return True

        except Exception as e:
            self.errors.append(f"Validation error: {e}")
            return False

    def _validate_schema(self, rows: list[dict[str, Any]], schema: dict[str, Any]) -> bool:
        """Validate CSV schema.

        Args:
            rows: CSV rows
            schema: Schema definition

        Returns:
            True if valid
        """
        if not rows:
            self.errors.append("Empty submission")
            return False

        # Check columns
        required_columns = set(schema["columns"])
        actual_columns = set(rows[0].keys())

        missing = required_columns - actual_columns
        if missing:
            self.errors.append(f"Missing columns: {missing}")
            return False

        extra = actual_columns - required_columns
        if extra:
            self.warnings.append(f"Extra columns: {extra}")

        return True

    def _validate_moves(self, rows: list[dict[str, Any]], positions: dict[str, Position]) -> bool:
        """Validate move legality.

        Args:
            rows: CSV rows
            positions: Position mapping

        Returns:
            True if all moves are legal
        """
        illegal_count = 0

        for row in rows:
            pos_id = row.get("id")
            move_uci = row.get("move")

            if not pos_id or not move_uci:
                continue

            if pos_id not in positions:
                self.warnings.append(f"Unknown position ID: {pos_id}")
                continue

            position = positions[pos_id]

            # Parse move
            try:
                move = Move.from_uci(move_uci)
            except Exception:
                self.errors.append(f"Invalid move format: {move_uci} for position {pos_id}")
                illegal_count += 1
                continue

            # Check if move is legal
            legal_moves = position.generate_legal_moves()
            if move not in legal_moves:
                self.errors.append(f"Illegal move: {move_uci} for position {pos_id}")
                illegal_count += 1

        if illegal_count > 0:
            self.errors.append(f"Total illegal moves: {illegal_count}")
            return False

        return True

    def _check_duplicates(self, rows: list[dict[str, Any]]) -> bool:
        """Check for duplicate IDs.

        Args:
            rows: CSV rows

        Returns:
            True if no duplicates
        """
        ids = [row.get("id") for row in rows]
        unique_ids = set(ids)

        if len(ids) != len(unique_ids):
            duplicates = [id for id in ids if ids.count(id) > 1]
            self.errors.append(f"Duplicate IDs found: {set(duplicates)}")
            return False

        return True

    def get_errors(self) -> list[str]:
        """Get validation errors.

        Returns:
            List of error messages
        """
        return self.errors

    def get_warnings(self) -> list[str]:
        """Get validation warnings.

        Returns:
            List of warning messages
        """
        return self.warnings


def create_submission_metadata(engine, config: dict[str, Any]) -> SubmissionMetadata:
    """Create submission metadata from engine configuration.

    Args:
        engine: QRATUM engine instance
        config: Configuration dictionary

    Returns:
        Submission metadata
    """
    # Generate configuration hash
    config_str = str(sorted(config.items()))
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    # Extract engine parameters
    novelty_pressure = config.get("novelty_pressure", 0.0)
    search_depth = config.get("search_depth", 15)

    # Get cortex weights if available
    cortex_weights = config.get(
        "cortex_weights", {"tactical": 0.33, "strategic": 0.33, "conceptual": 0.34}
    )

    # Get engine version
    engine_version = getattr(engine, "__version__", "unknown")
    if engine_version == "unknown" and hasattr(engine, "__class__"):
        engine_version = engine.__class__.__name__

    return SubmissionMetadata(
        qratum_hash=config_hash,
        engine_version=engine_version,
        novelty_pressure=novelty_pressure,
        cortex_weights=cortex_weights,
        search_depth=search_depth,
        timestamp=time.time(),
        config_snapshot=config,
    )
