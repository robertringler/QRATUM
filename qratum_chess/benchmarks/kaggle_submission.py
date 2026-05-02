"""Kaggle Submission Module for QRATUM Chess Benchmarking.

Handles submission of QRATUM engine results to Kaggle competitions,
including authentication, validation, and leaderboard interaction.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from qratum_chess.benchmarks.kaggle_config import KaggleConfig


@dataclass
class SubmissionResult:
    """Result of a Kaggle submission.

    Attributes:
        success: Whether submission was successful.
        submission_id: Kaggle submission ID (if successful).
        message: Status message.
        score: Competition score (if available).
        leaderboard_position: Position on leaderboard (if available).
        timestamp: Submission timestamp.
        error: Error message (if failed).
    """

    success: bool
    submission_id: str | None = None
    message: str = ""
    score: float | None = None
    leaderboard_position: int | None = None
    timestamp: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "success": self.success,
            "submission_id": self.submission_id,
            "message": self.message,
            "score": self.score,
            "leaderboard_position": self.leaderboard_position,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "error": self.error,
        }


class KaggleSubmission:
    """Handles submission to Kaggle competitions.

    Manages the complete submission workflow including formatting,
    validation, submission, and result retrieval.
    """

    def __init__(self, config: KaggleConfig):
        """Initialize Kaggle submission handler.

        Args:
            config: Kaggle configuration.
        """
        self.config = config

    def format_results_for_submission(self, results: list[dict[str, Any]]) -> str:
        """Format QRATUM results for Kaggle submission.

        Args:
            results: List of benchmark results.

        Returns:
            Formatted submission string (CSV or JSON).
        """
        if self.config.submission_format.format_type == "csv":
            return self._format_as_csv(results)
        elif self.config.submission_format.format_type == "json":
            return self._format_as_json(results)
        else:
            raise ValueError(f"Unsupported format: {self.config.submission_format.format_type}")

    def _format_as_csv(self, results: list[dict[str, Any]]) -> str:
        """Format results as CSV string.

        Args:
            results: List of benchmark results.

        Returns:
            CSV string.
        """
        output = io.StringIO()
        headers = self.config.submission_format.headers

        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()

        for result in results:
            # Extract only the required fields
            row = {field: result.get(field, "") for field in headers}
            writer.writerow(row)

        return output.getvalue()

    def _format_as_json(self, results: list[dict[str, Any]]) -> str:
        """Format results as JSON string.

        Args:
            results: List of benchmark results.

        Returns:
            JSON string.
        """
        return json.dumps(results, indent=2)

    def validate_submission(self, results: list[dict[str, Any]]) -> tuple[bool, str]:
        """Validate submission data before posting.

        Args:
            results: List of benchmark results.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Use format validator
        is_valid, error = self.config.submission_format.validate_submission_data(results)

        if not is_valid:
            return False, error

        # Additional validation
        if not results:
            return False, "No results to submit"

        # Check for duplicate position IDs
        position_ids = [r.get("position_id") for r in results]
        if len(position_ids) != len(set(position_ids)):
            return False, "Duplicate position IDs found"

        return True, ""

    def submit_to_kaggle(
        self,
        results: list[dict[str, Any]],
        message: str = "QRATUM Chess Engine Submission",
        dry_run: bool = False,
    ) -> SubmissionResult:
        """Submit results to Kaggle competition.

        Args:
            results: List of benchmark results.
            message: Submission message/description.
            dry_run: If True, validate but don't actually submit.

        Returns:
            SubmissionResult with status and details.
        """
        # Validate submission
        is_valid, error = self.validate_submission(results)
        if not is_valid:
            return SubmissionResult(
                success=False, message="Validation failed", error=error, timestamp=datetime.now()
            )

        # Format submission data
        try:
            submission_data = self.format_results_for_submission(results)
        except Exception as e:
            return SubmissionResult(
                success=False, message="Formatting failed", error=str(e), timestamp=datetime.now()
            )

        # Dry run - return success without submitting
        if dry_run:
            return SubmissionResult(
                success=True, message="Dry run successful (not submitted)", timestamp=datetime.now()
            )

        # Submit to Kaggle API
        try:
            result = self._post_to_kaggle_api(submission_data, message)
            return result
        except Exception as e:
            return SubmissionResult(
                success=False, message="Submission failed", error=str(e), timestamp=datetime.now()
            )

    def _post_to_kaggle_api(self, submission_data: str, message: str) -> SubmissionResult:
        """Post submission to Kaggle API.

        Args:
            submission_data: Formatted submission data.
            message: Submission message.

        Returns:
            SubmissionResult.
        """
        # Get API endpoint
        endpoint = self.config.competition.get_submission_endpoint()

        # Prepare request
        headers = self.config.get_auth_headers()

        # For file upload, we need multipart/form-data
        files = {"file": ("submission.csv", submission_data, "text/csv")}

        # Remove Content-Type header as requests will set it for multipart
        if "Content-Type" in headers:
            del headers["Content-Type"]

        data = {"message": message}

        # Make API request
        try:
            response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)

            response.raise_for_status()

            # Parse response
            result_data = response.json()

            return SubmissionResult(
                success=True,
                submission_id=result_data.get("id", None),
                message="Submission successful",
                timestamp=datetime.now(),
            )

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            return SubmissionResult(
                success=False,
                message="API request failed",
                error=error_msg,
                timestamp=datetime.now(),
            )

        except requests.exceptions.RequestException as e:
            return SubmissionResult(
                success=False, message="Network error", error=str(e), timestamp=datetime.now()
            )

    def get_submission_status(self, submission_id: str, timeout: int = 300) -> SubmissionResult:
        """Get status of a submission.

        Args:
            submission_id: Kaggle submission ID.
            timeout: Maximum time to wait for results (seconds).

        Returns:
            SubmissionResult with updated status.
        """
        endpoint = (
            f"https://www.kaggle.com/api/v1/competitions/"
            f"{self.config.competition.competition_id}/submissions/{submission_id}"
        )

        headers = self.config.get_auth_headers()

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(endpoint, headers=headers, timeout=10)
                response.raise_for_status()

                data = response.json()
                status = data.get("status", "unknown")

                if status == "complete":
                    return SubmissionResult(
                        success=True,
                        submission_id=submission_id,
                        message="Submission complete",
                        score=data.get("publicScore", None),
                        timestamp=datetime.now(),
                    )
                elif status == "error":
                    return SubmissionResult(
                        success=False,
                        submission_id=submission_id,
                        message="Submission failed",
                        error=data.get("errorDescription", "Unknown error"),
                        timestamp=datetime.now(),
                    )

                # Still processing, wait and retry
                time.sleep(5)

            except requests.RequestException as e:
                return SubmissionResult(
                    success=False,
                    submission_id=submission_id,
                    message="Failed to get status",
                    error=str(e),
                    timestamp=datetime.now(),
                )

        # Timeout
        return SubmissionResult(
            success=False,
            submission_id=submission_id,
            message="Timeout waiting for results",
            timestamp=datetime.now(),
        )

    def get_leaderboard_position(
        self, username: str | None = None
    ) -> tuple[int | None, float | None]:
        """Get current leaderboard position.

        Args:
            username: Kaggle username. Uses config username if None.

        Returns:
            Tuple of (position, score) or (None, None) if not found.
        """
        if username is None:
            username = self.config.credentials.username

        endpoint = (
            f"https://www.kaggle.com/api/v1/competitions/"
            f"{self.config.competition.competition_id}/leaderboard"
        )

        headers = self.config.get_auth_headers()

        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Find user in leaderboard
            for entry in data.get("submissions", []):
                if entry.get("teamName") == username:
                    return entry.get("position"), entry.get("score")

            return None, None

        except requests.RequestException:
            return None, None

    def display_submission_summary(self, result: SubmissionResult) -> None:
        """Display submission result summary.

        Args:
            result: SubmissionResult to display.
        """
        print("\n" + "=" * 70)
        print("Kaggle Submission Summary")
        print("=" * 70)

        if result.success:
            print("✓ Status: SUCCESS")
            print(f"  Submission ID: {result.submission_id}")
            print(f"  Message: {result.message}")

            if result.score is not None:
                print(f"  Score: {result.score:.4f}")

            if result.leaderboard_position is not None:
                print(f"  Leaderboard Position: #{result.leaderboard_position}")

            # Try to get current leaderboard position
            position, score = self.get_leaderboard_position()
            if position is not None:
                print("\n  Current Leaderboard Standing:")
                print(f"    Position: #{position}")
                print(f"    Score: {score:.4f}")
        else:
            print("✗ Status: FAILED")
            print(f"  Message: {result.message}")
            if result.error:
                print(f"  Error: {result.error}")

        print("=" * 70 + "\n")
