"""Kaggle API Configuration for QRATUM Chess Benchmarking.

Manages Kaggle API credentials, competition IDs, and submission formats.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class KaggleCredentials:
    """Kaggle API credentials.

    Attributes:
        username: Kaggle username.
        key: Kaggle API key.
    """

    username: str
    key: str

    @classmethod
    def from_file(cls, path: str | None = None) -> KaggleCredentials:
        """Load credentials from kaggle.json file.

        Args:
            path: Path to kaggle.json. Defaults to ~/.kaggle/kaggle.json

        Returns:
            KaggleCredentials instance.

        Raises:
            FileNotFoundError: If credentials file doesn't exist.
            ValueError: If credentials file is invalid.
        """
        if path is None:
            path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Kaggle credentials not found at {path}. "
                f"Please download your credentials from https://www.kaggle.com/settings "
                f"and place them at ~/.kaggle/kaggle.json"
            )

        with open(path) as f:
            data = json.load(f)

        if "username" not in data or "key" not in data:
            raise ValueError(
                f"Invalid credentials file at {path}. " f"Expected 'username' and 'key' fields."
            )

        return cls(username=data["username"], key=data["key"])

    @classmethod
    def from_env(cls) -> KaggleCredentials:
        """Load credentials from environment variables.

        Returns:
            KaggleCredentials instance.

        Raises:
            ValueError: If environment variables are not set.
        """
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")

        if not username or not key:
            raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set")

        return cls(username=username, key=key)


@dataclass
class KaggleCompetitionConfig:
    """Configuration for Kaggle competition.

    Attributes:
        competition_id: Kaggle competition identifier.
        competition_name: Human-readable competition name.
        submission_format: Expected submission format ('csv', 'json').
        required_fields: Required fields in submission.
        api_endpoint: Kaggle API endpoint for submissions.
    """

    competition_id: str = "chess-engine-leaderboard"
    competition_name: str = "Chess Engine Leaderboard"
    submission_format: str = "csv"
    required_fields: list[str] = None
    api_endpoint: str = "https://www.kaggle.com/api/v1/competitions/{competition}/submissions"

    def __post_init__(self):
        """Initialize default required fields."""
        if self.required_fields is None:
            # Default fields for chess engine benchmark
            self.required_fields = ["position_id", "best_move", "evaluation"]

    def get_submission_endpoint(self) -> str:
        """Get the submission API endpoint.

        Returns:
            Full API endpoint URL.
        """
        return self.api_endpoint.format(competition=self.competition_id)


@dataclass
class SubmissionFormat:
    """Format specification for Kaggle submissions.

    Attributes:
        format_type: Type of submission ('csv', 'json').
        headers: Column headers for CSV format.
        json_schema: JSON schema for JSON format.
    """

    format_type: str = "csv"
    headers: list[str] = None
    json_schema: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize default headers."""
        if self.headers is None:
            self.headers = ["position_id", "best_move", "evaluation", "nodes_searched", "time_ms"]

    def validate_submission_data(self, data: list[dict[str, Any]]) -> tuple[bool, str]:
        """Validate submission data format.

        Args:
            data: List of submission records.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not data:
            return False, "Submission data is empty"

        # Check all required fields are present
        for i, record in enumerate(data):
            for field in self.headers[:3]:  # Check at least first 3 required fields
                if field not in record:
                    return False, f"Record {i} missing required field: {field}"

        return True, ""


class KaggleConfig:
    """Main configuration for Kaggle integration.

    Combines credentials, competition config, and submission format.
    """

    def __init__(
        self,
        credentials_path: str | None = None,
        use_env_credentials: bool = False,
        competition_config: KaggleCompetitionConfig | None = None,
        submission_format: SubmissionFormat | None = None,
    ):
        """Initialize Kaggle configuration.

        Args:
            credentials_path: Path to kaggle.json credentials file.
            use_env_credentials: Use environment variables for credentials.
            competition_config: Competition configuration.
            submission_format: Submission format specification.
        """
        # Load credentials
        if use_env_credentials:
            self.credentials = KaggleCredentials.from_env()
        else:
            self.credentials = KaggleCredentials.from_file(credentials_path)

        # Set competition config
        self.competition = competition_config or KaggleCompetitionConfig()

        # Set submission format
        self.submission_format = submission_format or SubmissionFormat()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "competition_id": self.competition.competition_id,
            "competition_name": self.competition.competition_name,
            "submission_format": self.submission_format.format_type,
            "required_fields": self.submission_format.headers,
        }

    def get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests.

        Returns:
            Dictionary of HTTP headers for authentication.
        """
        import base64

        # Create basic auth token
        auth_string = f"{self.credentials.username}:{self.credentials.key}"
        auth_bytes = auth_string.encode("ascii")
        auth_b64 = base64.b64encode(auth_bytes).decode("ascii")

        # Note: Content-Type is intentionally not set here as it should be
        # determined by the request type (e.g., multipart/form-data for file uploads)
        return {
            "Authorization": f"Basic {auth_b64}",
        }


# Default configuration is created on demand to avoid requiring credentials at import time
DEFAULT_CONFIG = None


def get_default_config() -> KaggleConfig:
    """Get or create default configuration instance.

    Returns:
        Default KaggleConfig instance.
    """
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = KaggleConfig()
    return DEFAULT_CONFIG


def load_config(credentials_path: str | None = None, use_env: bool = False) -> KaggleConfig:
    """Load Kaggle configuration.

    Args:
        credentials_path: Path to credentials file.
        use_env: Use environment variables for credentials.

    Returns:
        KaggleConfig instance.
    """
    return KaggleConfig(credentials_path=credentials_path, use_env_credentials=use_env)
