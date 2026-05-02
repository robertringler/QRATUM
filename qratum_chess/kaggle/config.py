"""Kaggle configuration management for QRATUM-Chess.

Handles secure loading of Kaggle API credentials from ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class KaggleConfig:
    """Kaggle API configuration.

    Attributes:
        username: Kaggle username
        key: Kaggle API key
        proxy: Optional HTTP proxy
        ssl_verify: Whether to verify SSL certificates
    """

    username: str
    key: str
    proxy: str | None = None
    ssl_verify: bool = True

    @classmethod
    def from_file(cls, config_path: str | Path | None = None) -> KaggleConfig:
        """Load configuration from kaggle.json file.

        Args:
            config_path: Path to kaggle.json (default: ~/.kaggle/kaggle.json)

        Returns:
            KaggleConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if config_path is None:
            config_path = Path.home() / ".kaggle" / "kaggle.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Kaggle config not found at {config_path}. "
                "Please create ~/.kaggle/kaggle.json with your API credentials. "
                "Get your key from: https://www.kaggle.com/settings/account"
            )

        # Check file permissions (should be 600)
        if os.name != "nt":  # Unix-like systems
            stat_info = config_path.stat()
            if stat_info.st_mode & 0o077:
                print(
                    f"Warning: {config_path} has insecure permissions. "
                    f"Run: chmod 600 {config_path}"
                )

        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")

        if "username" not in data or "key" not in data:
            raise ValueError("Invalid Kaggle config. Must contain 'username' and 'key' fields.")

        return cls(
            username=data["username"],
            key=data["key"],
            proxy=data.get("proxy"),
            ssl_verify=data.get("ssl_verify", True),
        )

    @classmethod
    def from_env(cls) -> KaggleConfig:
        """Load configuration from environment variables.

        Returns:
            KaggleConfig instance

        Raises:
            ValueError: If required env vars are not set
        """
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")

        if not username or not key:
            raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set")

        return cls(
            username=username,
            key=key,
            proxy=os.environ.get("KAGGLE_PROXY"),
            ssl_verify=os.environ.get("KAGGLE_SSL_VERIFY", "true").lower() == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes sensitive key).

        Returns:
            Config dictionary with masked key
        """
        return {
            "username": self.username,
            "key": "***" if self.key else None,
            "proxy": self.proxy,
            "ssl_verify": self.ssl_verify,
        }
