"""Kaggle API client for QRATUM-Chess competitions.

Provides authenticated access to Kaggle competition API including:
- Competition dataset download
- Submission upload
- Competition metadata retrieval
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qratum_chess.kaggle.config import KaggleConfig


@dataclass
class CompetitionInfo:
    """Information about a Kaggle competition.

    Attributes:
        id: Competition ID
        title: Competition title
        url: Competition URL
        deadline: Submission deadline
        category: Competition category
        reward: Prize amount
        team_count: Number of participating teams
        user_has_entered: Whether user has entered
    """

    id: str
    title: str
    url: str
    deadline: str | None = None
    category: str = "getting-started"
    reward: str | None = None
    team_count: int = 0
    user_has_entered: bool = False


@dataclass
class DatasetInfo:
    """Information about downloaded competition dataset.

    Attributes:
        competition_id: Competition ID
        files: List of downloaded files
        checksum: SHA256 checksum of dataset
        download_time: Timestamp of download
        size_bytes: Total size in bytes
    """

    competition_id: str
    files: list[str]
    checksum: str
    download_time: float
    size_bytes: int


class KaggleClient:
    """Client for Kaggle competition API.

    Handles authentication, dataset management, and submissions.
    Uses Kaggle CLI under the hood for reliable API access.
    """

    def __init__(self, config: KaggleConfig | None = None):
        """Initialize Kaggle client.

        Args:
            config: Kaggle configuration (loads from file if None)
        """
        if config is None:
            try:
                config = KaggleConfig.from_file()
            except FileNotFoundError:
                try:
                    config = KaggleConfig.from_env()
                except ValueError:
                    raise FileNotFoundError(
                        "No Kaggle credentials found. Please either:\n"
                        "1. Create ~/.kaggle/kaggle.json with your API credentials\n"
                        "2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
                        "Get your API key from: https://www.kaggle.com/settings/account"
                    )

        self.config = config
        self._setup_environment()
        self._verify_kaggle_cli()

    def _setup_environment(self) -> None:
        """Setup environment variables for Kaggle CLI."""
        os.environ["KAGGLE_USERNAME"] = self.config.username
        os.environ["KAGGLE_KEY"] = self.config.key

        if self.config.proxy:
            os.environ["KAGGLE_PROXY"] = self.config.proxy

        if not self.config.ssl_verify:
            os.environ["KAGGLE_SSL_VERIFY"] = "false"

    def _verify_kaggle_cli(self) -> None:
        """Verify Kaggle CLI is installed and working."""
        try:
            result = subprocess.run(
                ["kaggle", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Kaggle CLI not working properly")
        except FileNotFoundError:
            raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Kaggle CLI timeout")

    def get_competition_info(self, competition_id: str) -> CompetitionInfo:
        """Get information about a competition.

        Args:
            competition_id: Competition identifier

        Returns:
            Competition information
        """
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "list", "-s", competition_id],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to get competition info: {result.stderr}")

            # Parse competition info from output
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                raise ValueError(f"Competition not found: {competition_id}")

            # Simple parsing (header + data row)
            parts = lines[1].split()

            return CompetitionInfo(
                id=competition_id,
                title=" ".join(parts[1:-5]) if len(parts) > 6 else competition_id,
                url=f"https://www.kaggle.com/c/{competition_id}",
                deadline=parts[-4] if len(parts) > 4 else None,
                category=parts[-3] if len(parts) > 3 else "getting-started",
                reward=parts[-2] if len(parts) > 2 else None,
                team_count=int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0,
                user_has_entered=False,
            )
        except Exception:
            # Return minimal info if query fails
            return CompetitionInfo(
                id=competition_id,
                title=competition_id,
                url=f"https://www.kaggle.com/c/{competition_id}",
            )

    def download_competition_data(
        self, competition_id: str, output_dir: str | Path, force: bool = False
    ) -> DatasetInfo:
        """Download competition dataset.

        Args:
            competition_id: Competition identifier
            output_dir: Output directory for dataset
            force: Force re-download if already exists

        Returns:
            Dataset information
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading competition data: {competition_id}")
        print(f"Output directory: {output_path}")

        try:
            # Download using Kaggle CLI
            cmd = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition_id,
                "-p",
                str(output_path),
            ]
            if force:
                cmd.append("-f")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Download failed: {result.stderr}")

            print(result.stdout)

            # List downloaded files
            files = list(output_path.glob("*"))
            file_list = [str(f.name) for f in files if f.is_file()]

            # Calculate dataset checksum
            checksum = self._calculate_dataset_checksum(output_path)

            # Calculate total size
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            dataset_info = DatasetInfo(
                competition_id=competition_id,
                files=file_list,
                checksum=checksum,
                download_time=time.time(),
                size_bytes=total_size,
            )

            # Save dataset metadata
            metadata_file = output_path / "dataset_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(
                    {
                        "competition_id": dataset_info.competition_id,
                        "files": dataset_info.files,
                        "checksum": dataset_info.checksum,
                        "download_time": dataset_info.download_time,
                        "size_bytes": dataset_info.size_bytes,
                    },
                    f,
                    indent=2,
                )

            print(f"✓ Downloaded {len(file_list)} files ({total_size / 1024 / 1024:.2f} MB)")

            return dataset_info

        except subprocess.TimeoutExpired:
            raise RuntimeError("Download timeout (5 minutes exceeded)")
        except Exception as e:
            raise RuntimeError(f"Download error: {e}")

    def submit_competition(
        self, competition_id: str, submission_file: str | Path, message: str = ""
    ) -> dict[str, Any]:
        """Submit predictions to competition.

        Args:
            competition_id: Competition identifier
            submission_file: Path to submission CSV
            message: Optional submission message

        Returns:
            Submission result information
        """
        submission_path = Path(submission_file)

        if not submission_path.exists():
            raise FileNotFoundError(f"Submission file not found: {submission_path}")

        print(f"Submitting to competition: {competition_id}")
        print(f"Submission file: {submission_path}")

        try:
            cmd = [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                competition_id,
                "-f",
                str(submission_path),
            ]
            if message:
                cmd.extend(["-m", message])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                raise RuntimeError(f"Submission failed: {result.stderr}")

            print(result.stdout)

            return {
                "success": True,
                "competition_id": competition_id,
                "file": str(submission_path),
                "message": message,
                "timestamp": time.time(),
                "output": result.stdout,
            }

        except subprocess.TimeoutExpired:
            raise RuntimeError("Submission timeout")
        except Exception as e:
            raise RuntimeError(f"Submission error: {e}")

    def list_submissions(self, competition_id: str) -> list[dict[str, Any]]:
        """List user's submissions for a competition.

        Args:
            competition_id: Competition identifier

        Returns:
            List of submission information
        """
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "submissions", competition_id],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return []

            # Parse submission list
            submissions = []
            lines = result.stdout.strip().split("\n")

            # Skip header
            for line in lines[1:]:
                if line.strip():
                    submissions.append({"raw": line})

            return submissions

        except Exception:
            return []

    def _calculate_dataset_checksum(self, directory: Path) -> str:
        """Calculate SHA256 checksum of all files in directory.

        Args:
            directory: Directory to checksum

        Returns:
            SHA256 hex digest
        """
        sha256 = hashlib.sha256()

        for file in sorted(directory.glob("*")):
            if file.is_file() and file.suffix != ".json":
                with open(file, "rb") as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)

        return sha256.hexdigest()
