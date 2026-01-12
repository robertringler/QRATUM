"""
File I/O and Manifest Utilities for SEER-Medicare Pipeline

Provides utilities for:
- File discovery and validation
- SHA256 manifest generation
- Safe chunked reading
- Environment and reproducibility artifact generation

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".txt", ".dat", ".sas7bdat"}


@dataclass
class FileInfo:
    """Information about a discovered file.

    Attributes:
        path: Absolute file path
        name: File name
        size_bytes: File size in bytes
        sha256: SHA256 hash of file contents
        extension: File extension
        modified_time: Last modification time
    """

    path: str
    name: str
    size_bytes: int
    sha256: str = ""
    extension: str = ""
    modified_time: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "extension": self.extension,
            "modified_time": self.modified_time,
        }


@dataclass
class DatasetManifest:
    """Manifest of input dataset files.

    Attributes:
        created_at: Manifest creation timestamp
        files: List of FileInfo objects
        total_size_bytes: Total size of all files
        file_count: Number of files
        source_directory: Source directory path
        notes: Additional notes
    """

    created_at: str
    files: list[FileInfo] = field(default_factory=list)
    total_size_bytes: int = 0
    file_count: int = 0
    source_directory: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (safe for output, no PHI)."""
        return {
            "created_at": self.created_at,
            "files": [f.to_dict() for f in self.files],
            "total_size_bytes": self.total_size_bytes,
            "file_count": self.file_count,
            "source_directory": self.source_directory,
            "notes": self.notes,
        }

    def save_json(self, output_path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath: Path to file
        chunk_size: Chunk size for reading

    Returns:
        SHA256 hex digest
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_files(
    directory: Path,
    extensions: Optional[set[str]] = None,
    recursive: bool = True,
    compute_hashes: bool = True,
) -> list[FileInfo]:
    """Discover data files in a directory.

    Args:
        directory: Directory to scan
        extensions: Allowed file extensions (default: SUPPORTED_EXTENSIONS)
        recursive: Whether to scan recursively
        compute_hashes: Whether to compute SHA256 hashes

    Returns:
        List of FileInfo objects
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    files = []
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return files

    pattern = "**/*" if recursive else "*"
    for filepath in directory.glob(pattern):
        if not filepath.is_file():
            continue

        ext = filepath.suffix.lower()
        if ext not in extensions:
            continue

        stat = filepath.stat()
        file_hash = ""
        if compute_hashes:
            try:
                file_hash = compute_file_hash(filepath)
            except Exception as e:
                logger.warning(f"Failed to hash {filepath}: {e}")

        modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        files.append(
            FileInfo(
                path=str(filepath.absolute()),
                name=filepath.name,
                size_bytes=stat.st_size,
                sha256=file_hash,
                extension=ext,
                modified_time=modified_time,
            )
        )

    return sorted(files, key=lambda f: f.name)


def create_dataset_manifest(
    seer_dir: Optional[Path] = None,
    claims_dir: Optional[Path] = None,
    compute_hashes: bool = True,
) -> DatasetManifest:
    """Create a manifest of all input data files.

    Args:
        seer_dir: SEER data directory
        claims_dir: Medicare claims directory
        compute_hashes: Whether to compute SHA256 hashes

    Returns:
        DatasetManifest object
    """
    files = []
    source_dirs = []

    if seer_dir:
        seer_dir = Path(seer_dir)
        seer_files = discover_files(seer_dir, compute_hashes=compute_hashes)
        files.extend(seer_files)
        source_dirs.append(str(seer_dir))

    if claims_dir:
        claims_dir = Path(claims_dir)
        claims_files = discover_files(claims_dir, compute_hashes=compute_hashes)
        files.extend(claims_files)
        source_dirs.append(str(claims_dir))

    total_size = sum(f.size_bytes for f in files)

    return DatasetManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        files=files,
        total_size_bytes=total_size,
        file_count=len(files),
        source_directory="; ".join(source_dirs),
        notes="File manifest for SEER-Medicare pipeline run",
    )


def read_csv_chunked(
    filepath: Path,
    chunk_size: int = 10000,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Generator[list[dict[str, str]], None, None]:
    """Read a CSV file in chunks.

    Args:
        filepath: Path to CSV file
        chunk_size: Number of rows per chunk
        encoding: File encoding
        delimiter: Field delimiter

    Yields:
        Lists of dictionaries (rows as dicts)
    """
    with open(filepath, encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        chunk = []

        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk


def read_fixed_width(
    filepath: Path,
    column_specs: list[tuple[str, int, int]],
    skip_header: int = 0,
    encoding: str = "utf-8",
) -> Generator[dict[str, str], None, None]:
    """Read a fixed-width format file.

    Args:
        filepath: Path to file
        column_specs: List of (name, start, end) tuples (0-indexed)
        skip_header: Number of header lines to skip
        encoding: File encoding

    Yields:
        Dictionaries of column_name -> value
    """
    with open(filepath, encoding=encoding) as f:
        for _ in range(skip_header):
            next(f, None)

        for line in f:
            row = {}
            for name, start, end in column_specs:
                value = line[start:end].strip()
                row[name] = value
            yield row


def generate_environment_info() -> dict[str, Any]:
    """Generate environment information for reproducibility.

    Returns:
        Dictionary of environment details
    """
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Get pip freeze
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        info["pip_freeze"] = result.stdout.strip().split("\n") if result.returncode == 0 else []
    except Exception:
        info["pip_freeze"] = []

    return info


def get_git_commit() -> str:
    """Get current git commit hash.

    Returns:
        Git commit hash or "unknown" if not available
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def save_run_config(
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """Save run configuration to JSON.

    Args:
        output_dir: Output directory
        config: Configuration dictionary
    """
    output_path = output_dir / "run_config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)


def save_environment_info(output_dir: Path) -> None:
    """Save environment information to file.

    Args:
        output_dir: Output directory
    """
    env_info = generate_environment_info()

    # Save as text
    output_path = output_dir / "environment.txt"
    with open(output_path, "w") as f:
        f.write(f"Python Version: {env_info['python_version']}\n")
        f.write(f"Platform: {env_info['platform']}\n")
        f.write(f"Processor: {env_info['processor']}\n")
        f.write(f"Timestamp: {env_info['timestamp']}\n")
        f.write("\n--- pip freeze ---\n")
        for pkg in env_info.get("pip_freeze", []):
            f.write(f"{pkg}\n")


def save_git_commit(output_dir: Path) -> None:
    """Save git commit hash to file.

    Args:
        output_dir: Output directory
    """
    commit = get_git_commit()
    output_path = output_dir / "git_commit.txt"
    with open(output_path, "w") as f:
        f.write(commit)


def create_output_directory(base_dir: str = "artifacts") -> Path:
    """Create a timestamped output directory.

    Args:
        base_dir: Base directory for outputs

    Returns:
        Path to created output directory
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"seer_medicare_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@dataclass
class RunArtifacts:
    """Container for pipeline run artifacts.

    Attributes:
        output_dir: Output directory path
        run_config: Run configuration
        dataset_manifest: Dataset manifest
        cohort_counts: Cohort count statistics
        timeline_summary: Timeline summary statistics
        qratum_sequences: QRATUM analysis results
    """

    output_dir: Path
    run_config: dict[str, Any] = field(default_factory=dict)
    dataset_manifest: Optional[DatasetManifest] = None
    cohort_counts: dict[str, Any] = field(default_factory=dict)
    timeline_summary: dict[str, Any] = field(default_factory=dict)
    qratum_sequences: dict[str, Any] = field(default_factory=dict)

    def save_all(self) -> None:
        """Save all artifacts to output directory."""
        # Save run config
        save_run_config(self.output_dir, self.run_config)

        # Save environment
        save_environment_info(self.output_dir)

        # Save git commit
        save_git_commit(self.output_dir)

        # Save dataset manifest
        if self.dataset_manifest:
            self.dataset_manifest.save_json(self.output_dir / "dataset_manifest.json")

        # Save cohort counts
        if self.cohort_counts:
            with open(self.output_dir / "cohort_counts.json", "w") as f:
                json.dump(self.cohort_counts, f, indent=2, default=str)

        # Save timeline summary
        if self.timeline_summary:
            with open(self.output_dir / "timeline_summary.json", "w") as f:
                json.dump(self.timeline_summary, f, indent=2, default=str)

        # Save QRATUM sequences
        if self.qratum_sequences:
            with open(self.output_dir / "qratum_sequences.json", "w") as f:
                json.dump(self.qratum_sequences, f, indent=2, default=str)
