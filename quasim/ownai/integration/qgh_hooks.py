"""QGH causal history hooks for run tracking."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class QGHLedger:
    """QGH causal history ledger for tracking runs.

    Maintains an append-only log of run metadata and prediction hashes.

    Attributes
    ----------
    ledger_path : Path
        Path to the ledger file
    """

    def __init__(self, ledger_path: Path | None = None):
        if ledger_path is None:
            ledger_path = Path("runs/.qgh_ledger.jsonl")

        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def append_run(
        self,
        run_id: str,
        model: str,
        task: str,
        dataset: str,
        seed: int,
        prediction_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a run record to the ledger.

        Parameters
        ----------
        run_id : str
            Unique run identifier
        model : str
            Model name
        task : str
            Task name
        dataset : str
            Dataset name
        seed : int
            Random seed
        prediction_hash : str
            Hash of predictions
        metadata : dict, optional
            Additional metadata
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "model": model,
            "task": task,
            "dataset": dataset,
            "seed": seed,
            "prediction_hash": prediction_hash,
            "metadata": metadata or {},
        }

        # Append to ledger
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_ledger(self) -> list[dict[str, Any]]:
        """Read all records from the ledger.

        Returns
        -------
        list[dict]
            List of run records
        """
        if not self.ledger_path.exists():
            return []

        records = []
        with open(self.ledger_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        return records

    def verify_consensus(
        self,
        model: str,
        task: str,
        dataset: str,
        seed: int,
    ) -> bool:
        """Verify consensus for a specific configuration.

        Parameters
        ----------
        model : str
            Model name
        task : str
            Task name
        dataset : str
            Dataset name
        seed : int
            Random seed

        Returns
        -------
        bool
            True if all matching runs have the same prediction hash
        """
        records = self.read_ledger()

        # Filter matching records
        matching = [
            r
            for r in records
            if r["model"] == model
            and r["task"] == task
            and r["dataset"] == dataset
            and r["seed"] == seed
        ]

        if len(matching) < 2:
            return True  # Not enough data to compare

        # Check if all hashes are identical
        hashes = [r["prediction_hash"] for r in matching]
        return len(set(hashes)) == 1


def create_run_id(model: str, task: str, dataset: str, seed: int) -> str:
    """Create a unique run identifier.

    Parameters
    ----------
    model : str
        Model name
    task : str
        Task name
    dataset : str
        Dataset name
    seed : int
        Random seed

    Returns
    -------
    str
        Unique run ID
    """
    timestamp = datetime.now().isoformat()
    content = f"{model}_{task}_{dataset}_{seed}_{timestamp}"
    run_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"run_{run_hash}"
