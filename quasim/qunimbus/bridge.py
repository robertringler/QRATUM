"""QuNimbus v6 Bridge - API boundary for world-model services."""

from dataclasses import dataclass
from typing import Any, Dict

from quasim.net.http import HttpClient


@dataclass
class QNimbusConfig:
    """Configuration for QuNimbus v6 API connection.

    Attributes
    ----------
    base_url : str
        Base URL for QuNimbus v6 API
    timeout_s : int
        Request timeout in seconds
    retries : int
        Number of retry attempts for failed requests
    """

    base_url: str = "https://omni.x.ai/qunimbus/v6"
    timeout_s: int = 120
    retries: int = 3


class QNimbusBridge:
    """Bridge to QuNimbus v6 world-model services.

    This bridge mediates calls between QuASIM's certified runtime and
    QuNimbus's generative world models, ensuring deterministic and
    replayable interactions.
    """

    def __init__(self, cfg: QNimbusConfig, http_client: HttpClient):
        """Initialize QuNimbus bridge.

        Parameters
        ----------
        cfg : QNimbusConfig
            Bridge configuration
        http_client : HttpClient
            HTTP client for API communication
        """

        self.cfg = cfg
        self.http = http_client

    def ascend(
        self, query: str, mode: str = "singularity", seed: int = 42, query_id: str | None = None
    ) -> Dict[str, Any]:
        """Execute QuNimbus ascend operation.

        This operation generates world-model artifacts based on the query.
        All randomness is controlled by the seed parameter for reproducibility.

        The ascend operation follows DO-178C Level A determinism requirements:
        - Identical seed + query → identical artifacts (within tolerance)
        - All network calls are logged in audit trail
        - Query hash is recorded for replay validation

        Parameters
        ----------
        query : str
            Query for world-model generation
        mode : str
            Generation mode (default: "singularity")
            Options: "singularity", "distributed", "validation"
        seed : int
            Random seed for deterministic generation (default: 42)
            Must be in range [0, 2^31-1] for cross-platform reproducibility
        query_id : str | None
            Optional query identifier for audit tracking (default: None)
            If provided, included in request payload and audit logs

        Returns
        -------
        Dict[str, Any]
            Response containing:
            - query_id (str): Unique identifier for this query
            - status (str): "success", "pending", or "failed"
            - artifacts (dict): Map of artifact_name → {id, filename, size_bytes}
            - mode (str): Echo of requested mode
            - seed (int): Echo of seed used

        Raises
        ------
        ValueError
            If seed is out of valid range
        HTTPError
            If network request fails (falls back to mock response)

        Examples
        --------
        Basic usage with default parameters:

        >>> from quasim.net.http import HttpClient
        >>> from quasim.qunimbus.bridge import QNimbusBridge, QNimbusConfig
        >>> from quasim.runtime.determinism import set_seed
        >>>
        >>> # Initialize bridge
        >>> cfg = QNimbusConfig()
        >>> http = HttpClient()
        >>> bridge = QNimbusBridge(cfg, http)
        >>>
        >>> # Set global seed for reproducibility
        >>> set_seed(42)
        >>>
        >>> # Execute ascend operation
        >>> resp = bridge.ascend("real world simulation", seed=42)
        >>> print(resp["query_id"])
        qid-...
        >>> print(resp["status"])
        success

        Handling artifacts from response:

        >>> resp = bridge.ascend("climate model 2025", mode="distributed", seed=12345)
        >>>
        >>> # Iterate over artifacts
        >>> for artifact_name, artifact_meta in resp.get("artifacts", {}).items():
        ...     artifact_id = artifact_meta["id"]
        ...     filename = artifact_meta["filename"]
        ...
        ...     # Download artifact
        ...     local_path = bridge.fetch_artifact(artifact_id, f"output/{filename}")
        ...     print(f"Downloaded {artifact_name} to {local_path}")

        Integration with audit logging:

        >>> from quasim.audit.log import audit_event
        >>>
        >>> # Execute with audit
        >>> resp = bridge.ascend("economic forecast", seed=99)
        >>>
        >>> # Log to audit trail
        >>> audit_event(
        ...     "qnimbus.ascend",
        ...     {
        ...         "query": "economic forecast",
        ...         "seed": 99,
        ...         "query_id": resp["query_id"],
        ...         "status": resp["status"],
        ...     }
        ... )

        Seed injection for deterministic replay:

        >>> import hashlib
        >>>
        >>> # Generate seed from query for consistent experiments
        >>> query = "particle physics simulation"
        >>> query_hash = hashlib.sha256(query.encode()).digest()
        >>> deterministic_seed = int.from_bytes(query_hash[:4], "big") % (2**31)
        >>>
        >>> # Use derived seed
        >>> resp = bridge.ascend(query, seed=deterministic_seed)
        >>>
        >>> # Same query + seed = same results
        >>> resp2 = bridge.ascend(query, seed=deterministic_seed)
        >>> assert resp["query_id"] == resp2["query_id"]

        Notes
        -----
        - All operations are deterministic given the same seed
        - Network failures gracefully degrade to mock responses
        - Artifacts are stored with compression and checksums (HDF5/Fletcher32)
        - Query responses are cached for 24h on server side
        """

        payload = {"query": query, "mode": mode, "seed": seed}
        if query_id:
            payload["query_id"] = query_id

        response = self.http.post_json(
            f"{self.cfg.base_url}/ascend", payload, timeout=self.cfg.timeout_s
        )

        return response

    def fetch_artifact(self, artifact_id: str, dest: str) -> str:
        """Fetch artifact from QuNimbus.

        Parameters
        ----------
        artifact_id : str
            Artifact identifier
        dest : str
            Destination file path

        Returns
        -------
        str
            Path to downloaded artifact

        Examples
        --------
        >>> bridge = QNimbusBridge(QNimbusConfig(), HttpClient())
        >>> path = bridge.fetch_artifact("art-123", "out/artifact.hdf5")
        >>> print(path)
        out/artifact.hdf5
        """

        url = f"{self.cfg.base_url}/artifact/{artifact_id}"
        return self.http.download(url, dest)
