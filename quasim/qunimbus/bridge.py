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

    def ascend(self, query: str, mode: str = "singularity", seed: int = 42) -> Dict[str, Any]:
        """Execute QuNimbus ascend operation.

        This operation generates world-model artifacts based on the query.
        All randomness is controlled by the seed parameter for reproducibility.

        Parameters
        ----------
        query : str
            Query for world-model generation
        mode : str
            Generation mode (default: "singularity")
        seed : int
            Random seed for deterministic generation

        Returns
        -------
        Dict[str, Any]
            Response containing query_id and artifact metadata

        Examples
        --------
        >>> from quasim.net.http import HttpClient
        >>> bridge = QNimbusBridge(QNimbusConfig(), HttpClient())
        >>> resp = bridge.ascend("real world simulation", seed=42)
        >>> print(resp["query_id"])
        qid-...
        """
        payload = {"query": query, "mode": mode, "seed": seed}

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
