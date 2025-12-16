"""
KSG Mutual Information Estimator

Kraskov-Stögbauer-Grassberger mutual information estimator.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


class KSGEstimator:
    """
    KSG mutual information estimator.

    Uses k-nearest neighbors to estimate MI non-parametrically.
    """

    def __init__(self, k: int = 3):
        """
        Initialize KSG estimator.

        Args:
            k: Number of nearest neighbors
        """
        self.k = k

    def estimate_mi(self, data_x: np.ndarray, data_y: np.ndarray) -> float:
        """
        Estimate mutual information I(X;Y).

        Args:
            data_x: Data for variable X (n_samples, dim_x)
            data_y: Data for variable Y (n_samples, dim_y)

        Returns:
            Estimated mutual information in nats
        """
        # Reshape if needed
        if data_x.ndim == 1:
            data_x = data_x.reshape(-1, 1)
        if data_y.ndim == 1:
            data_y = data_y.reshape(-1, 1)

        n_samples = data_x.shape[0]

        # Normalize data
        data_x = (data_x - np.mean(data_x, axis=0)) / (np.std(data_x, axis=0) + 1e-10)
        data_y = (data_y - np.mean(data_y, axis=0)) / (np.std(data_y, axis=0) + 1e-10)

        # Joint space
        joint_data = np.hstack([data_x, data_y])

        # Build KD-trees
        tree_joint = cKDTree(joint_data)
        tree_x = cKDTree(data_x)
        tree_y = cKDTree(data_y)

        # Find k-nearest neighbors in joint space
        distances, _ = tree_joint.query(joint_data, k=self.k + 1)
        epsilon = distances[:, -1]

        # Count neighbors within epsilon in marginal spaces
        nx = np.array(
            [
                len(tree_x.query_ball_point(point, r=eps - 1e-15)) - 1
                for point, eps in zip(data_x, epsilon)
            ]
        )
        ny = np.array(
            [
                len(tree_y.query_ball_point(point, r=eps - 1e-15)) - 1
                for point, eps in zip(data_y, epsilon)
            ]
        )

        # KSG estimate
        mi = digamma(self.k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n_samples)

        return max(0.0, float(mi))  # MI cannot be negative
