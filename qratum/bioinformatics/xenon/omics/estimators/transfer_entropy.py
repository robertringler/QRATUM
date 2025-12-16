"""
Transfer Entropy Estimator

Measures directed information flow between time series.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import numpy as np


class TransferEntropyEstimator:
    """
    Transfer entropy estimator for time series.
    
    Quantifies information flow from source to target.
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize transfer entropy estimator.
        
        Args:
            k: Number of nearest neighbors for estimation
        """
        self.k = k
    
    def compute_te(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Compute transfer entropy from source to target.
        
        TE(X→Y) = I(Y_t ; X_{t-lag} | Y_{t-1})
        
        Args:
            source: Source time series
            target: Target time series
            lag: Time lag for transfer
            
        Returns:
            Transfer entropy in nats
        """
        # Construct lagged variables
        n = len(target)
        
        if n < lag + 2:
            return 0.0
        
        # Y_t (target present)
        y_present = target[lag + 1:].reshape(-1, 1)
        
        # Y_{t-1} (target past)
        y_past = target[lag:n-1].reshape(-1, 1)
        
        # X_{t-lag} (source past)
        x_past = source[:n-lag-1].reshape(-1, 1)
        
        # Compute conditional MI: I(Y_t ; X_{t-lag} | Y_{t-1})
        # TE = I(Y_t, X_{t-lag}, Y_{t-1}) - I(Y_t, Y_{t-1})
        
        from .ksg import KSGEstimator
        ksg = KSGEstimator(k=self.k)
        
        # Joint MI with source
        joint_with_source = np.hstack([y_present, x_past, y_past])
        mi_with_source = ksg.estimate_mi(
            joint_with_source[:, :1],  # y_present
            joint_with_source[:, 1:]   # x_past, y_past
        )
        
        # MI without source
        joint_without_source = np.hstack([y_present, y_past])
        mi_without_source = ksg.estimate_mi(
            joint_without_source[:, :1],  # y_present
            joint_without_source[:, 1:]   # y_past
        )
        
        te = mi_with_source - mi_without_source
        
        return max(0.0, float(te))
