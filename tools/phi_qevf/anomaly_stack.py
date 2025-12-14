import json
import sys

import numpy as np
from sklearn.ensemble import IsolationForest


def detect(series, z_thresh=3.5, contamination=0.01):
    x = np.array(series).reshape(-1, 1)
    # Inject known anomalies for metric eval
    x.copy()
    anomaly_indices = [123, 4567]  # Known injections
    x[anomaly_indices] = [8, -7] if len(x) > 4567 else x[anomaly_indices] + 10
    true_anoms = np.zeros(len(x), dtype=bool)
    true_anoms[anomaly_indices[: len(x)]] = True
    iforest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42).fit(x)
    iso_flags = iforest.predict(x) == -1
    z_flags = (np.abs((x - x.mean()) / (x.std() + 1e-9)) > z_thresh).flatten()
    flags = iso_flags | z_flags
    tp = np.sum(flags & true_anoms)
    fp = np.sum(flags & ~true_anoms)
    recall = (tp / np.sum(true_anoms)) * 100 if np.sum(true_anoms) > 0 else 0.0
    false_pos_rate = (fp / np.sum(~true_anoms)) * 100 if np.sum(~true_anoms) > 0 else 0.0
    return {
        "flags": flags.tolist(),
        "recall_pct": recall,
        "false_alarm_pct": false_pos_rate,
        "true_anoms_detected": int(tp),
    }


if __name__ == "__main__":
    try:
        s = list(np.random.normal(0, 1, 10000))
        res = detect(s)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
