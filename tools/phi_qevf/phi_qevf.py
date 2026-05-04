import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


def rmse(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.mean(np.abs(a - b)))


def variance(a):
    a = np.array(a)
    return float(np.var(a))


def ks_confidence(a, b, alpha=0.05):
    # KS p-value to confidence: 100 * (1 - alpha if p >= alpha else 0)
    stat, p = ks_2samp(a, b)
    return float(100 * (1 if p >= alpha else 0))


def verify(trace, ref, tol_floor=0.02):
    r = {
        "rmse": rmse(trace, ref),
        "mae": mae(trace, ref),
        "variance_pct": variance(trace) * 100.0,
        "ks_confidence_pct": ks_confidence(trace, ref),
    }
    r["adaptive_tolerance_pct"] = max(tol_floor * 100.0, np.sqrt(r["variance_pct"]))
    r["pass"] = (r["ks_confidence_pct"] >= 95.0) and (r["variance_pct"] < 5.0)
    return r


def write_signed(path, obj):
    b = json.dumps(obj, indent=2).encode("utf-8")
    Path(path).write_bytes(b)
    sig = hashlib.sha256(b).hexdigest()
    Path(str(path) + ".sha256").write_text(sig)


if __name__ == "__main__":
    try:
        n = 10000
        ref = np.sin(np.linspace(0, 100, n))
        trace = ref + np.random.normal(0, 0.003, size=n)
        res = verify(trace, ref, tol_floor=0.02)
        Path("out").mkdir(exist_ok=True)
        write_signed("out/phi_qevf_summary.json", res)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
