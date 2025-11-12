import hashlib
import json
import os
import sys
import time
from pathlib import Path

import zstandard as zstd

INTERVAL = int(os.getenv("ORD_INTERVAL_SEC", "30"))
CHECKPOINT_H = int(os.getenv("ORD_CHECKPOINT_HOURS", "6"))

out = Path("ord_out")
out.mkdir(exist_ok=True)


def write(doc, name):
    raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    cctx = zstd.ZstdCompressor(level=9)
    comp = cctx.compress(raw)
    p = out / f"{name}.zst"
    p.write_bytes(comp)
    sig = hashlib.sha256(comp).hexdigest()
    (out / f"{name}.sha256").write_text(sig)


def run(bounded_loops=10):
    t0 = time.time()
    i = 0
    while i < bounded_loops:
        doc = {"ts": time.time(), "metrics": {"rmse": 0.42, "mae": 0.31}}
        write(doc, f"m_{i:06d}")
        if (time.time() - t0) >= CHECKPOINT_H * 3600:
            write({"checkpoint": True, "ts": time.time()}, "checkpoint")
            t0 = time.time()  # Reset for next
        time.sleep(INTERVAL)
        i += 1


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
